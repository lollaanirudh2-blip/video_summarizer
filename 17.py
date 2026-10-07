"""
Media Summarizer (Doctor's view) - video/link -> AssemblyAI transcript (with timestamps) -> LLM summary by topic.
Run:  python 10.py   then open http://127.0.0.1:5000
Install:  pip install flask yt-dlp requests python-dotenv

.env (same folder as this script):
    ASSEMBLYAI_API_KEY=your-assemblyai-key
    DEEPSEEK_API_KEY=sk-your-deepseek-key      (optional)

Summary engines, tried in order: AssemblyAI LLM Gateway -> DeepSeek -> Ollama (local) -> built-in basic summarizer.
"""
import base64, glob, ipaddress, json, mimetypes, os, re, shutil, socket, subprocess, sys, tempfile, threading, time, uuid
import urllib.error, urllib.request
from urllib.parse import urlparse, urljoin
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv

ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(ENV_PATH)

from flask import Flask, jsonify, make_response, request, send_file
import yt_dlp

# ======================= SETTINGS =======================
LANGUAGE = "en"                     # "en", "hi", ... or None to auto-detect
COOKIE_BROWSERS = ["edge", "chrome", "firefox"]
MAX_VIDEO_HEIGHT = 360               # lower = faster download (360p is plenty for transcription + preview)

ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
ASSEMBLYAI_URL = "https://api.assemblyai.com"
ASSEMBLYAI_MODELS = ["universal-3-pro", "universal-2"]
TRANSCRIBE_TIMEOUT = 3600
AAI_LLM_GATEWAY_URL = "https://llm-gateway.assemblyai.com/v1/chat/completions"
AAI_LLM_MODELS = ["claude-sonnet-4-6", "claude-sonnet-4.6", "qwen3.5-4b-32k-fast"]

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
MAX_TRANSCRIPT_CHARS = 100000

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.2:3b"
MAX_LOCAL_CHARS = 24000
PARALLEL_REQUESTS = 3               # long videos are cut into parts that are summarized at the same time
# ========================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 ** 3
WORK_DIR = Path(tempfile.gettempdir()) / "media_summarizer"
WORK_DIR.mkdir(exist_ok=True)
JOBS = {}
SYSTEM = "You are an experienced physician. You reply with valid JSON only."


# ---------- helpers ----------
def find_deno():
    found = shutil.which("deno")
    if found:
        return found
    pattern = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages", "DenoLand.Deno*", "deno.exe")
    hits = glob.glob(pattern)
    fallback = Path.home() / ".deno" / "bin" / "deno.exe"
    return hits[0] if hits else (str(fallback) if fallback.exists() else None)


def clean_error(msg) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", str(msg)).strip()


def prune_old_jobs(max_age=3600):
    for jid in [j for j, v in list(JOBS.items()) if time.time() - v["created"] > max_age]:
        job = JOBS.pop(jid, None)
        if job and job.get("media_path"):
            Path(job["media_path"]).unlink(missing_ok=True)


def update(jid, **kw):
    JOBS[jid].update(kw)


def mmss(sec) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ---------- 1. download (video is kept so the page can play it) ----------
MAX_DOWNLOAD = 2 * 1024 ** 3
UA = {"User-Agent": "Mozilla/5.0 (compatible; MediaSummarizer)"}


def check_public_url(url: str):
    """Only http/https links to public servers (blocks localhost / private network addresses)."""
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise RuntimeError("Please paste a full link starting with http:// or https://")
    try:
        for info in socket.getaddrinfo(u.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise RuntimeError("That link points to a private address and cannot be downloaded.")
    except socket.gaierror:
        raise RuntimeError("Could not find that website. Check the link.")


def classify_link(url: str):
    """Returns (kind, value): drive -> file id, direct -> url to fetch as a file, other -> let yt-dlp handle it."""
    u = urlparse(url)
    host = (u.hostname or "").lower()
    if host.endswith("drive.google.com") or host.endswith("docs.google.com") or host.endswith("drive.usercontent.google.com"):
        m = re.search(r"/d/([\w-]{10,})", u.path) or re.search(r"[?&]id=([\w-]{10,})", url)
        if m:
            return "drive", m.group(1)
        if "/folders/" in u.path:
            raise RuntimeError("That is a Google Drive FOLDER link. Open the video, click Share > Copy link, and paste that file link.")
    if host.endswith("dropbox.com"):
        return "direct", re.sub(r"([?&])dl=0", r"\1dl=1", url) if "dl=" in url else url + ("&" if "?" in url else "?") + "dl=1"
    if host in ("1drv.ms", "onedrive.live.com") or host.endswith(".sharepoint.com"):
        if host.endswith(".sharepoint.com"):
            return "direct", url + ("&" if "?" in url else "?") + "download=1"
        enc = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
        return "direct", f"https://api.onedrive.com/v1.0/shares/u!{enc}/root/content"
    if re.search(r"\.(mp4|mov|m4v|webm|mkv|avi|mp3|m4a|wav|aac|ogg|flac)(\?|#|$)", u.path + "?", re.I):
        return "direct", url
    return "other", url


def save_stream(r, on_stage):
    """Stream an HTTP response to a file in WORK_DIR. Returns (path, display name)."""
    cd = r.headers.get("Content-Disposition", "")
    m = re.search(r"filename\*=UTF-8''([^;]+)|filename=\"?([^\";]+)", cd)
    from urllib.parse import unquote
    name = unquote((m.group(1) or m.group(2)).strip()) if m else Path(urlparse(r.url).path).name
    ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
    ext = Path(name).suffix.lower()
    if ext not in (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac"):
        ext = {"video/quicktime": ".mov", "video/webm": ".webm", "audio/mpeg": ".mp3", "audio/mp4": ".m4a",
               "audio/wav": ".wav", "audio/x-wav": ".wav"}.get(ctype) or mimetypes.guess_extension(ctype) or ".mp4"
        if ext in (".bin", ".html", ".htm"):
            ext = ".mp4"
    out, size, last = WORK_DIR / f"{uuid.uuid4().hex}{ext}", 0, 0
    try:
        with open(out, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                size += len(chunk)
                if size > MAX_DOWNLOAD:
                    raise RuntimeError("The file is larger than 2 GB.")
                f.write(chunk)
                if time.time() - last > 1:
                    last = time.time()
                    on_stage(f"Downloading from link... {size / 1048576:.0f} MB")
    except Exception:
        out.unlink(missing_ok=True)
        raise
    return out, (Path(name).stem if name else "video") or "video"


def fetch_drive(fid: str, on_stage):
    sess = requests.Session()
    sess.headers.update(UA)
    base = "https://drive.usercontent.google.com/download"
    r = sess.get(base, params={"id": fid, "export": "download", "confirm": "t"}, stream=True, timeout=60)
    for _ in range(2):
        if "text/html" not in r.headers.get("Content-Type", ""):
            break
        html = r.text   # virus-scan page for big files has a hidden form we can submit
        form = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', html))
        if "id" in form and ("confirm" in form or "uuid" in form):
            r = sess.get(base, params=form, stream=True, timeout=60)
        else:
            break
    if r.status_code != 200 or "text/html" in r.headers.get("Content-Type", ""):
        raise RuntimeError("Google Drive would not give the file. Open it in Drive > Share > General access > "
                           "'Anyone with the link', then paste the link again.")
    return save_stream(r, on_stage)


def fetch_direct(url: str, on_stage):
    sess = requests.Session()
    sess.headers.update(UA)
    r = sess.get(url, stream=True, timeout=60, allow_redirects=False)
    for _ in range(6):                       # follow redirects ourselves so every hop is checked
        if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
            nxt = urljoin(r.url, r.headers["Location"])
            check_public_url(nxt)
            r = sess.get(nxt, stream=True, timeout=60, allow_redirects=False)
        else:
            break
    if r.status_code != 200:
        raise RuntimeError(f"The link returned HTTP {r.status_code}. Make sure it is shared publicly.")
    if "text/html" in r.headers.get("Content-Type", ""):
        raise RuntimeError("That link opens a web page, not a video file. Share the file with 'Anyone with the link'.")
    return save_stream(r, on_stage)


def download_ytdlp(url: str, on_stage=lambda s: None):
    cookie_file = Path(__file__).with_name("cookies.txt")
    # Try WITHOUT cookies first - fast, and works for almost every public video.
    # Only fall back to cookies when the error looks like a sign-in/bot block.
    attempts = [{}]
    if cookie_file.exists():
        attempts.append({"cookiefile": str(cookie_file)})
    attempts += [{"cookiesfrombrowser": (b,)} for b in COOKIE_BROWSERS]

    # Tried in order. Separate video+audio streams are merged with ffmpeg; the last choice is audio only.
    formats = [
        f"bv*[height<={MAX_VIDEO_HEIGHT}]+ba/b[height<={MAX_VIDEO_HEIGHT}]",
        "bv*+ba/b",
        "b",
        "ba/b",
    ]

    deno, last = find_deno(), ""
    try:
        ffmpeg_path = get_ffmpeg()
    except Exception:
        ffmpeg_path = None

    def hook(d):
        if d.get("status") == "downloading":
            pct = (d.get("_percent_str") or "").strip()
            speed = (d.get("_speed_str") or "").strip()
            eta = (d.get("_eta_str") or "").strip()
            on_stage(f"Downloading video... {pct} {speed} ETA {eta}".strip())
        elif d.get("status") == "finished":
            on_stage("Download finished, processing...")

    for extra in attempts:
        needs_auth = False
        for fmt in formats:
            opts = {
                "format": fmt,
                "quiet": True, "noplaylist": True,
                "outtmpl": str(WORK_DIR / f"{uuid.uuid4().hex}.%(ext)s"),
                "merge_output_format": "mp4",
                "concurrent_fragment_downloads": 8,
                "socket_timeout": 20,
                "progress_hooks": [hook],
                **extra,
            }
            if ffmpeg_path:
                opts["ffmpeg_location"] = ffmpeg_path
            if deno:
                opts["js_runtimes"] = {"deno": {"path": deno}}
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    path = Path(ydl.prepare_filename(info))
                    if not path.exists():                      # after merging the extension can change
                        alt = path.with_suffix(".mp4")
                        path = alt if alt.exists() else path
                    return path, str(info.get("title") or "video")
            except Exception as e:
                last = clean_error(e)
                if "not a bot" in last or "Sign in" in last or "private video" in last.lower():
                    needs_auth = True
                    break                                      # try the next cookie source
                if "format is not available" in last.lower():
                    continue                                   # try the next format choice
                raise RuntimeError(last)                       # some other real error: stop
        if not needs_auth and "format is not available" not in last.lower():
            break
    if "not a bot" in last or "Sign in" in last:
        raise RuntimeError("YouTube blocked this download (bot check). Sign in to YouTube in Edge/Chrome, "
                           "add a cookies.txt file, or download the video and use Choose video.")
    raise RuntimeError(last or "Download failed.")


def get_ffmpeg():
    """Use ffmpeg already on this PC, or install a small bundled copy automatically."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "imageio-ffmpeg"])
        import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(path: Path, on_stage=lambda s: None) -> Path:
    """Pull just the audio track out before uploading - much smaller & faster than sending the whole video."""
    out = WORK_DIR / f"{uuid.uuid4().hex}.m4a"
    try:
        ff = get_ffmpeg()
        on_stage("Preparing audio...")
        subprocess.run(
            [ff, "-y", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "aac", "-b:a", "48k", str(out)],
            check=True, timeout=1800,
        )
        if out.exists() and out.stat().st_size > 1024:
            return out
    except Exception:
        pass  # if anything goes wrong, just upload the original file instead
    out.unlink(missing_ok=True)
    return path


def download_media(url: str, on_stage=lambda s: None):
    """Link -> (local file path, title). Handles Google Drive, Dropbox, OneDrive, direct files, YouTube and more."""
    check_public_url(url)
    kind, val = classify_link(url)
    if kind == "drive":
        on_stage("Downloading from Google Drive...")
        return fetch_drive(val, on_stage)
    if kind == "direct":
        on_stage("Downloading from link...")
        try:
            return fetch_direct(val, on_stage)
        except RuntimeError:
            if urlparse(url).hostname and any(h in urlparse(url).hostname for h in ("dropbox", "1drv", "onedrive", "sharepoint")):
                raise
    on_stage("Downloading video...")
    return download_ytdlp(url, on_stage)


# ---------- 2. transcribe (AssemblyAI) with timestamps ----------
def aai_error(resp) -> str:
    if resp.status_code in (401, 403):
        return "AssemblyAI rejected the API key. Check ASSEMBLYAI_API_KEY in your .env file."
    try:
        detail = resp.json().get("error", "")
    except Exception:
        detail = resp.text[:200]
    return f"AssemblyAI error {resp.status_code}: {detail}"


def transcribe(path: Path, on_stage=lambda s: None):
    """Returns (full_text, sentences[(seconds, text)])."""
    if not ASSEMBLYAI_API_KEY:
        raise RuntimeError(f"ASSEMBLYAI_API_KEY is missing. Add it to {ENV_PATH} and restart the app.")
    headers = {"authorization": ASSEMBLYAI_API_KEY}

    on_stage("Uploading to AssemblyAI...")
    with open(path, "rb") as f:
        r = requests.post(ASSEMBLYAI_URL + "/v2/upload", headers=headers, data=f, timeout=1800)
    if r.status_code != 200:
        raise RuntimeError(aai_error(r))

    body = {"audio_url": r.json()["upload_url"], "speech_models": ASSEMBLYAI_MODELS}
    if LANGUAGE:
        body["language_code"] = LANGUAGE
    else:
        body["language_detection"] = True
    r = requests.post(ASSEMBLYAI_URL + "/v2/transcript", headers=headers, json=body, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(aai_error(r))
    tid = r.json()["id"]

    started = time.time()
    while time.time() - started < TRANSCRIBE_TIMEOUT:
        r = requests.get(f"{ASSEMBLYAI_URL}/v2/transcript/{tid}", headers=headers, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(aai_error(r))
        data = r.json()
        if data.get("status") == "completed":
            text = (data.get("text") or "").strip()
            sents = []
            try:
                rs = requests.get(f"{ASSEMBLYAI_URL}/v2/transcript/{tid}/sentences", headers=headers, timeout=60)
                if rs.status_code == 200:
                    sents = [(s["start"] / 1000.0, s["text"].strip()) for s in rs.json().get("sentences", []) if s.get("text")]
            except Exception:
                pass
            return text, sents or [(0.0, text)]
        if data.get("status") == "error":
            raise RuntimeError(f"AssemblyAI transcription failed: {data.get('error', 'unknown error')}")
        on_stage(f"Transcribing audio ({data.get('status')}, {int(time.time() - started)}s)...")
        time.sleep(3)
    raise RuntimeError("Transcription timed out.")


# ---------- 3. summarizers ----------
def build_prompt(sents, lang="") -> str:
    lines = "\n".join(f"[{int(t)}] {s}" for t, s in sents)[:MAX_TRANSCRIPT_CHARS]
    lang_rule = (f"write ALL text values (overview, titles, points, takeaways) in {lang}" if lang
                 else "write in the language of the transcript") + "."
    return (
        "Below is a timestamped transcript of a video. Each line starts with [seconds].\n"
        "Write the summary from a DOCTOR'S point of view: explain clinical/health relevance, point out risks, "
        "inaccuracies or myths, and give practical medical advice. If the content is not medical, comment on "
        "any health or wellbeing implications.\n"
        "Split it BY TOPIC: every time the subject of discussion changes, start a new topic. "
        "'start_seconds' must be the [seconds] value of the first line of that topic.\n"
        "Reply with JSON ONLY (no markdown fences, no commentary):\n"
        '{"overview": "2-3 sentence overview in a doctor\'s voice",\n'
        ' "topics": [{"start_seconds": 0, "title": "Short topic title", "points": ["clear point", "..."]}],\n'
        ' "takeaways": ["key clinical takeaway", "..."]}\n'
        "Rules: 2-4 concise points per topic; wrap the 1-3 most important keywords of each point in **double asterisks**; "
        "3-5 takeaways; " + lang_rule + " Keep it compact. Never use double quotes inside text "
        "(use single quotes instead) and never put line breaks inside a string.\n\nTRANSCRIPT:\n" + lines)


def build_chunk_prompt(sents, lang="") -> str:
    """Prompt for ONE part of a long video (topics only; overview/takeaways are written afterwards)."""
    lines = "\n".join(f"[{int(t)}] {s}" for t, s in sents)
    lang_rule = (f"write ALL text values (titles, points) in {lang}" if lang
                 else "write in the language of the transcript") + "."
    return (
        "Below is ONE PART of a longer timestamped video transcript. Each line starts with [seconds].\n"
        "Write the summary from a DOCTOR'S point of view: explain clinical/health relevance, point out risks, "
        "inaccuracies or myths, and give practical medical advice. If the content is not medical, comment on "
        "any health or wellbeing implications.\n"
        "Split this part BY TOPIC: every time the subject changes, start a new topic. "
        "'start_seconds' must be the [seconds] value of the first line of that topic.\n"
        "Reply with JSON ONLY (no markdown fences, no commentary):\n"
        '{"topics": [{"start_seconds": 0, "title": "Short topic title", "points": ["clear point", "..."]}]}\n'
        "Rules: 2-4 concise points per topic; wrap the 1-3 most important keywords of each point in **double asterisks**; "
        + lang_rule + " Keep it compact. Never use double quotes inside text "
        "(use single quotes instead) and never put line breaks inside a string.\n\nTRANSCRIPT PART:\n" + lines)


def build_final_prompt(topics, lang="") -> str:
    """Short prompt: overview + takeaways written from the topic notes of the whole video."""
    lang_rule = (f"write ALL text values in {lang}" if lang else "write in the same language as the notes") + "."
    body = "\n".join(f"- [{mmss(t['start'])}] {t['title']}: " + " ".join(p.replace("**", "") for p in t["points"])
                     for t in topics)[:MAX_TRANSCRIPT_CHARS]
    return (
        "Below are a doctor's topic notes covering a whole video.\n"
        "Write (1) a 2-3 sentence overview in a doctor's voice and (2) 3-5 key clinical takeaways. "
        "Wrap the 1-3 most important keywords of each takeaway in **double asterisks**.\n"
        'Reply with JSON ONLY: {"overview": "...", "takeaways": ["...", "..."]}\n'
        "Rules: " + lang_rule + " Never use double quotes inside text (use single quotes) and never put line "
        "breaks inside a string.\n\nTOPIC NOTES:\n" + body)


def _close_truncated(s: str) -> str:
    """Close any open string/brackets of a JSON text that was cut off."""
    stack, in_str, esc = [], False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    if in_str:
        s += '"'
    return re.sub(r",\s*$", "", s) + "".join(reversed(stack))


def _lenient(raw: str) -> dict:
    """Last resort: pull the fields out with regexes (survives stray quotes inside text)."""
    topics = []
    for m in re.finditer(r'"start_seconds"\s*:\s*"?([\d.]+)"?\s*,\s*"title"\s*:\s*"(.*?)"\s*,\s*"points"\s*:\s*\[(.*?)\]\s*\}', raw, re.S):
        pts = [p.strip().strip('"').strip() for p in re.split(r'"\s*,\s*"', m.group(3).strip()) if p.strip().strip('"').strip()]
        topics.append({"start_seconds": m.group(1), "title": m.group(2).strip(), "points": pts})
    if not topics:
        raise ValueError("The AI did not return valid JSON.")
    ov = re.search(r'"overview"\s*:\s*"(.*?)"\s*,\s*"topics"', raw, re.S)
    tail = raw[raw.rfind('"takeaways"'):] if '"takeaways"' in raw else ""
    tk = re.search(r'\[(.*?)\]', tail, re.S)
    take = [t.strip().strip('"').strip() for t in re.split(r'"\s*,\s*"', tk.group(1).strip())] if tk else []
    return {"overview": ov.group(1).strip() if ov else "", "topics": topics, "takeaways": [t for t in take if t]}


def extract_json(raw: str) -> dict:
    """Parse JSON even if it is fenced, has text around it, was cut off, or has stray quotes. Raises ValueError."""
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    i = raw.find("{")
    if i < 0:
        raise ValueError("The AI did not return valid JSON.")
    raw = raw[i:]
    cands = [raw, raw[:raw.rfind("}") + 1]]
    for c in cands:
        try:
            return json.loads(c, strict=False)
        except ValueError:
            pass
    cut = raw                                   # cut-off answer: drop the unfinished tail topic by topic
    for _ in range(12):
        try:
            return json.loads(_close_truncated(cut), strict=False)
        except ValueError:
            k = cut.rfind("},")
            if k < 0:
                break
            cut = cut[:k + 1]
    return _lenient(raw)


def normalize(d: dict, duration: float) -> dict:
    topics = []
    for x in d.get("topics", []):
        if not isinstance(x, dict):
            continue
        try:
            start = max(0.0, float(x.get("start_seconds", 0)))
        except (TypeError, ValueError):
            start = 0.0
        pts = [str(p).strip() for p in x.get("points", []) if str(p).strip()]
        if pts:
            topics.append({"start": start, "title": str(x.get("title", "")).strip() or "Topic", "points": pts})
    topics.sort(key=lambda t: t["start"])
    for i, t in enumerate(topics):
        t["end"] = topics[i + 1]["start"] if i + 1 < len(topics) else duration
    return {"overview": str(d.get("overview", "")).strip(), "topics": topics,
            "takeaways": [str(t).strip() for t in d.get("takeaways", []) if str(t).strip()]}


def http_json(url, payload, headers, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


GATEWAY_OK = {}     # remembers which gateway model worked, so parallel calls don't all retry the bad ones


def llm_gateway(prompt: str, max_tokens: int = 8000) -> str:
    order = ([GATEWAY_OK["m"]] if "m" in GATEWAY_OK else []) + [m for m in AAI_LLM_MODELS if m != GATEWAY_OK.get("m")]
    last = ""
    for model in order:
        for attempt in range(4):                       # a busy gateway (429/5xx) is retried with growing pauses
            r = requests.post(AAI_LLM_GATEWAY_URL, headers={"authorization": ASSEMBLYAI_API_KEY}, timeout=300, json={
                "model": model, "max_tokens": max_tokens, "temperature": 0.2,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                try:
                    wait = float(r.headers.get("Retry-After", 0))
                except ValueError:
                    wait = 0
                time.sleep(min(30, wait or 5 * (attempt + 1)))
                continue
            break
        if r.status_code == 200:
            GATEWAY_OK["m"] = model
            return r.json()["choices"][0]["message"]["content"]
        last = f"{model}: HTTP {r.status_code} {r.text[:150]}"
        if r.status_code in (401, 403):
            break
    raise RuntimeError(f"LLM Gateway failed ({last})")


def llm_deepseek(prompt: str, max_tokens: int = 8000) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("NO_KEY")
    try:
        data = http_json(DEEPSEEK_URL + "/chat/completions", {
            "model": DEEPSEEK_MODEL, "temperature": 0.2, "max_tokens": min(max_tokens, 8000),
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]},
            {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"})
    except urllib.error.HTTPError as e:
        raise RuntimeError({401: "Invalid DeepSeek API key.", 402: "DeepSeek balance is insufficient.",
                            429: "DeepSeek rate limit reached."}.get(e.code, f"DeepSeek error {e.code}"))
    return data["choices"][0]["message"]["content"]


def llm_ollama(prompt: str, max_tokens: int = 3000) -> str:
    try:
        data = http_json(OLLAMA_URL + "/api/chat", {
            "model": OLLAMA_MODEL, "stream": False, "format": "json",
            "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": max_tokens},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt[:MAX_LOCAL_CHARS]}]},
            {}, timeout=900)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama model not found: run 'ollama pull {OLLAMA_MODEL}'" if e.code == 404 else f"Ollama error {e.code}")
    except urllib.error.URLError:
        raise RuntimeError("Ollama is not running")
    return data["message"]["content"]


STOPWORDS = set("""a about after again all also am an and any are as at be because been before being both but by can could did do
does for from had has have he her here him his how i if in into is it its just like me more most my no not now of on or our out
over so some such than that the their them then there these they this those to too up very was we were what when where which
while who why will with would you your yeah okay really going get got know think say said one thing things want need make""".split())


def extractive_summary(sents, duration) -> dict:
    """Offline fallback: split the video into time parts and pick the most important sentences of each."""
    toks = [[w for w in re.findall(r"[a-zA-Z']+", s.lower()) if w not in STOPWORDS and len(w) > 2] for _, s in sents]
    freq = Counter(w for t in toks for w in t)
    score = [sum(freq[w] for w in t) / (max(len(t), 1) ** 0.6) for t in toks]
    n = len(sents)
    groups = min(6, max(1, n // 6))
    size = -(-n // groups)
    topics = []
    for g in range(groups):
        idx = list(range(g * size, min(n, (g + 1) * size)))
        if idx:
            best = sorted(sorted(idx, key=lambda i: -score[i])[:3])
            topics.append({"start": sents[idx[0]][0], "title": f"Part {g + 1}", "points": [sents[i][1] for i in best]})
    for i, t in enumerate(topics):
        t["end"] = topics[i + 1]["start"] if i + 1 < len(topics) else duration
    top = sorted(sorted(range(n), key=lambda i: -score[i])[:2])
    return {"overview": " ".join(sents[i][1] for i in top), "topics": topics, "takeaways": []}


def format_summary(s: dict) -> str:
    strip = lambda t: t.replace("**", "")
    out = []
    if s.get("overview"):
        out += ["DOCTOR'S OVERVIEW", strip(s["overview"]), ""]
    for t in s["topics"]:
        out.append(f"[{mmss(t['start'])} - {mmss(t['end'])}] {t['title'].upper()}")
        out += [f"  - {strip(p)}" for p in t["points"]]
        out.append("")
    if s.get("takeaways"):
        out.append("KEY TAKEAWAYS")
        out += [f"  {i}. {strip(t)}" for i, t in enumerate(s["takeaways"], 1)]
    return "\n".join(out).strip()


# ---------- 4. background jobs ----------
def split_chunks(sents, max_chars, max_seconds=900):
    """Cut the transcript into parts of ~15 minutes / max_chars so they can be summarized in parallel."""
    chunks, cur, size = [], [], 0
    for t, s in sents:
        if cur and (size + len(s) > max_chars or t - cur[0][0] > max_seconds):
            chunks.append(cur)
            cur, size = [], 0
        cur.append((t, s))
        size += len(s) + 10
    if cur:
        chunks.append(cur)
    return chunks


NO_RETRY = ("HTTP 401", "HTTP 403", "Invalid DeepSeek", "insufficient", "not running", "not found")


def run_json(fn, prompt, max_tokens, tries=3):
    """Call one AI engine and parse its JSON. Busy server / broken JSON -> wait a little and ask again.
    Wrong key / engine not running -> give up immediately."""
    err = None
    for attempt in range(tries):
        try:
            return extract_json(fn(prompt, max_tokens))
        except Exception as e:
            err = e
            if any(k in str(e) for k in NO_RETRY):
                break
            if attempt < tries - 1:
                time.sleep(3 * (attempt + 1))
    raise err


def part_topics(fn, part, lang):
    r = run_json(fn, build_chunk_prompt(part, lang), 4000)
    topics = normalize(r, part[-1][0] + 5)["topics"]
    if not topics:
        raise ValueError("the AI returned no topics")
    return topics


def basic_topics(part):
    out = extractive_summary(part, part[-1][0] + 5)["topics"]
    for t in out:
        t["title"] = f"Section at {mmss(t['start'])}"
    return out


def summarize_with(jid, name, fn, sents, duration, lang, chunk_chars, workers):
    """Short video -> one request. Long video -> parts summarized in parallel; any part that fails is
    retried one by one (split in half) before the basic summarizer is used. Returns (summary, notes)."""
    chunks = split_chunks(sents, chunk_chars)
    if len(chunks) == 1:
        update(jid, stage=f"Writing the summary with {name}...")
        return normalize(run_json(fn, build_prompt(sents, lang), 8000), duration), []

    total, done = len(chunks), 0
    results, errs = [None] * total, {}
    update(jid, stage=f"Summarizing part 0/{total} with {name}...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(part_topics, fn, c, lang): i for i, c in enumerate(chunks)}
        for f in as_completed(futs):
            i = futs[f]
            try:
                results[i] = f.result()
            except Exception as e:
                errs[i] = clean_error(e)
            done += 1
            update(jid, stage=f"Summarizing part {done}/{total} with {name}...")

    # second pass: failed parts, one at a time (no rate-limit pressure), each cut in half (shorter prompt)
    failed = [i for i in range(total) if results[i] is None]
    if failed and len(failed) == total and any(k in errs[failed[0]] for k in NO_RETRY):
        raise RuntimeError(errs[failed[0]])              # e.g. wrong key: let the next engine take over
    notes = []
    for n, i in enumerate(failed, 1):
        update(jid, stage=f"Retrying part {i + 1} ({n}/{len(failed)}) with {name}...")
        c = chunks[i]
        halves = [h for h in (c[:len(c) // 2], c[len(c) // 2:]) if h]
        got, used_basic = [], False
        for h in halves:
            try:
                got += part_topics(fn, h, lang)
            except Exception as e:
                errs[i] = clean_error(e)
                got += basic_topics(h)
                used_basic = True
        results[i] = got
        if used_basic:
            notes.append(f"Part {i + 1} used the basic summarizer ({errs[i][:140]})")

    if all(r and r[0]["title"].startswith("Section at") for r in results):
        raise RuntimeError(next(iter(errs.values()), "no topics returned"))

    topics = [t for r in results for t in r]
    topics.sort(key=lambda t: t["start"])
    for k, t in enumerate(topics):
        t["end"] = topics[k + 1]["start"] if k + 1 < len(topics) else duration

    update(jid, stage=f"Writing overview and takeaways with {name}...")
    overview, take = "", []
    try:
        fin = run_json(fn, build_final_prompt(topics, lang), 2000)
        overview = str(fin.get("overview", "")).strip()
        take = [str(t).strip() for t in fin.get("takeaways", []) if str(t).strip()]
    except Exception:
        pass
    if not overview:
        overview = "Topics covered: " + "; ".join(t["title"] for t in topics[:8]) + "."
    return {"overview": overview, "topics": topics, "takeaways": take}, notes


def summarize_job(jid, text, sents, lang):
    """Writes the topic summary (in `lang`, or the video's language if empty) from an existing transcript."""
    duration = sents[-1][0] + 5
    # (name, function, characters per part, parallel requests)
    engines = [("AssemblyAI LLM Gateway", llm_gateway, 14000, PARALLEL_REQUESTS)]
    if DEEPSEEK_API_KEY:
        engines.append((f"DeepSeek ({DEEPSEEK_MODEL})", llm_deepseek, 14000, PARALLEL_REQUESTS))
    engines.append((f"Local AI ({OLLAMA_MODEL})", llm_ollama, 5000, 1))

    summary, method, notes, extra = None, "", [], []
    for name, fn, chunk_chars, workers in engines:
        try:
            cand, extra = summarize_with(jid, name, fn, sents, duration, lang, chunk_chars, workers)
            if cand["topics"]:
                summary, method = cand, name
                break
            notes.append(f"{name} returned no topics.")
        except Exception as e:
            notes.append(f"{name}: {clean_error(e)}")

    warning = None
    if summary is None:
        update(jid, stage="Writing basic summary...")
        summary, method = extractive_summary(sents, duration), "Built-in basic summarizer"
        warning = "Basic offline summary (no AI engine worked). " + " | ".join(notes)
        if lang:
            warning += " The basic summary cannot translate, so it stays in the video's language."
    elif extra:
        warning = " ".join(extra)

    update(jid, status="done", stage="Done", transcript=text, summary=summary, summary_text=format_summary(summary),
           method=method, warning=warning, lang=lang, transcript_timed=[{"t": t, "s": s} for t, s in sents])


def run_job(jid, url, file_path, lang=""):
    path, keep, audio_path = file_path, False, None
    try:
        if url:
            update(jid, stage="Downloading video...")
            path, title = download_media(url, lambda s: update(jid, stage=s))
            keep = True
            update(jid, media_path=str(path), media_url=f"/api/media/{jid}", title=title)
        # Upload just the audio track - much smaller & faster than sending the whole video,
        # the full video file itself is kept untouched so the player above still works.
        audio_path = extract_audio(Path(path), lambda s: update(jid, stage=s))
        update(jid, stage="Uploading to AssemblyAI...")
        text, sents = transcribe(audio_path, lambda s: update(jid, stage=s))
        if not text:
            raise RuntimeError("No speech was found in this media.")
        update(jid, sents=sents)
        summarize_job(jid, text, sents, lang)
    except Exception as e:
        update(jid, status="error", error=f"Could not process this media: {clean_error(e)}")
    finally:
        if audio_path and audio_path != path:
            Path(audio_path).unlink(missing_ok=True)
        if path and not keep:
            Path(path).unlink(missing_ok=True)


def resummarize_job(jid, lang):
    """Same transcript, new summary language (no re-upload needed)."""
    try:
        job = JOBS[jid]
        summarize_job(jid, job["transcript"], job["sents"], lang)
    except Exception as e:
        update(jid, status="error", error=f"Could not re-write the summary: {clean_error(e)}")


# ---------- 5. web page ----------
INDEX_HTML = r'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Innoclique - Media Summarizer</title>
<link rel="icon" href="/logo.webp">
<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#f4f7fb;--card:#fff;--line:#e2e8f0;--text:#1f2a37;--muted:#64748b;--accent:#1470f0;--sky:#0ea5e9;--orange:#f5a000;--warn:#b45309;--soft:#eff6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"Segoe UI",system-ui,-apple-system,sans-serif;-webkit-font-smoothing:antialiased}
.top{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.95);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);box-shadow:0 1px 4px rgba(15,23,42,.05)}
.top::after{content:"";display:block;height:3px;background:linear-gradient(90deg,var(--sky),var(--accent) 60%,var(--orange))}
.top-in{max-width:1300px;margin:0 auto;height:60px;padding:0 16px;display:flex;align-items:center;gap:12px}
.logo{height:34px;width:auto;display:block}
.sep{width:1px;height:22px;background:var(--line)}
.ttl{font-weight:600;color:#334155;font-size:15px}
.pill{margin-left:auto;font-size:12px;font-weight:600;color:var(--accent);background:var(--soft);border:1px solid #bfdbfe;border-radius:999px;padding:4px 12px}
.wrap{max-width:1300px;margin:0 auto;padding:20px 16px 32px;display:flex;flex-direction:column;gap:16px}
.card{background:var(--card);border:1px solid rgba(226,232,240,.9);border-radius:14px;padding:16px;box-shadow:0 4px 24px rgba(15,23,42,.06)}
h2{margin:0 0 12px;font-size:16px;font-weight:700;color:#0f172a;display:flex;align-items:center;gap:8px}
h2::before{content:"";width:4px;height:16px;border-radius:2px;background:var(--orange)}
input[type=text]{flex:1;min-width:220px;padding:11px 12px;border-radius:10px;border:1px solid #cbd5e1;background:#fff;color:var(--text);font:inherit;font-size:14px}
input[type=text]:focus,select:focus{outline:none;border-color:var(--sky);box-shadow:0 0 0 3px rgba(14,165,233,.2)}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
button,select{font:inherit;font-size:14px;font-weight:500;padding:9px 14px;border-radius:10px;border:1px solid var(--line);background:#fff;color:#334155;cursor:pointer;transition:.16s}
button:hover:not(:disabled){background:#f8fafc;border-color:var(--accent);color:var(--accent)}
button.primary{background:linear-gradient(135deg,var(--sky),var(--accent));border-color:transparent;color:#fff;font-weight:600;box-shadow:0 4px 12px rgba(20,112,240,.3)}
button.primary:hover:not(:disabled){background:linear-gradient(135deg,#0284c7,#0f5fd0);color:#fff}
button:disabled{opacity:.55;cursor:not-allowed}
.main{display:grid;grid-template-columns:1.15fr 1fr;gap:16px;align-items:start}
@media(max-width:900px){.main{grid-template-columns:1fr}}
.player{background:#0b1220;border-radius:12px;aspect-ratio:16/9;position:relative;overflow:hidden}
video{width:100%;height:100%;display:block;background:#0b1220}
#ph{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#94a3b8;text-align:center;padding:20px;font-size:14px;line-height:1.6}
.ctrl{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:12px}
.ctrl button{padding:8px 12px} #seek{flex:1 1 100%;order:-1;accent-color:var(--accent)}
#play{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600;min-width:92px}
#play:hover{background:#0f5fd0;color:#fff}
#time{color:var(--muted);font-size:13px;font-family:Consolas,monospace}
.sumbox{height:calc(100vh - 250px);min-height:420px;max-height:720px;overflow-y:auto;position:relative;padding-right:6px}
.sumbox::-webkit-scrollbar{width:8px}.sumbox::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:8px}
.ph2{color:var(--muted);padding:40px 8px;text-align:center;font-size:14px}
.loader{display:none;flex-direction:column;align-items:center;justify-content:center;gap:16px;height:100%;text-align:center;color:var(--muted);font-size:14px}
.spin{width:56px;height:56px;border:5px solid #e2e8f0;border-top-color:var(--accent);border-right-color:var(--orange);border-radius:50%;animation:sp .9s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.bar{width:70%;height:6px;background:#e2e8f0;border-radius:6px;overflow:hidden}
.bar div{height:100%;width:40%;background:linear-gradient(90deg,var(--sky),var(--accent));border-radius:6px;animation:sl 1.4s ease-in-out infinite}
@keyframes sl{0%{margin-left:-40%}100%{margin-left:100%}}
.overview{background:var(--soft);border-left:4px solid var(--accent);padding:12px 14px;border-radius:10px;margin-bottom:12px;line-height:1.6;font-size:14px;color:#1e293b}
.topic{border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:10px;background:#fff;transition:.2s}
.topic:hover{box-shadow:0 4px 16px rgba(15,23,42,.08)}
.topic.active{border-color:var(--accent);background:var(--soft);box-shadow:0 0 0 3px rgba(20,112,240,.12)}
.topic h3{margin:8px 0 6px;font-size:15px;font-weight:700;color:#0f172a}
.ts{font-family:Consolas,monospace;font-size:12px;padding:3px 10px;border-radius:999px;border:1px solid var(--accent);color:var(--accent);background:#fff}
.ts:hover:not(:disabled){background:var(--accent)!important;color:#fff!important}
.topic ul{margin:0;padding-left:20px;line-height:1.6;font-size:14px} .topic li{margin-bottom:4px}
b{color:#0f172a;background:#fff3cf;padding:0 3px;border-radius:3px;font-weight:600}
.take{margin-top:14px} .take ol{padding-left:22px;line-height:1.6;font-size:14px}
#warn{color:var(--warn);background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:8px 12px;font-size:13px;display:none;margin-bottom:10px}
.note{color:var(--muted);font-size:12px;margin-top:10px}
details{color:var(--muted);font-size:14px} summary{cursor:pointer;font-weight:600;color:#334155}
#transcript{white-space:pre-wrap;color:#334155;font-size:13px;line-height:1.6;margin-top:10px}
.foot{text-align:center;color:var(--muted);font-size:12px}.foot b{background:none;color:var(--orange)}
#lang{max-width:230px}
/* ---- single combined box: paste a link OR choose a video ---- */
.src{flex:1 1 420px;display:flex;align-items:center;gap:6px;padding:4px;border:1px solid #cbd5e1;border-radius:10px;background:#fff;transition:.16s}
.src:focus-within{border-color:var(--sky);box-shadow:0 0 0 3px rgba(14,165,233,.2)}
.src input[type=text]{flex:1;min-width:0;border:0;box-shadow:none;background:transparent;padding:8px 10px}
.src input[type=text]:focus{box-shadow:none}
.src button{white-space:nowrap;background:var(--soft);border-color:#bfdbfe;color:var(--accent);font-weight:600}
</style></head>
<body>
<header class="top"><div class="top-in">
  <img src="/logo.webp" alt="Innoclique" class="logo" onerror="this.style.display='none'"><span class="sep"></span>
  <span class="ttl">Media Summarizer</span>
</div></header>
<div class="wrap">

<div class="card">
  <div class="row">
    <div class="src">
      <input type="text" id="url" placeholder="Paste a link (YouTube, Drive, Dropbox, OneDrive, .mp4) or choose a video &rarr;">
      <input type="file" id="file" accept="video/*,audio/*" hidden>
      <button id="pick">📁 Choose video</button>
    </div>
    <select id="lang" title="Language of the summary"><option value="">🌐 Same language as video</option><option value="English">English</option><option value="Hindi">Hindi</option><option value="Telugu">Telugu</option><option value="Tamil">Tamil</option><option value="Kannada">Kannada</option><option value="Malayalam">Malayalam</option><option value="Marathi">Marathi</option><option value="Bengali">Bengali</option><option value="Gujarati">Gujarati</option><option value="Urdu">Urdu</option><option value="Spanish">Spanish</option><option value="French">French</option><option value="German">German</option><option value="Arabic">Arabic</option><option value="Chinese">Chinese</option><option value="Japanese">Japanese</option><option value="Russian">Russian</option><option value="Portuguese">Portuguese</option></select>
    <button class="primary" id="go">Summarize</button>
    <button id="clear">Clear</button>
  </div>
</div>

<div class="main">
  <div class="card">
    <div class="player"><video id="v" playsinline></video><div id="ph">Your video will play here.<br>Choose a file or paste a link.</div></div>
    <div class="ctrl">
      <input type="range" id="seek" min="0" max="1000" value="0">
      <button id="back" title="Back 10s">⏪ 10s</button>
      <button id="play" title="Play / Pause">▶ Play</button>
      <button id="fwd" title="Forward 10s">10s ⏩</button>
      <span id="time">00:00 / 00:00</span>
      <select id="speed"><option value="0.75">0.75x</option><option value="1" selected>1x</option><option value="1.25">1.25x</option><option value="1.5">1.5x</option><option value="2">2x</option></select>
      <button id="mute" title="Mute">🔊</button>
      <button id="full" title="Fullscreen">⛶</button>
    </div>
  </div>

  <div class="card">
    <h2>Doctor's Summary by Topic</h2>
    <div class="sumbox">
      <div class="loader" id="loader"><div class="spin"></div><div id="stage">Starting...</div><div class="bar"><div></div></div></div>
      <div id="out"><div class="ph2">Your topic-by-topic summary will appear here.</div></div>
    </div>
    <div class="row" style="margin-top:10px">
      <button id="read">Read aloud</button><button id="stop">Stop</button>
      <button id="copy">Copy</button><button id="download">Download PDF</button>
    </div>
    <div class="note">Educational summary of the video's content - not a personal medical diagnosis or advice.</div>
  </div>
</div>

<div class="card"><details><summary>Show full transcript</summary><div class="row" style="margin-top:10px"><button id="dlt">Download PDF</button></div><div id="transcript"></div></details></div>
<div class="foot"><b>Innoclique</b> · Exploring Possibilities..</div>
</div>

<script>
const $=id=>document.getElementById(id), v=$("v");
let chosen=null, timer=null, topics=[], plain="", objUrl=null, jobId=null, summ=null, timed=[], baseName="video", videoFromLink=false;
const fmt=s=>{s=Math.floor(s||0);const h=Math.floor(s/3600),m=Math.floor(s%3600/60),c=s%60,p=n=>String(n).padStart(2,"0");return (h?h+":":"")+p(m)+":"+p(c)};
const esc=t=>String(t).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const hl=t=>esc(t).replace(/\*\*(.+?)\*\*/g,"<b>$1</b>");

// ---- file / link (one combined box) ----
function setSrc(src){v.src=src;$("ph").style.display="none";}
function resetSource(){            // forget the chosen file / old video
  chosen=null;$("file").value="";jobId=null;summ=null;baseName="video";videoFromLink=false;
  if(objUrl){URL.revokeObjectURL(objUrl);objUrl=null;}
  v.pause();v.removeAttribute("src");v.load();$("ph").style.display="flex";
}
$("pick").onclick=()=>$("file").click();
$("file").onchange=e=>{const f=e.target.files[0];if(!f)return;
  resetSource();
  chosen=f;$("url").value=f.name;baseName=f.name.replace(/\.[^.]+$/,"");
  objUrl=URL.createObjectURL(f);setSrc(objUrl);};

// typing or pasting a link into the same box replaces a chosen file
$("url").onpaste=()=>{if(chosen){resetSource();$("url").value="";}};
$("url").onkeydown=e=>{
  if(chosen&&!e.ctrlKey&&!e.metaKey&&(e.key.length===1||e.key==="Backspace"||e.key==="Delete")){resetSource();$("url").value="";}
};
$("url").oninput=()=>{if(!chosen&&videoFromLink){resetSource();}};

// ---- player controls ----
const seekTo=s=>{v.currentTime=Math.max(0,Math.min(s,v.duration||s));};
$("play").onclick=()=>{if(!v.src)return;v.paused?v.play():v.pause();};
v.onclick=()=>$("play").click();
v.onplay=()=>$("play").textContent="⏸ Pause"; v.onpause=()=>$("play").textContent="▶ Play";
$("back").onclick=()=>seekTo(v.currentTime-10); $("fwd").onclick=()=>seekTo(v.currentTime+10);
$("speed").onchange=e=>v.playbackRate=parseFloat(e.target.value);
$("mute").onclick=()=>{v.muted=!v.muted;$("mute").textContent=v.muted?"🔇":"🔊";};
$("full").onclick=()=>{const p=document.querySelector(".player");document.fullscreenElement?document.exitFullscreen():p.requestFullscreen();};
$("seek").oninput=e=>{if(v.duration)v.currentTime=v.duration*e.target.value/1000;};
v.ontimeupdate=()=>{
  $("time").textContent=fmt(v.currentTime)+" / "+fmt(v.duration);
  if(v.duration)$("seek").value=v.currentTime/v.duration*1000;
  document.querySelectorAll(".topic").forEach((el,i)=>{const t=topics[i];el.classList.toggle("active",v.currentTime>=t.start&&v.currentTime<t.end);});
};
v.onloadedmetadata=()=>{$("time").textContent="00:00 / "+fmt(v.duration);};
document.addEventListener("keydown",e=>{if(/INPUT|SELECT/.test(document.activeElement.tagName))return;
  if(e.code==="Space"){e.preventDefault();$("play").click();}
  if(e.code==="ArrowRight")seekTo(v.currentTime+10); if(e.code==="ArrowLeft")seekTo(v.currentTime-10);});

// upload with a progress percentage (a 1-hour video file can be large)
function sendForm(url,fd,onProg){
  return new Promise((resolve,reject)=>{
    const x=new XMLHttpRequest(); x.open("POST",url);
    if(onProg)x.upload.onprogress=e=>{if(e.lengthComputable)onProg(Math.round(e.loaded/e.total*100));};
    x.onload=()=>{let j={};try{j=JSON.parse(x.responseText);}catch(_){}resolve({ok:x.status>=200&&x.status<300,json:j});};
    x.onerror=()=>reject(new Error("Network error while sending the file"));
    x.send(fd);
  });
}

// ---- summarize ----
$("go").onclick=async()=>{
  const url=$("url").value.trim();
  if(!chosen&&!url){alert("Please choose a file or paste a video link first.");return;}
  const fd=new FormData(); if(chosen)fd.append("file",chosen); else fd.append("url",url);
  fd.append("lang",$("lang").value);
  $("go").disabled=true; summ=null; showLoader("Starting...");
  try{
    const up=chosen?p=>{$("stage").textContent="Sending the video to the app... "+p+"%";}:null;
    const {ok,json:j}=await sendForm("/api/summarize",fd,up);
    if(!ok)throw new Error(j.error||"Request failed"); jobId=j.job_id; poll(jobId);
  }catch(err){fail(err.message);}
};
// changing the language after a summary exists re-writes it from the same transcript (no re-upload)
$("lang").onchange=async()=>{
  if(!jobId||!summ)return;
  const fd=new FormData(); fd.append("lang",$("lang").value);
  $("go").disabled=true; showLoader("Re-writing the summary...");
  try{
    const r=await fetch("/api/resummarize/"+jobId,{method:"POST",body:fd}); const j=await r.json();
    if(!r.ok)throw new Error(j.error||"Request failed"); poll(jobId);
  }catch(err){fail(err.message);}
};
function showLoader(msg){$("loader").style.display="flex";$("out").style.display="none";$("stage").textContent=msg;}
function poll(id){
  clearInterval(timer);
  timer=setInterval(async()=>{
    try{
      const job=await (await fetch("/api/status/"+id)).json();
      if(job.status==="error"){clearInterval(timer);return fail(job.error);}
      $("stage").textContent=job.stage||"Working...";
      if(job.media_url&&!chosen&&!v.getAttribute("src")){setSrc(job.media_url);videoFromLink=true;}
      if(job.title&&!chosen)baseName=job.title.replace(/[\\/:*?"<>|]+/g,"_").slice(0,80);
      if(job.status==="done"){clearInterval(timer);render(job);}
    }catch(err){clearInterval(timer);fail(err.message);}
  },1000);
}
function fail(msg){
  $("go").disabled=false;$("loader").style.display="none";$("out").style.display="block";
  $("out").innerHTML='<div class="ph2" style="color:#dc2626">'+esc(msg||"Something went wrong")+'</div>';
}
function render(job){
  $("go").disabled=false;$("loader").style.display="none";$("out").style.display="block";
  const s=job.summary; summ=s; topics=s.topics; plain=job.summary_text; timed=job.transcript_timed||[];
  let h="";
  if(job.warning)h+='<div id="warn" style="display:block">'+esc(job.warning)+'</div>';
  if(s.overview)h+='<div class="overview"><b>Overview:</b> '+hl(s.overview)+'</div>';
  topics.forEach(t=>{
    h+='<div class="topic"><button class="ts" data-s="'+t.start+'">▶ '+fmt(t.start)+' – '+fmt(t.end)+'</button><h3>'+esc(t.title)+'</h3><ul>'+
       t.points.map(p=>"<li>"+hl(p)+"</li>").join("")+"</ul></div>";
  });
  if(s.takeaways.length)h+='<div class="take"><h2>Key takeaways</h2><ol>'+s.takeaways.map(t=>"<li>"+hl(t)+"</li>").join("")+"</ol></div>";
  h+='<div class="note">Written by: '+esc(job.method)+'</div>';
  $("out").innerHTML=h;
  $("out").querySelectorAll(".ts").forEach(b=>b.onclick=()=>{seekTo(parseFloat(b.dataset.s));v.play();});
  $("transcript").textContent=timed.map(x=>"["+fmt(x.t)+"] "+x.s).join("\n")||job.transcript;
}

// ---- clear / read ----
$("clear").onclick=()=>{
  clearInterval(timer);speechSynthesis.cancel();
  resetSource();
  topics=[];plain="";timed=[];
  $("url").value="";
  $("loader").style.display="none";$("out").style.display="block";
  $("out").innerHTML='<div class="ph2">Your topic-by-topic summary will appear here.</div>';
  $("transcript").textContent="";$("go").disabled=false;$("time").textContent="00:00 / 00:00";
};
$("read").onclick=()=>{if(!plain)return;speechSynthesis.cancel();speechSynthesis.speak(new SpeechSynthesisUtterance(plain));};
$("stop").onclick=()=>speechSynthesis.cancel();
$("copy").onclick=()=>{if(plain)navigator.clipboard.writeText(plain);};

// ---- PDF export (drawn from HTML so Hindi, Telugu, Arabic etc. print correctly) ----
const HEAD=(t,sub)=>'<div style="display:flex;align-items:center;justify-content:space-between;border-bottom:3px solid #1470f0;padding-bottom:10px;margin-bottom:6px"><img src="/logo.webp" style="height:34px"><div style="text-align:right"><div style="font-weight:700;font-size:16px;color:#0f172a">'+esc(t)+'</div><div style="font-size:11px;color:#64748b">'+esc(sub)+'</div></div></div>';
async function makePdf(blocks,filename,btn){
  const old=btn.textContent; btn.disabled=true; btn.textContent="Preparing PDF...";
  const box=document.createElement("div"); box.id="pdfbox";
  box.style.cssText="position:fixed;left:-10000px;top:0;width:720px;background:#fff;color:#1f2a37;font:14px/1.6 Inter,'Segoe UI','Nirmala UI',system-ui,sans-serif";
  document.body.appendChild(box);
  try{
    const {jsPDF}=window.jspdf, doc=new jsPDF({unit:"pt",format:"a4"});
    const W=doc.internal.pageSize.getWidth(), H=doc.internal.pageSize.getHeight(), M=36;
    let y=M;
    for(const html of blocks){
      box.innerHTML=html;
      const c=await html2canvas(box,{scale:2,backgroundColor:"#ffffff",useCORS:true,
        onclone:d=>{const e=d.getElementById("pdfbox");e.style.left="0";}});
      let w=W-2*M, h=c.height*w/c.width;
      if(h>H-2*M){w=w*(H-2*M)/h;h=H-2*M;}
      if(y+h>H-M&&y>M){doc.addPage();y=M;}
      doc.addImage(c.toDataURL("image/jpeg",0.92),"JPEG",M,y,w,h); y+=h+4;
    }
    doc.save(filename);
  }catch(e){alert("Could not create the PDF: "+e.message+"\n(The PDF libraries load from the internet - check your connection.)");}
  box.remove(); btn.disabled=false; btn.textContent=old;
}
const today=()=>new Date().toLocaleDateString();
$("download").onclick=()=>{
  if(!summ)return; const s=summ, b=[HEAD("Video Summary",baseName+" · "+today())];
  if(s.overview)b.push('<div style="background:#eff6ff;border-left:4px solid #1470f0;padding:10px 14px;border-radius:8px;margin:8px 0"><b>Overview:</b> '+hl(s.overview)+'</div>');
  s.topics.forEach(t=>b.push('<div style="border:1px solid #e2e8f0;border-radius:10px;padding:10px 14px;margin:6px 0"><span style="font-family:Consolas,monospace;font-size:12px;color:#1470f0;border:1px solid #1470f0;border-radius:999px;padding:2px 9px">'+fmt(t.start)+' – '+fmt(t.end)+'</span><div style="font-weight:700;font-size:15px;margin:8px 0 4px">'+esc(t.title)+'</div><ul style="margin:0;padding-left:20px">'+t.points.map(p=>"<li>"+hl(p)+"</li>").join("")+"</ul></div>"));
  if(s.takeaways.length)b.push('<div style="margin:10px 0"><div style="font-weight:700;font-size:15px;margin-bottom:4px">Key takeaways</div><ol style="margin:0;padding-left:22px">'+s.takeaways.map(t=>"<li>"+hl(t)+"</li>").join("")+"</ol></div>");
  makePdf(b,baseName+"-summary.pdf",$("download"));
};
$("dlt").onclick=()=>{
  if(!timed.length){alert("No transcript yet - summarize a file first.");return;}
  const b=[HEAD("Full Transcript",baseName+" · "+today())]; let cur="",n=0;
  const flush=()=>{if(cur)b.push('<div style="font-size:13px;line-height:1.7">'+cur+'</div>');cur="";n=0;};
  timed.forEach(x=>{cur+='<div><span style="color:#1470f0;font-family:Consolas,monospace">['+fmt(x.t)+']</span> '+esc(x.s)+'</div>';n+=x.s.length;if(n>1500)flush();});
  flush(); makePdf(b,baseName+"-transcript.pdf",$("dlt"));
};
</script></body></html>
'''


@app.route("/")
def index():
    resp = make_response(INDEX_HTML)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    prune_old_jobs()
    url, file_path = request.form.get("url", "").strip(), None
    lang = re.sub(r"[^\w ()-]", "", request.form.get("lang", ""))[:30].strip()
    f = request.files.get("file")
    if f and f.filename:
        file_path = WORK_DIR / f"{uuid.uuid4().hex}{Path(f.filename).suffix}"
        f.save(file_path)
        url = ""
    elif not url:
        return jsonify(error="Paste a link or choose a file."), 400
    jid = uuid.uuid4().hex
    JOBS[jid] = {"status": "running", "stage": "Starting...", "created": time.time()}
    threading.Thread(target=run_job, args=(jid, url, file_path, lang), daemon=True).start()
    return jsonify(job_id=jid)


@app.route("/api/status/<jid>")
def api_status(jid):
    job = JOBS.get(jid)
    if not job:
        return jsonify(status="error", error="Job not found (the app may have restarted)."), 404
    return jsonify({k: v for k, v in job.items() if k not in ("created", "media_path", "sents")})


@app.route("/api/resummarize/<jid>", methods=["POST"])
def api_resummarize(jid):
    job = JOBS.get(jid)
    if not job or not job.get("sents"):
        return jsonify(error="This session expired. Please press Summarize again."), 404
    lang = re.sub(r"[^\w ()-]", "", request.form.get("lang", ""))[:30].strip()
    job.update(status="running", stage="Re-writing the summary...", created=time.time())
    threading.Thread(target=resummarize_job, args=(jid, lang), daemon=True).start()
    return jsonify(job_id=jid)


@app.route("/api/media/<jid>")
def api_media(jid):
    p = (JOBS.get(jid) or {}).get("media_path")
    if not p or not Path(p).exists():
        return "Not found", 404
    return send_file(p, conditional=True)   # supports Range requests, so seeking works


# ---------- Innoclique logo ----------
# >>> PASTE YOUR ORIGINAL LOGO STRING BETWEEN THE QUOTES BELOW (the long "UklGRjpTAABXRUJQ..." value from your old file). <<<
# If you leave it empty the app still works, it just shows no logo.
LOGO_B64 = ""
try:
    LOGO_BYTES = base64.b64decode(LOGO_B64) if LOGO_B64 else b""
except Exception:
    LOGO_BYTES = b""


@app.route("/logo.webp")
def logo():
    if not LOGO_BYTES:
        return "", 404
    resp = make_response(LOGO_BYTES)
    resp.headers["Content-Type"] = "image/webp"
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


if __name__ == "__main__":
    print(f".env file : {ENV_PATH} -> {'FOUND' if ENV_PATH.exists() else 'NOT FOUND'}")
    print(f"  ASSEMBLYAI_API_KEY: {'loaded' if ASSEMBLYAI_API_KEY else 'not set'}")
    print(f"  DEEPSEEK_API_KEY  : {'loaded' if DEEPSEEK_API_KEY else 'not set'}")
    app.run(debug=False, threaded=True)