"""
Media Summarizer (Doctor's view) - video/link -> AssemblyAI transcript (with timestamps) -> LLM summary by topic.
Run:  python 10.py   then open http://127.0.0.1:5000
Install:  pip install flask yt-dlp requests python-dotenv

.env (same folder as this script):
    ASSEMBLYAI_API_KEY=your-assemblyai-key
    DEEPSEEK_API_KEY=sk-your-deepseek-key      (optional)

Summary engines, tried in order: AssemblyAI LLM Gateway -> DeepSeek -> Ollama (local) -> built-in basic summarizer.
"""
import glob, json, os, re, shutil, tempfile, threading, time, uuid
import urllib.error, urllib.request
from collections import Counter
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
def download_media(url: str) -> Path:
    cookie_file = Path(__file__).with_name("cookies.txt")
    attempts = [{"cookiefile": str(cookie_file)}] if cookie_file.exists() else []
    attempts += [{"cookiesfrombrowser": (b,)} for b in COOKIE_BROWSERS] + [{}]
    deno, last = find_deno(), ""
    for extra in attempts:
        opts = {"format": "best[height<=480][ext=mp4]/best[height<=480]/best", "quiet": True, "noplaylist": True,
                "outtmpl": str(WORK_DIR / f"{uuid.uuid4().hex}.%(ext)s"), **extra}
        if deno:
            opts["js_runtimes"] = {"deno": {"path": deno}}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                return Path(ydl.prepare_filename(ydl.extract_info(url, download=True)))
        except Exception as e:
            last = clean_error(e)
    if "not a bot" in last or "Sign in" in last:
        raise RuntimeError("YouTube blocked this download (bot check). Sign in to YouTube in Edge/Chrome, "
                           "add a cookies.txt file, or download the video and use the upload box.")
    raise RuntimeError(last or "Download failed.")


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


def llm_gateway(prompt: str) -> str:
    last = ""
    for model in AAI_LLM_MODELS:
        r = requests.post(AAI_LLM_GATEWAY_URL, headers={"authorization": ASSEMBLYAI_API_KEY}, timeout=300, json={
            "model": model, "max_tokens": 8000, "temperature": 0.2,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        last = f"{model}: HTTP {r.status_code} {r.text[:150]}"
        if r.status_code in (401, 403):
            break
    raise RuntimeError(f"LLM Gateway failed ({last})")


def llm_deepseek(prompt: str) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("NO_KEY")
    try:
        data = http_json(DEEPSEEK_URL + "/chat/completions", {
            "model": DEEPSEEK_MODEL, "temperature": 0.2, "max_tokens": 8000,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]},
            {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"})
    except urllib.error.HTTPError as e:
        raise RuntimeError({401: "Invalid DeepSeek API key.", 402: "DeepSeek balance is insufficient.",
                            429: "DeepSeek rate limit reached."}.get(e.code, f"DeepSeek error {e.code}"))
    return data["choices"][0]["message"]["content"]


def llm_ollama(prompt: str) -> str:
    try:
        data = http_json(OLLAMA_URL + "/api/chat", {
            "model": OLLAMA_MODEL, "stream": False, "format": "json", "options": {"temperature": 0.2, "num_ctx": 8192},
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
def summarize_job(jid, text, sents, lang):
    """Writes the topic summary (in `lang`, or the video's language if empty) from an existing transcript."""
    duration = sents[-1][0] + 5
    prompt = build_prompt(sents, lang)
    engines = [("AssemblyAI LLM Gateway", llm_gateway)]
    if DEEPSEEK_API_KEY:
        engines.append((f"DeepSeek ({DEEPSEEK_MODEL})", llm_deepseek))
    engines.append((f"Local AI ({OLLAMA_MODEL})", llm_ollama))

    summary, method, notes = None, "", []
    for name, fn in engines:
        try:
            update(jid, stage=f"Writing the summary with {name}...")
            for attempt in (1, 2):
                try:
                    cand = normalize(extract_json(fn(prompt)), duration)
                    break
                except ValueError:
                    if attempt == 2:
                        raise
                    update(jid, stage=f"Fixing the summary format ({name}, retrying)...")
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

    update(jid, status="done", stage="Done", transcript=text, summary=summary, summary_text=format_summary(summary),
           method=method, warning=warning, lang=lang, transcript_timed=[{"t": t, "s": s} for t, s in sents])


def run_job(jid, url, file_path, lang=""):
    path, keep = file_path, False
    try:
        if url:
            update(jid, stage="Downloading video...")
            path, keep = download_media(url), True
            update(jid, media_path=str(path), media_url=f"/api/media/{jid}")
        update(jid, stage="Uploading to AssemblyAI...")
        text, sents = transcribe(path, lambda s: update(jid, stage=s))
        if not text:
            raise RuntimeError("No speech was found in this media.")
        update(jid, sents=sents)
        summarize_job(jid, text, sents, lang)
    except Exception as e:
        update(jid, status="error", error=f"Could not process this media: {clean_error(e)}")
    finally:
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
#fname{color:var(--muted);font-size:13px}
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
#fname{cursor:pointer;background:#f8fafc;color:var(--text);font-size:14px}
#lang{max-width:230px}
</style></head>
<body>
<header class="top"><div class="top-in">
  <img src="/logo.webp" alt="Innoclique" class="logo"><span class="sep"></span>
  <span class="ttl">Media Summarizer</span>
</div></header>
<div class="wrap">

<div class="card">
  <div class="row">
    <input type="text" id="fname" readonly placeholder="No file chosen - click here or press Choose video">
    <input type="file" id="file" accept="video/*,audio/*" hidden>
    <button id="pick">Choose video</button>
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
let chosen=null, timer=null, topics=[], plain="", objUrl=null, jobId=null, summ=null, timed=[], baseName="video";
const fmt=s=>{s=Math.floor(s||0);const h=Math.floor(s/3600),m=Math.floor(s%3600/60),c=s%60,p=n=>String(n).padStart(2,"0");return (h?h+":":"")+p(m)+":"+p(c)};
const esc=t=>String(t).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const hl=t=>esc(t).replace(/\*\*(.+?)\*\*/g,"<b>$1</b>");

// ---- file / video ----
function setSrc(src){v.src=src;$("ph").style.display="none";}
const pickFile=()=>$("file").click();
$("pick").onclick=pickFile; $("fname").onclick=pickFile;
$("file").onchange=e=>{const f=e.target.files[0];if(!f)return;
  chosen=f;$("fname").value=f.name;baseName=f.name.replace(/\.[^.]+$/,"");jobId=null;summ=null;
  if(objUrl)URL.revokeObjectURL(objUrl);objUrl=URL.createObjectURL(f);setSrc(objUrl);};

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

// ---- summarize ----
$("go").onclick=async()=>{
  if(!chosen){alert("Please choose a video or audio file first.");return;}
  const fd=new FormData(); fd.append("file",chosen); fd.append("lang",$("lang").value);
  $("go").disabled=true; summ=null; showLoader("Starting...");
  try{
    const r=await fetch("/api/summarize",{method:"POST",body:fd}); const j=await r.json();
    if(!r.ok)throw new Error(j.error||"Request failed"); jobId=j.job_id; poll(jobId);
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
  clearInterval(timer);speechSynthesis.cancel();v.pause();v.removeAttribute("src");v.load();$("ph").style.display="flex";
  chosen=null;topics=[];plain="";jobId=null;summ=null;timed=[];baseName="video";
  $("file").value="";$("fname").value="";
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


# ---------- Innoclique logo (embedded so this single file is all you need) ----------
import base64
LOGO_B64 = "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"
LOGO_BYTES = base64.b64decode(LOGO_B64)


@app.route("/logo.webp")
def logo():
    resp = make_response(LOGO_BYTES)
    resp.headers["Content-Type"] = "image/webp"
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


if __name__ == "__main__":
    print(f".env file : {ENV_PATH} -> {'FOUND' if ENV_PATH.exists() else 'NOT FOUND'}")
    print(f"  ASSEMBLYAI_API_KEY: {'loaded' if ASSEMBLYAI_API_KEY else 'not set'}")
    print(f"  DEEPSEEK_API_KEY  : {'loaded' if DEEPSEEK_API_KEY else 'not set'}")
    app.run(debug=False, threaded=True)