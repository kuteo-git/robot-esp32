"""
Live log viewer for the robot (real-time tail over SSE). Default port 8009.
Open: http://<SERVER_IP>:8009
"""
import os
import asyncio
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

from _logsetup import make_logger, install_request_logging

PORT = int(os.environ.get("LOGWEB_PORT", "8009"))
# services/ is a sibling of xiaozhi-esp32-server/ at the repo root -> resolve relative to this file.
_REPO_ROOT = Path(__file__).resolve().parent.parent
BASE = os.environ.get(
    "XIAOZHI_LOG_PATH",
    str(_REPO_ROOT / "xiaozhi-esp32-server/main/xiaozhi-server/tmp/server.log"),
)
LOGS = {
    "xiaozhi (chi tiết)": BASE,
    "xiaozhi (stdout)": "/tmp/robot-xiaozhi.log",
    "vieneu (TTS)": "/tmp/robot-vieneu.log",
    "whisper (ASR)": "/tmp/robot-whisper.log",
    "moonshine (ASR)": "/tmp/robot-moonshine.log",
    "pytube (yt:114)": "/tmp/robot-pytube.log",
    "weather (:8010)": "/tmp/robot-weather.log",
    "power-outage (:8011)": "/tmp/robot-poweroutage.log",
    "search (:8012)": "/tmp/robot-search.log",
    "lunar (:8013)": "/tmp/robot-lunar.log",
    "news (:8014)": "/tmp/robot-news.log",
    "r1-watchdog": "/tmp/robot-r1-watchdog.log",
    "claude-cli-adapter (out)": "/tmp/claude-adapter.log",
    "claude-cli-adapter (err)": "/tmp/claude-adapter.err",
}

app = FastAPI()
log = install_request_logging(app, "logweb")


def last_lines(path, n=300):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            data = b""
            pos = size
            while pos > 0 and data.count(b"\n") <= n:
                step = min(8192, pos)
                pos -= step
                f.seek(pos)
                data = f.read(step) + data
            return data.decode("utf-8", "replace").splitlines()[-n:]
    except Exception as e:
        return [f"(không đọc được {path}: {e})"]


async def follow(path):
    for line in last_lines(path):
        yield line
    try:
        f = open(path, "r", errors="replace")
    except Exception as e:
        yield f"(không mở được {path}: {e})"
        return
    f.seek(0, 2)
    inode = os.fstat(f.fileno()).st_ino
    while True:
        line = f.readline()
        if line:
            yield line.rstrip("\n")
            continue
        await asyncio.sleep(0.4)
        # detect the file being rotated/replaced (service restart) -> reopen it
        try:
            if os.stat(path).st_ino != inode:
                f.close()
                f = open(path, "r", errors="replace")
                inode = os.fstat(f.fileno()).st_ino
        except Exception:
            pass


@app.get("/stream")
async def stream(name: str):
    path = LOGS.get(name)

    async def gen():
        if not path:
            yield f"data: (không có log '{name}')\n\n"
            return
        try:
            async for line in follow(path):
                yield "data: " + line.replace("\r", " ") + "\n\n"
        except asyncio.CancelledError:
            return

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot log live</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{color-scheme:dark;
  --bg:#0B0D10;--surface:#14171C;--surface-2:#1C2027;--elev:#242A33;--border:#262B33;
  --text:#F2F5F9;--muted:#8B95A5;--faint:#5C6675;
  --accent:#4C8DFF;--accent-2:#7B5CFF;
  --teal:#00B894;--violet:#7C5CFF;--amber:#F5A623;--red:#FF5A52;
  --ease:cubic-bezier(.22,1,.36,1);
  --radius-card:14px;--radius-control:10px;--radius-pill:999px;
}
:root[data-theme="light"]{
  --bg:#F7F8FA;--surface:#FFFFFF;--surface-2:#F0F2F6;--elev:#E8EBF1;--border:#E3E7EE;
  --text:#0E1116;--muted:#5B6572;--faint:#8B95A5;
}
@media (prefers-color-scheme: light){
  :root:not([data-theme="dark"]){
    --bg:#F7F8FA;--surface:#FFFFFF;--surface-2:#F0F2F6;--elev:#E8EBF1;--border:#E3E7EE;
    --text:#0E1116;--muted:#5B6572;--faint:#8B95A5;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font:13px/1.5 "Geist Mono",ui-monospace,Menlo,Consolas,monospace}
header{position:sticky;top:0;z-index:2;display:flex;gap:8px;flex-wrap:wrap;align-items:center;
  padding:10px 14px;background:var(--surface);border-bottom:1px solid var(--border);
  box-shadow:0 1px 2px rgba(0,0,0,.3),0 4px 16px rgba(0,0,0,.15)}
.brand{display:flex;align-items:center;gap:8px;margin-right:4px;font-family:"Geist",ui-sans-serif,system-ui;font-weight:600}
.brand .glyph{width:20px;height:20px;border-radius:6px;flex:none;
  background:linear-gradient(120deg,var(--accent),var(--accent-2))}
select,input,button{background:var(--surface-2);color:var(--text);border:1px solid var(--border);
  border-radius:var(--radius-control);padding:6px 10px;font:12px "Geist Mono",monospace;
  transition:background .15s var(--ease),border-color .15s var(--ease)}
button{cursor:pointer;border-radius:var(--radius-pill)}
button:hover{background:var(--elev)}
button.on{background:linear-gradient(120deg,var(--accent),var(--accent-2));border-color:transparent;color:#fff}
.dot{width:8px;height:8px;border-radius:50%;background:var(--red);display:inline-block;flex:none;
  box-shadow:0 0 0 3px color-mix(in srgb, var(--red) 25%, transparent)}
.dot.live{background:var(--teal);box-shadow:0 0 0 3px color-mix(in srgb, var(--teal) 25%, transparent)}
#log{padding:10px 14px;white-space:pre-wrap;word-break:break-word}
#log .line{padding:1px 0;border-radius:4px}
#log .hi{background:color-mix(in srgb, var(--amber) 20%, transparent)}
.muted{color:var(--muted)}
.tsel{appearance:none}
/* logcat-style level colors */
.lvl-FATAL{color:#fff;background:var(--red);padding:0 4px;border-radius:3px}
.lvl-ERROR{color:var(--red)}
.lvl-WARN{color:var(--amber)}
.lvl-INFO{color:var(--teal)}
.lvl-DEBUG{color:var(--accent)}
.lvl-TRACE{color:var(--faint)}
.tok-date{color:var(--faint)}
.tok-ver{color:var(--violet)}
@media (max-width:640px){
  header{padding:8px 10px;gap:6px}
  .brand{width:100%;order:-2}
  .dot{order:-1}
  select,#flt{flex:1 1 auto;min-width:0}
  button{padding:7px 10px}
  #cnt{width:100%;order:9;text-align:right}
  #log{padding:8px 10px;font-size:12px}
}
</style></head><body>
<header>
  <span class="brand"><span class="glyph"></span>Robot log</span>
  <span class="dot" id="dot"></span>
  <select id="src" class="tsel"></select>
  <input id="flt" placeholder="lọc (chữ con)..." size="18">
  <button id="pause">Tạm dừng</button>
  <button id="wrap" class="on">Wrap</button>
  <button id="clear">Xoá màn</button>
  <button id="theme">☀/☾</button>
  <span class="muted" id="cnt">0 dòng</span>
</header>
<div id="log"></div>
<script>
(function(){
  var saved=localStorage.getItem('relay-theme');
  if(saved)document.documentElement.setAttribute('data-theme',saved);
})();
document.getElementById('theme').onclick=function(){
  var cur=document.documentElement.getAttribute('data-theme');
  var next=cur==='light'?'dark':'light';
  document.documentElement.setAttribute('data-theme',next);
  localStorage.setItem('relay-theme',next);
};
// logcat-style classification: date/time, version tokens, level keywords
var DATE_RE=/\\b(\\d{4}-\\d{2}-\\d{2}[ T]\\d{2}:\\d{2}:\\d{2}(?:[.,]\\d+)?|\\d{2}:\\d{2}:\\d{2}(?:[.,]\\d+)?)\\b/;
var VER_RE=/\\b(v?\\d+\\.\\d+(?:\\.\\d+)?(?:-[a-zA-Z0-9]+)?)\\b/;
var LEVEL_RE=/\\b(FATAL|CRITICAL|ERROR|WARN(?:ING)?|INFO|DEBUG|TRACE)\\b/;
function escapeHtml(s){
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function colorize(line){
  var lvlClass='';
  var m=line.match(LEVEL_RE);
  if(m){
    var lv=m[1].toUpperCase();
    if(lv==='WARNING')lv='WARN';
    if(lv==='CRITICAL')lv='FATAL';
    lvlClass='lvl-'+lv;
  }
  var html=escapeHtml(line);
  html=html.replace(DATE_RE,function(x){return '<span class="tok-date">'+x+'</span>'});
  html=html.replace(VER_RE,function(x){return '<span class="tok-ver">'+x+'</span>'});
  if(m){
    var esc=escapeHtml(m[1]);
    html=html.replace(esc,'<span class="'+lvlClass+'">'+esc+'</span>');
  }
  return html;
}
</script>
<script>
const srcSel=document.getElementById('src'), log=document.getElementById('log'),
  flt=document.getElementById('flt'), dot=document.getElementById('dot'),
  cnt=document.getElementById('cnt'), pauseBtn=document.getElementById('pause');
const SOURCES=__SOURCES__;
SOURCES.forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=n;srcSel.appendChild(o)});
const wrapBtn=document.getElementById('wrap');
const ST_KEY='logweb-state';
function loadState(){try{return JSON.parse(localStorage.getItem(ST_KEY))||{};}catch(e){return {};}}
function saveState(patch){const s=Object.assign(loadState(),patch);localStorage.setItem(ST_KEY,JSON.stringify(s));}
const initial=loadState();
if(initial.src && SOURCES.includes(initial.src))srcSel.value=initial.src;
if(typeof initial.flt==='string')flt.value=initial.flt;
if(initial.wrap===false){wrapBtn.classList.remove('on');log.style.whiteSpace='pre';}
let es=null,paused=!!initial.paused,n=0;
if(paused){pauseBtn.classList.add('on');pauseBtn.textContent='Tiếp tục';}
function atBottom(){return window.innerHeight+window.scrollY>=document.body.scrollHeight-60}
function applyFilterToLine(div,f){
  const show=!f || div.dataset.raw.toLowerCase().includes(f);
  div.style.display=show?'':'none';
  div.classList.toggle('hi',!!f && show);
}
function reapplyFilter(){
  const f=flt.value.trim().toLowerCase();
  for(const div of log.children)applyFilterToLine(div,f);
  if(!paused && atBottom())window.scrollTo(0,document.body.scrollHeight);
}
function add(line){
  const div=document.createElement('div');
  div.className='line';
  div.dataset.raw=line;
  div.innerHTML=colorize(line);
  log.appendChild(div); n++;
  applyFilterToLine(div,flt.value.trim().toLowerCase());
  while(log.childNodes.length>4000)log.removeChild(log.firstChild);
  cnt.textContent=n+' dòng';
  if(!paused && atBottom())window.scrollTo(0,document.body.scrollHeight);
}
function connect(){
  if(es)es.close(); log.innerHTML=''; n=0;
  es=new EventSource('/stream?name='+encodeURIComponent(srcSel.value));
  es.onopen=()=>dot.classList.add('live');
  es.onerror=()=>dot.classList.remove('live');
  es.onmessage=e=>{if(!paused)add(e.data)};
}
srcSel.onchange=()=>{saveState({src:srcSel.value});connect();};
flt.oninput=()=>{saveState({flt:flt.value});reapplyFilter();};
pauseBtn.onclick=()=>{paused=!paused;pauseBtn.classList.toggle('on',paused);pauseBtn.textContent=paused?'Tiếp tục':'Tạm dừng';saveState({paused});};
wrapBtn.onclick=function(){this.classList.toggle('on');
  log.style.whiteSpace=this.classList.contains('on')?'pre-wrap':'pre';saveState({wrap:this.classList.contains('on')});};
document.getElementById('clear').onclick=()=>{log.innerHTML='';n=0;cnt.textContent='0 dòng'};
connect();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    import json
    return PAGE.replace("__SOURCES__", json.dumps(list(LOGS.keys()), ensure_ascii=False))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
