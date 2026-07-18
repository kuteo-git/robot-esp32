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
    "pytube (yt:114)": "/tmp/robot-pytube.log",
    "weather (:8010)": "/tmp/robot-weather.log",
    "power-outage (:8011)": "/tmp/robot-poweroutage.log",
    "search (:8012)": "/tmp/robot-search.log",
    "r1-watchdog": "/tmp/robot-r1-watchdog.log",
    "claude-cli-adapter (out)": "/tmp/claude-adapter.log",
    "claude-cli-adapter (err)": "/tmp/claude-adapter.err",
}

app = FastAPI()


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
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0b0e14;color:#cdd6f4;font:13px/1.45 ui-monospace,Menlo,Consolas,monospace}
header{position:sticky;top:0;display:flex;gap:8px;flex-wrap:wrap;align-items:center;
  padding:8px 10px;background:#11151f;border-bottom:1px solid #222a3a}
select,input,button{background:#1b2030;color:#cdd6f4;border:1px solid #2a3346;border-radius:6px;
  padding:6px 9px;font:inherit}
button{cursor:pointer}
button.on{background:#2d6cdf;border-color:#2d6cdf}
.dot{width:9px;height:9px;border-radius:50%;background:#f38ba8;display:inline-block}
.dot.live{background:#a6e3a1}
#log{padding:8px 10px;white-space:pre-wrap;word-break:break-word}
#log .hi{background:#3a2f00}
.muted{color:#7f8aa3}
#cfg{display:none;gap:10px;flex-wrap:wrap;align-items:center;padding:8px 10px;background:#0f1420;border-bottom:1px solid #222a3a}
#cfg.show{display:flex}
#cfg label{display:flex;align-items:center;gap:5px}
</style></head><body>
<header>
  <span class="dot" id="dot"></span>
  <select id="src"></select>
  <input id="flt" placeholder="lọc (chữ con)..." size="18">
  <button id="pause">Tạm dừng</button>
  <button id="wrap" class="on">Wrap</button>
  <button id="clear">Xoá màn</button>
  <button id="cfgBtn">⚙ STT ảo giác</button>
  <span class="muted" id="cnt">0 dòng</span>
</header>
<div id="cfg">
  <b>STT chống ảo giác (whisper :8001 · live, khỏi restart):</b>
  <label><input type="checkbox" id="cfgVad"> VAD (lọc ồn/im)</label>
  <label>VAD ngưỡng <input id="cfgVadT" size="4"></label>
  <label>logprob min <input id="cfgLp" size="5"></label>
  <label>dur min <input id="cfgDur" size="4"></label>
  <button id="cfgSave" class="on">Lưu</button>
  <span class="muted" id="cfgMsg"></span>
</div>
<div id="log"></div>
<script>
const srcSel=document.getElementById('src'), log=document.getElementById('log'),
  flt=document.getElementById('flt'), dot=document.getElementById('dot'),
  cnt=document.getElementById('cnt'), pauseBtn=document.getElementById('pause');
const SOURCES=__SOURCES__;
SOURCES.forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=n;srcSel.appendChild(o)});
let es=null,paused=false,n=0;
function atBottom(){return window.innerHeight+window.scrollY>=document.body.scrollHeight-60}
function add(line){
  const f=flt.value.trim().toLowerCase();
  if(f && !line.toLowerCase().includes(f))return;
  const div=document.createElement('div');
  if(f){div.className='hi'}
  div.textContent=line; log.appendChild(div); n++;
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
srcSel.onchange=connect;
flt.oninput=()=>{};
pauseBtn.onclick=()=>{paused=!paused;pauseBtn.classList.toggle('on',paused);pauseBtn.textContent=paused?'Tiếp tục':'Tạm dừng'};
document.getElementById('wrap').onclick=function(){this.classList.toggle('on');
  log.style.whiteSpace=this.classList.contains('on')?'pre-wrap':'pre'};
document.getElementById('clear').onclick=()=>{log.innerHTML='';n=0;cnt.textContent='0 dòng'};
// --- STT anti-hallucination config (calls whisper :8001 directly, same host) ---
const WHOST=`http://${location.hostname}:8001`, cfg=document.getElementById('cfg');
const $c=id=>document.getElementById(id);
async function loadCfg(){try{const c=await(await fetch(WHOST+'/config')).json();
  $c('cfgVad').checked=c.vad_enabled;$c('cfgVadT').value=c.vad_threshold;
  $c('cfgLp').value=c.min_logprob;$c('cfgDur').value=c.min_dur;$c('cfgMsg').textContent='';
}catch(e){$c('cfgMsg').textContent='(không nối được whisper :8001)';}}
const setCfg=(k,v)=>fetch(`${WHOST}/config?key=${k}&value=${encodeURIComponent(v)}`,{method:'POST'});
$c('cfgBtn').onclick=()=>{cfg.classList.toggle('show');if(cfg.classList.contains('show'))loadCfg();};
$c('cfgSave').onclick=async()=>{
  await setCfg('vad_enabled',$c('cfgVad').checked?'1':'0');
  await setCfg('vad_threshold',$c('cfgVadT').value);
  await setCfg('min_logprob',$c('cfgLp').value);
  await setCfg('min_dur',$c('cfgDur').value);
  $c('cfgMsg').textContent='đã lưu ✓';setTimeout(()=>$c('cfgMsg').textContent='',1500);
};
connect();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    import json
    return PAGE.replace("__SOURCES__", json.dumps(list(LOGS.keys()), ensure_ascii=False))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
