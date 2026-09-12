"""Frame-stepping ground-truth labeller.

Serves an extracted run and writes (frame, time, yard) labels to CSV. Frames are
shown unannotated so labels do not inherit the detector's errors.
"""

import argparse
import csv
import json
import pathlib
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = None
OUT = None

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>Yard labeller</title>
<style>
:root{--bg:#0d0f12;--panel:#161a20;--line:#262c35;--ink:#e8ecf2;--mut:#8b95a3;
      --acc:#35c1d6;--ok:#58d68d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif;
     display:grid;grid-template-columns:1fr 330px;height:100vh}
main{padding:14px;display:flex;flex-direction:column;gap:10px;min-width:0}
#wrap{flex:1;display:flex;align-items:center;justify-content:center;
      background:#000;border:1px solid var(--line);border-radius:8px;overflow:hidden}
#img{max-width:100%;max-height:100%;display:block}
.bar{display:flex;align-items:center;gap:12px}
.bar b{font-variant-numeric:tabular-nums;font-size:18px}
input[type=range]{flex:1;accent-color:var(--acc)}
aside{border-left:1px solid var(--line);padding:16px;overflow:auto;background:var(--panel)}
h2{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--mut);
   margin:0 0 10px}
.yard{display:flex;gap:8px;margin-bottom:10px}
.yard input{flex:1;background:#0e1116;border:1px solid var(--line);color:var(--ink);
   border-radius:6px;padding:9px 10px;font-size:17px;font-variant-numeric:tabular-nums}
button{background:#1d2530;border:1px solid var(--line);color:var(--ink);
   border-radius:6px;padding:9px 12px;cursor:pointer;font-size:13px}
button:hover{border-color:var(--acc)}
button.pri{background:var(--acc);border-color:var(--acc);color:#06232a;font-weight:600}
button.ok{background:var(--ok);border-color:var(--ok);color:#07301c;font-weight:600}
.quick{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-bottom:16px}
.quick button{padding:7px 0;font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:13px;
      font-variant-numeric:tabular-nums}
td,th{padding:6px 4px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mut);font-weight:500;font-size:11px;letter-spacing:.08em;
   text-transform:uppercase}
tr.cur{background:#1b2b31}
td.del{text-align:right;color:var(--mut);cursor:pointer}
td.del:hover{color:#ff8fab}
kbd{background:#0e1116;border:1px solid var(--line);border-radius:4px;
    padding:1px 6px;font-size:11px}
.hint{color:var(--mut);font-size:12px;margin-top:14px;line-height:1.7}
#msg{margin-top:10px;font-size:12px;color:var(--ok);min-height:16px}
</style>
<main>
  <div id="wrap"><img id="img" alt=""></div>
  <div class="bar">
    <b id="fno">—</b><span id="tt" style="color:var(--mut)"></span>
    <input type="range" id="sl" min="1" value="1">
  </div>
</main>
<aside>
  <h2>Label this frame</h2>
  <div style="margin-bottom:12px">
    <label style="display:flex;gap:8px;align-items:center;cursor:pointer;color:var(--mut)">
      <input type="checkbox" id="ov" style="accent-color:var(--acc)">
      show detector overlay
    </label>
  </div>
  <div class="yard">
    <input id="y" type="number" step="0.5" placeholder="yard" autocomplete="off">
    <button class="pri" id="add">Add</button>
  </div>
  <div class="quick" id="quick"></div>
  <h2>Labels (<span id="cnt">0</span>)</h2>
  <table><thead><tr><th>frame</th><th>t</th><th>yard</th><th></th></tr></thead>
  <tbody id="rows"></tbody></table>
  <div style="margin-top:14px"><button class="ok" id="save">Save CSV</button></div>
  <div id="msg"></div>
  <div class="hint">
    <kbd>←</kbd><kbd>→</kbd> step 1 &nbsp; <kbd>shift</kbd>+ step 10<br>
    <kbd>enter</kbd> add label &nbsp; <kbd>del</kbd> remove current<br><br>
    Mark the frame where the athlete's <b>hip</b> is over a known mark.
    Label with the overlay <b>off</b> — it is there to inspect the detector,
    not to label against.<br><br>
    magenta = 5-yard line &middot; yellow = lane centreline &middot;
    green dot = where the line meets the centreline &middot;
    white = pose &middot; cyan cross = hip centre
  </div>
</aside>
<script>
let M={}, i=1, L=[];
const img=document.getElementById('img'), sl=document.getElementById('sl');
const fno=document.getElementById('fno'), tt=document.getElementById('tt');
const yEl=document.getElementById('y'), msg=document.getElementById('msg');

fetch('/meta.json').then(r=>r.json()).then(m=>{
  M=m; sl.max=m.n; show(1);
  [5,10,15,20,25,30,35,40,45,50].forEach(v=>{
    const b=document.createElement('button');
    b.textContent=v; b.onclick=()=>{yEl.value=v; add();};
    document.getElementById('quick').appendChild(b);
  });
});
function tOf(n){ return (n-1)/M.fps + M.clip_start - M.t_zero; }
function show(n){
  i=Math.max(1,Math.min(M.n,n)); sl.value=i;
  const dir=document.getElementById('ov').checked?'overlay':'frames';
  img.src='/'+dir+'/f'+String(i).padStart(4,'0')+'.jpg';
  fno.textContent='frame '+i+' / '+M.n;
  tt.textContent='t = '+tOf(i).toFixed(3)+' s';
  draw();
}
sl.oninput=()=>show(+sl.value);
document.getElementById('ov').onchange=()=>show(i);
function add(){
  const v=parseFloat(yEl.value);
  if(isNaN(v)){yEl.focus();return;}
  L=L.filter(r=>r.frame!==i);
  L.push({frame:i,t:+tOf(i).toFixed(4),yard:v});
  L.sort((a,b)=>a.frame-b.frame); yEl.value=''; draw();
}
function draw(){
  document.getElementById('cnt').textContent=L.length;
  document.getElementById('rows').innerHTML=L.map(r=>
    `<tr class="${r.frame===i?'cur':''}"><td><a href="#" onclick="show(${r.frame});return false"
     style="color:var(--acc)">${r.frame}</a></td><td>${r.t.toFixed(3)}</td>
     <td>${r.yard}</td><td class="del" onclick="rm(${r.frame})">remove</td></tr>`).join('');
}
function rm(f){ L=L.filter(r=>r.frame!==f); draw(); }
document.getElementById('add').onclick=add;
document.getElementById('save').onclick=()=>{
  fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({meta:M,labels:L})})
   .then(r=>r.json()).then(j=>{msg.textContent='Saved '+j.n+' labels to '+j.path;});
};
addEventListener('keydown',e=>{
  if(e.key==='ArrowRight'){show(i+(e.shiftKey?10:1));e.preventDefault();}
  else if(e.key==='ArrowLeft'){show(i-(e.shiftKey?10:1));e.preventDefault();}
  else if(e.key==='Enter'){add();}
  else if(e.key==='Delete'||e.key==='Backspace'){
    if(document.activeElement!==yEl){rm(i);e.preventDefault();}
  }
});
</script>
"""


def _safe(x):
    s = "" if x is None else str(x)
    return "'" + s if s[:1] in "=+-@\t\r" else s


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _local(self):
        host = self.headers.get("Host", "").split(":")[0]
        if host not in ("localhost", "127.0.0.1", "[::1]", "::1"):
            self._send(403, b"bad host", "text/plain")
            return False
        origin = self.headers.get("Origin")
        if origin and origin.split("://")[-1].split(":")[0] not in (
                "localhost", "127.0.0.1"):
            self._send(403, b"bad origin", "text/plain")
            return False
        return True

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._local():
            return
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if self.path == "/meta.json":
            return self._send(200, (ROOT / "meta.json").read_bytes(),
                              "application/json")
        for d in ("frames", "overlay"):
            if self.path.startswith(f"/{d}/"):
                p = ROOT / d / pathlib.Path(self.path).name
                if p.exists():
                    return self._send(200, p.read_bytes(), "image/jpeg")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if not self._local():
            return
        if self.path != "/save":
            return self._send(404, b"no", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        rows = data.get("labels", [])
        meta = json.loads((ROOT / "meta.json").read_text())
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "player_name", "frame", "t_rel_s", "yard"])
            for r in rows:
                w.writerow([_safe(meta.get("video_id")),
                            _safe(meta.get("athlete")),
                            int(r["frame"]), float(r["t"]), float(r["yard"])])
        self._send(200, json.dumps({"n": len(rows), "path": str(OUT)}).encode(),
                   "application/json")


def main() -> int:
    global ROOT, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/labels.csv")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    ROOT = pathlib.Path(a.root)
    OUT = pathlib.Path(a.out)
    print(f"labelling {ROOT} -> {OUT} on http://localhost:{a.port}", file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
