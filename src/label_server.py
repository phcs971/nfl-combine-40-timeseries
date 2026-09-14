"""Hand labeller for the planned runs.

Serves each run's frames, parks the viewer on the sprint model's predicted frame
for the mark being labelled, and merges the confirmed crossings into one CSV.
Frames are served unannotated by default: the detector overlay is a toggle, so a
label is never read off the detector's own guess.
"""

import argparse
import csv
import io
import json
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import pandas as pd

COLS = ["video_id", "bib", "player_name", "cls", "split", "frame", "t_rel_s", "yard"]

ROOT = OUT = None
RUNS: list[dict] = []
_LOCK = threading.Lock()
_OVERLAY_LOCK = threading.Lock()

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>40-yard labeller</title>
<style>
:root{--bg:#0d0f12;--panel:#161a20;--line:#262c35;--ink:#e8ecf2;--mut:#8b95a3;
      --acc:#35c1d6;--ok:#58d68d;--warn:#f0b429}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif;
     display:grid;grid-template-columns:1fr 360px;height:100vh}
main{padding:12px;display:flex;flex-direction:column;gap:9px;min-width:0}
#wrap{flex:1;background:#000;border:1px solid var(--line);border-radius:8px;
      overflow:hidden;display:flex;align-items:center;justify-content:center;
      cursor:zoom-in;position:relative}
#wrap.z{cursor:zoom-out}
#img{max-width:100%;max-height:100%;display:block;transition:transform .12s}
.bar{display:flex;align-items:center;gap:12px}
.bar b{font-variant-numeric:tabular-nums;font-size:17px;min-width:130px}
input[type=range]{flex:1;accent-color:var(--acc)}
aside{border-left:1px solid var(--line);padding:14px;overflow:auto;
      background:var(--panel)}
h2{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--mut);
   margin:16px 0 8px}
h2:first-child{margin-top:0}
select,button{background:#1d2530;border:1px solid var(--line);color:var(--ink);
   border-radius:6px;padding:8px 10px;cursor:pointer;font-size:13px;font-family:inherit}
select{width:100%}
button:hover{border-color:var(--acc)}
button.pri{background:var(--acc);border-color:var(--acc);color:#06232a;font-weight:600}
button.ok{background:var(--ok);border-color:var(--ok);color:#07301c;font-weight:600}
.row{display:flex;gap:8px;align-items:center}
.meta{color:var(--mut);font-size:12px;margin:8px 0 0;line-height:1.65}
.meta b{color:var(--ink);font-weight:600}
table{width:100%;border-collapse:collapse;font-size:13px;
      font-variant-numeric:tabular-nums}
td,th{padding:5px 4px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mut);font-weight:500;font-size:10px;letter-spacing:.08em;
   text-transform:uppercase}
tr.cur td{background:#1b2b31}
tr.done td:first-child{color:var(--ok)}
td.a{text-align:right;color:var(--mut);cursor:pointer;font-size:12px}
td.a:hover{color:var(--acc)}
.pill{display:inline-block;padding:1px 7px;border-radius:99px;font-size:11px;
      border:1px solid var(--line);color:var(--mut)}
kbd{background:#0e1116;border:1px solid var(--line);border-radius:4px;
    padding:1px 6px;font-size:11px}
.hint{color:var(--mut);font-size:12px;margin-top:14px;line-height:1.8}
#msg{margin-top:8px;font-size:12px;color:var(--ok);min-height:16px}
#prog{height:5px;background:#0e1116;border-radius:3px;overflow:hidden;margin-top:8px}
#prog i{display:block;height:100%;background:var(--ok)}
</style>
<main>
  <div id="wrap"><img id="img" alt=""></div>
  <div class="bar">
    <b id="fno">—</b><span id="tt" style="color:var(--mut)"></span>
    <input type="range" id="sl" min="1" value="1">
    <label style="color:var(--mut);display:flex;gap:6px;align-items:center;cursor:pointer">
      <input type="checkbox" id="ov" style="accent-color:var(--acc)">overlay <kbd>o</kbd>
    </label>
  </div>
</main>
<aside>
  <h2>Run <span id="rpos" class="pill"></span></h2>
  <select id="pick"></select>
  <div class="meta" id="rmeta"></div>
  <div id="prog"><i style="width:0"></i></div>
  <div class="row" style="margin-top:8px">
    <button id="prev">&larr; run</button><button id="next">run &rarr;</button>
    <button id="todo" style="margin-left:auto">next unlabelled</button>
  </div>

  <h2>Marks</h2>
  <table><thead><tr><th>yard</th><th>frame</th><th>t</th><th></th></tr></thead>
  <tbody id="rows"></tbody></table>
  <div class="row" style="margin-top:10px">
    <button class="pri" id="set" style="flex:1">Set mark at this frame <kbd>&crarr;</kbd></button>
  </div>
  <div class="row" style="margin-top:8px">
    <button class="ok" id="save" style="flex:1">Save</button>
    <button id="clear">Clear run</button>
  </div>
  <div id="msg"></div>
  <div class="hint">
    <kbd>&larr;</kbd><kbd>&rarr;</kbd> step 1 &middot; <kbd>shift</kbd> 10 &nbsp;
    <kbd>&uarr;</kbd><kbd>&darr;</kbd> mark<br>
    <kbd>&crarr;</kbd> set mark &amp; go to next &middot; <kbd>x</kbd> clear mark<br>
    <kbd>[</kbd> <kbd>]</kbd> run &middot; <kbd>o</kbd> overlay &middot; click image to zoom<br><br>
    Set the frame where the athlete's <b>hip</b> is over the mark. The frame
    offered is the sprint model's guess, usually within a frame or two.<br><br>
    Overlay: magenta = candidate yard lines (deliberately over-generous),
    bright = vanishing-point pick, green dot = where it meets the lane
    centreline, yellow = lane centreline.
  </div>
</aside>
<script>
let P=[], R=null, M={}, i=1, L={}, mi=0, zoom=false;
const $=id=>document.getElementById(id);
const img=$('img'), sl=$('sl');

fetch('/api/plan').then(r=>r.json()).then(p=>{
  P=p; $('pick').innerHTML=P.map((r,k)=>
    `<option value="${k}">${k+1}. ${r.athlete} — ${r.cls}/${r.split} · ${r.forty}s</option>`).join('');
  open(+(localStorage.getItem('run')||0));
});
function open(k){
  k=Math.max(0,Math.min(P.length-1,k)); localStorage.setItem('run',k);
  fetch('/api/run/'+k).then(r=>r.json()).then(m=>{
    R=k; M=m; L={}; (m.labels||[]).forEach(r=>L[r.yard]=r.frame);
    $('pick').value=k; $('rpos').textContent=(k+1)+' / '+P.length;
    $('rmeta').innerHTML=`<b>${m.athlete}</b> · ${m.pos} · bib ${m.bib}<br>`+
      `${m.video_id} · ${m.cls}/${m.split} · official <b>${m.official_forty}</b>s`+
      ` · ${m.drafted?'drafted':'undrafted'} · ${m.n} frames`;
    mi=firstTodo(); goMark(mi); draw();
  });
}
function firstTodo(){
  for(let k=0;k<M.marks.length;k++) if(!(M.marks[k] in L)) return k;
  return 0;
}
function tOf(n){ return (n-1)/M.fps + M.clip_start - M.t_zero; }
function frameOf(t){ return Math.round((t-(M.clip_start-M.t_zero))*M.fps)+1; }
function goMark(k){
  mi=Math.max(0,Math.min(M.marks.length-1,k));
  const y=M.marks[mi];
  show(y in L ? L[y] : (M.seed[y]!=null ? frameOf(M.seed[y]) : i));
}
function show(n){
  i=Math.max(1,Math.min(M.n,n)); sl.max=M.n; sl.value=i;
  const d=$('ov').checked?'overlay':'frame';
  img.src=`/${d}/${R}/${i}`;
  $('fno').textContent='frame '+i+' / '+M.n;
  $('tt').textContent='t = '+tOf(i).toFixed(3)+' s';
  draw();
}
function draw(){
  const done=M.marks.filter(y=>y in L).length;
  $('prog').firstChild.style.width=(100*done/M.marks.length)+'%';
  $('rows').innerHTML=M.marks.map((y,k)=>{
    const f=L[y], has=f!=null;
    const seed=M.seed[y]!=null?frameOf(M.seed[y]):null;
    const d=has&&seed!=null?(f-seed):null;
    return `<tr class="${k===mi?'cur':''} ${has?'done':''}">
      <td>${y}</td>
      <td>${has?f:(seed!=null?'<span style="color:var(--mut)">~'+seed+'</span>':'—')}</td>
      <td>${has?tOf(f).toFixed(3):''}</td>
      <td class="a" onclick="pickMark(${k})">${has?(d>0?'+':'')+d:'go'}</td></tr>`;
  }).join('');
  $('pick').selectedIndex=R;
}
function pickMark(k){ goMark(k); }
function setMark(){
  L[M.marks[mi]]=i;
  if(mi<M.marks.length-1) goMark(mi+1); else draw();
}
function clearMark(){ delete L[M.marks[mi]]; draw(); }
function save(){
  const rows=M.marks.filter(y=>y in L)
    .map(y=>({yard:y,frame:L[y],t:+tOf(L[y]).toFixed(4)}));
  fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({run:R,labels:rows})})
   .then(r=>r.json()).then(j=>{
     $('msg').textContent=`saved ${j.n} marks · ${j.total} rows in ${j.path}`;
     P[R].done=rows.length;
   });
}
$('pick').onchange=e=>open(+e.target.value);
$('prev').onclick=()=>open(R-1);
$('next').onclick=()=>open(R+1);
$('todo').onclick=()=>{
  fetch('/api/plan').then(r=>r.json()).then(p=>{
    P=p; const k=P.findIndex((r,j)=>j>R&&r.done<8);
    open(k>=0?k:P.findIndex(r=>r.done<8));
  });
};
$('set').onclick=setMark;
$('save').onclick=save;
$('clear').onclick=()=>{L={};draw();};
sl.oninput=()=>show(+sl.value);
$('ov').onchange=()=>show(i);
$('wrap').onclick=e=>{
  const r=img.getBoundingClientRect();
  zoom=!zoom; $('wrap').classList.toggle('z',zoom);
  img.style.transformOrigin=
    (100*(e.clientX-r.left)/r.width)+'% '+(100*(e.clientY-r.top)/r.height)+'%';
  img.style.transform=zoom?'scale(2.6)':'none';
};
addEventListener('keydown',e=>{
  if(e.target.tagName==='SELECT') return;
  const k=e.key;
  if(k==='ArrowRight'){show(i+(e.shiftKey?10:1));}
  else if(k==='ArrowLeft'){show(i-(e.shiftKey?10:1));}
  else if(k==='ArrowUp'){goMark(mi-1);}
  else if(k==='ArrowDown'){goMark(mi+1);}
  else if(k==='Enter'){setMark();}
  else if(k==='x'){clearMark();}
  else if(k===']'){open(R+1);}
  else if(k==='['){open(R-1);}
  else if(k==='o'){$('ov').checked=!$('ov').checked;show(i);}
  else if(k==='s'&&(e.metaKey||e.ctrlKey)){save();}
  else return;
  e.preventDefault();
});
</script>
"""


def _safe(x):
    s = "" if x is None or (isinstance(x, float) and x != x) else str(x)
    return "'" + s if s[:1] and s[0] in "=+-@\t\r" else s


def load_runs(plan: pd.DataFrame, root: pathlib.Path) -> list[dict]:
    runs = []
    for row in plan.itertuples():
        d = root / f"{row.video_id}_{int(row.bib):02d}"
        if not (d / "meta.json").exists():
            print(f"skip {d.name}: not extracted", file=sys.stderr)
            continue
        runs.append({"dir": d, "meta": json.loads((d / "meta.json").read_text())})
    return runs


def existing() -> pd.DataFrame:
    if not OUT.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(OUT)
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    df = df[COLS].copy()
    for c in ("bib", "cls", "split"):
        df[c] = df[c].astype(object)
    # labels written before a run joined the plan carry no bib/class/split
    for r in RUNS:
        m = r["meta"]
        hit = (df.video_id == m["video_id"]) & (df.player_name == m["athlete"])
        for c, v in (("bib", m["bib"]), ("cls", m["cls"]), ("split", m["split"])):
            df.loc[hit & df[c].isna(), c] = v
    return df


def labels_for(run: dict) -> list[dict]:
    m = run["meta"]
    df = existing()
    mine = df[(df.video_id == m["video_id"]) & (df.player_name == m["athlete"])]
    out = []
    for r in mine.itertuples():
        # frame is re-derived: a clip re-extracted with different padding
        # renumbers frames, while t_rel_s is fixed to the run.
        f = round((float(r.t_rel_s) - (m["clip_start"] - m["t_zero"])) * m["fps"]) + 1
        if 1 <= f <= m["n"]:
            out.append({"yard": int(float(r.yard)), "frame": int(f)})
    return out


def render_overlay(run: dict, n: int) -> bytes:
    import cv2
    import label_overlay

    src = run["dir"] / "frames" / f"f{n:04d}.jpg"
    cache = run["dir"] / "overlay"
    cache.mkdir(exist_ok=True)
    dst = cache / src.name
    if dst.exists():
        return dst.read_bytes()
    with _OVERLAY_LOCK:
        if dst.exists():
            return dst.read_bytes()
        im = cv2.imread(str(src))
        cv2.imwrite(str(dst), label_overlay.draw(im),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    return dst.read_bytes()


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

    def _json(self, obj):
        self._send(200, json.dumps(obj).encode(), "application/json")

    def _parts(self):
        return urlparse(self.path).path.strip("/").split("/")

    def do_GET(self):
        if not self._local():
            return
        p = self._parts()
        if p == [""] or p == ["index.html"]:
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")

        if p[:2] == ["api", "plan"]:
            df = existing()
            counts = df.groupby("player_name").size().to_dict() if len(df) else {}
            return self._json([{
                "athlete": r["meta"]["athlete"], "cls": r["meta"]["cls"],
                "split": r["meta"]["split"], "forty": r["meta"]["official_forty"],
                "done": int(counts.get(r["meta"]["athlete"], 0)),
            } for r in RUNS])

        if p[:2] == ["api", "run"] and len(p) == 3:
            run = self._run(p[2])
            if run is None:
                return
            m = dict(run["meta"])
            m["seed"] = {str(k): v for k, v in m["seed"].items()}
            m["labels"] = labels_for(run)
            return self._json(m)

        if p[0] in ("frame", "overlay") and len(p) == 3:
            run = self._run(p[1])
            if run is None:
                return
            try:
                n = int(p[2])
            except ValueError:
                return self._send(400, b"bad frame", "text/plain")
            if not (1 <= n <= run["meta"]["n"]):
                return self._send(404, b"no frame", "text/plain")
            if p[0] == "frame":
                f = run["dir"] / "frames" / f"f{n:04d}.jpg"
                return self._send(200, f.read_bytes(), "image/jpeg")
            try:
                return self._send(200, render_overlay(run, n), "image/jpeg")
            except Exception as e:
                print(f"overlay failed: {e}", file=sys.stderr)
                f = run["dir"] / "frames" / f"f{n:04d}.jpg"
                return self._send(200, f.read_bytes(), "image/jpeg")

        self._send(404, b"not found", "text/plain")

    def _run(self, s):
        try:
            k = int(s)
        except ValueError:
            k = -1
        if not (0 <= k < len(RUNS)):
            self._send(404, b"no run", "text/plain")
            return None
        return RUNS[k]

    def do_POST(self):
        if not self._local():
            return
        if self._parts() != ["api", "save"]:
            return self._send(404, b"no", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        run = self._run(data.get("run"))
        if run is None:
            return
        m = run["meta"]
        rows = data.get("labels", [])
        with _LOCK:
            df = existing()
            keep = df[~((df.video_id == m["video_id"]) &
                        (df.player_name == m["athlete"]))]
            add = pd.DataFrame([{
                "video_id": m["video_id"], "bib": m["bib"],
                "player_name": m["athlete"], "cls": m["cls"], "split": m["split"],
                "frame": int(r["frame"]), "t_rel_s": float(r["t"]),
                "yard": float(r["yard"]),
            } for r in rows], columns=COLS)
            out = pd.concat([keep, add], ignore_index=True)
            out = out.sort_values(["video_id", "bib", "yard"])
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(COLS)
            for r in out.itertuples(index=False):
                w.writerow([_safe(v) for v in r])
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(buf.getvalue())
        self._json({"n": len(rows), "total": len(out), "path": str(OUT)})


def main() -> int:
    global ROOT, OUT, RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="data/label_plan.csv")
    ap.add_argument("--root", default="label")
    ap.add_argument("--out", default="data/labels.csv")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    ROOT = pathlib.Path(a.root)
    OUT = pathlib.Path(a.out)
    RUNS = load_runs(pd.read_csv(a.plan), ROOT)
    if not RUNS:
        print("no extracted runs; run src/extract_run.py first", file=sys.stderr)
        return 1
    print(f"{len(RUNS)} runs -> {OUT}   http://localhost:{a.port}", file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
