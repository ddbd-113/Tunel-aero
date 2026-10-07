"""Odtwarzacz 3D lotu (three.js) - samodzielny plik HTML z danymi lotu."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .math3d import dcm_to_quat, quat_to_dcm

# NED -> układ three.js (x = wschód, y = góra, z = południe)
P_NED_TO_THREE = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])


def _frames(log, rate: float = 20.0) -> dict:
    d = log.data
    t = d["t"]
    step = max(1, int(round(1.0 / (rate * max(float(np.median(np.diff(t))), 1e-6))))) if len(t) > 1 else 1
    idx = np.arange(0, len(t), step)
    if idx[-1] != len(t) - 1:
        idx = np.append(idx, len(t) - 1)
    q3 = []
    for i in idx:
        R = quat_to_dcm(np.array([d["qw"][i], d["qx"][i], d["qy"][i], d["qz"][i]]))
        q = dcm_to_quat(P_NED_TO_THREE @ R)
        q3.append([round(float(q[1]), 4), round(float(q[2]), 4), round(float(q[3]), 4), round(float(q[0]), 4)])

    def col(k, dec=2):
        return [round(float(v), dec) for v in d[k][idx]] if k in d else []

    return {
        "t": col("t"), "n": col("pn"), "e": col("pe"), "h": [round(float(-v), 2) for v in d["pd"][idx]],
        "alt": col("alt"), "q": q3, "tas": col("tas"), "ias": col("est_ias"), "gs": col("gs"),
        "wn": col("wind_n"), "we": col("wind_e"), "wd": col("wind_d"), "soc": col("batt_soc", 3),
        "v": col("batt_v"), "thr": col("throttle"), "temp": col("temp_c", 1), "ice": col("ice", 2),
        "phase": [str(p) for p in d["phase"][idx]], "en": col("est_pn"), "ee": col("est_pe"),
        "gps": col("gps_ok", 0), "roll": col("roll", 1), "pitch": col("pitch", 1),
    }


def write_replay(result, path) -> Path:
    sc = result.scenario
    wind = ((sc.get("environment", {}) or {}).get("wind", {}) or {})
    wps = []
    for w in (sc.get("mission", {}) or {}).get("waypoints", []) or []:
        wps.append([w["north"], w["east"], w["alt"]] if isinstance(w, dict) else [w[0], w[1], w[2]])
    data = {
        "name": result.name,
        "vehicle": result.log.meta.get("vehicle", ""),
        "kind": result.log.meta.get("vehicle_kind", "multirotor"),
        "passed": result.passed,
        "end": result.log.meta.get("end_reason", ""),
        "frames": _frames(result.log),
        "events": [[float(t), m] for t, m in result.log.events],
        "waypoints": wps,
        "microbursts": [{"n": m.get("north", 0), "e": m.get("east", 0), "r": m.get("radius", 600),
                         "t0": m.get("t_start", 0), "dn": m.get("drift_north", 0), "de": m.get("drift_east", 0)}
                        for m in wind.get("microbursts", [])],
        "thermals": [{"n": th.get("north", 0), "e": th.get("east", 0), "r": th.get("radius", 80)}
                     for th in wind.get("thermals", [])],
        "icing": (sc.get("environment", {}) or {}).get("icing"),
    }
    vis = (sc.get("vehicle") or {}).get("_visual") if isinstance(sc.get("vehicle"), dict) else None
    if vis is not None:   # siatka z CAD (układ ciała FRD, metry, względem środka ciężkości)
        data["mesh"] = {"v": [round(float(x), 4) for x in np.asarray(vis["v"]).reshape(-1)],
                        "f": [int(i) for i in np.asarray(vis["f"]).reshape(-1)]}
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("__TITLE__", result.name.replace("<", ""))
    p = Path(path)
    p.write_text(html, encoding="utf-8")
    return p


TEMPLATE = r"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Odtwarzacz lotu - __TITLE__</title>
<style>
:root{--panel:rgba(18,22,28,.82);--fg:#e9edf2;--muted:#9aa4b1;--accent:#5fb0ff;--bad:#ff6b5e;--ok:#4cc38a}
html,body{margin:0;height:100%;background:#0d1117;color:var(--fg);font:13px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;overflow:hidden}
#c{position:fixed;inset:0}
.panel{position:fixed;background:var(--panel);border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:10px 12px;backdrop-filter:blur(6px)}
#hud{top:12px;left:12px;min-width:220px;max-width:calc(100vw - 24px)}
#hud h1{font-size:14px;margin:0 0 6px;max-width:340px}
#hud table{border-collapse:collapse}#hud td{padding:1px 8px 1px 0}#hud td.v{font-variant-numeric:tabular-nums;font-weight:600;text-align:right}
#ev{top:12px;right:12px;width:min(340px,calc(100vw - 24px));max-height:38vh;overflow:auto}
#ev div{padding:2px 0;border-bottom:1px dashed rgba(255,255,255,.08);color:var(--muted)}#ev div.crit{color:var(--bad)}#ev div.now{color:var(--fg)}
#bar{left:12px;right:12px;bottom:12px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
#bar input[type=range]{flex:1 1 200px}
button,select{background:#2a3240;color:var(--fg);border:1px solid #3a4454;border-radius:6px;padding:4px 10px;font:inherit;cursor:pointer}
label{color:var(--muted);user-select:none}
#prof{width:100%;height:60px;display:block;margin-top:6px}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;font-weight:700}
.pass{background:var(--ok);color:#000}.fail{background:var(--bad);color:#000}
#err{position:fixed;inset:0;display:none;align-items:center;justify-content:center;text-align:center;padding:24px}
@media (max-width:640px){#ev{top:auto;bottom:118px;max-height:22vh}#hud{font-size:12px}}
</style>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head><body>
<canvas id="c"></canvas>
<div id="hud" class="panel"><h1 id="title"></h1><table id="tbl"></table><canvas id="prof"></canvas></div>
<div id="ev" class="panel"></div>
<div id="bar" class="panel">
  <button id="play">&#9654; Odtwórz</button>
  <input id="slider" type="range" min="0" max="1000" value="0">
  <span id="time">0.0 s</span>
  <select id="speed"><option>0.5</option><option selected>1</option><option>2</option><option>5</option><option>10</option><option>20</option></select>
  <label><input id="follow" type="checkbox" checked> śledź</label>
  <label><input id="big" type="checkbox" checked> powiększ model</label>
  <label><input id="estp" type="checkbox"> estymata</label>
  <button id="ov">cały lot</button>
</div>
<div id="err" class="panel">Nie udało się wczytać biblioteki three.js (wymagany dostęp do internetu - cdn.jsdelivr.net).</div>
<script>
const D = __DATA__;
const F = D.frames, N = F.t.length, T_END = F.t[N-1];
document.getElementById('title').innerHTML = D.name + ' <span class="badge ' + (D.passed?'pass':'fail') + '">' + (D.passed?'PASS':'FAIL') + '</span>' + (D.vehicle ? '<div style="font-weight:400;color:#9aa4b1;font-size:12px">' + D.vehicle + '</div>' : '');
if (typeof THREE === 'undefined') { document.getElementById('err').style.display='flex'; throw new Error('three.js'); }

// ---------- scena
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x9cc3e6);
const camera = new THREE.PerspectiveCamera(55, 1, 0.5, 20000);
const controls = new THREE.OrbitControls(camera, canvas);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x445533, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 0.7); sun.position.set(300, 600, 200); scene.add(sun);

// NED -> three: x = E, y = up, z = -N
const P = (n, e, h) => new THREE.Vector3(e, h, -n);
let minN=Infinity,maxN=-Infinity,minE=Infinity,maxE=-Infinity,maxH=0;
for (let i=0;i<N;i++){minN=Math.min(minN,F.n[i]);maxN=Math.max(maxN,F.n[i]);minE=Math.min(minE,F.e[i]);maxE=Math.max(maxE,F.e[i]);maxH=Math.max(maxH,F.h[i]);}
D.waypoints.forEach(w=>{minN=Math.min(minN,w[0]);maxN=Math.max(maxN,w[0]);minE=Math.min(minE,w[1]);maxE=Math.max(maxE,w[1]);});
const span = Math.max(maxN-minN, maxE-minE, 60), cN=(minN+maxN)/2, cE=(minE+maxE)/2;
const size = span*1.6;
const ground = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshLambertMaterial({color:0x6f8f5a}));
ground.rotation.x = -Math.PI/2; ground.position.set(cE, 0, -cN); scene.add(ground);
const grid = new THREE.GridHelper(size, Math.round(size/ (span>400?50:10)), 0x3d5233, 0x58704a);
grid.position.set(cE, 0.05, -cN); scene.add(grid);
scene.fog = new THREE.Fog(0x9cc3e6, size*0.8, size*3);
// znacznik północy
const nArrow = new THREE.ArrowHelper(new THREE.Vector3(0,0,-1), P(minN - span*0.15, minE - span*0.15, 0.5), span*0.12, 0xffffff);
scene.add(nArrow);

// trajektoria
const pts = []; for (let i=0;i<N;i++) pts.push(P(F.n[i], F.e[i], F.h[i]));
const pathAll = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({color:0xffffff, transparent:true, opacity:0.35}));
scene.add(pathAll);
const donePos = new Float32Array(N*3); pts.forEach((p,i)=>{donePos[3*i]=p.x;donePos[3*i+1]=p.y;donePos[3*i+2]=p.z;});
const doneGeo = new THREE.BufferGeometry(); doneGeo.setAttribute('position', new THREE.BufferAttribute(donePos,3));
const pathDone = new THREE.Line(doneGeo, new THREE.LineBasicMaterial({color:0xffd23f})); scene.add(pathDone);
const shadowPts = pts.map(p=>new THREE.Vector3(p.x, 0.1, p.z));
scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(shadowPts), new THREE.LineBasicMaterial({color:0x223322, transparent:true, opacity:0.4})));
const estPts = []; for (let i=0;i<N;i++) estPts.push(P(F.en[i], F.ee[i], F.h[i]));
const estLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(estPts), new THREE.LineDashedMaterial({color:0xff66cc, dashSize:3, gapSize:2}));
estLine.computeLineDistances(); estLine.visible=false; scene.add(estLine);

// punkty trasy
D.waypoints.forEach((w,i)=>{
  const m = new THREE.Mesh(new THREE.SphereGeometry(Math.max(1, span/150), 12, 8), new THREE.MeshLambertMaterial({color:0xff8c32}));
  m.position.copy(P(w[0], w[1], w[2])); scene.add(m);
  const pole = new THREE.Line(new THREE.BufferGeometry().setFromPoints([P(w[0],w[1],0), P(w[0],w[1],w[2])]), new THREE.LineBasicMaterial({color:0xff8c32, transparent:true, opacity:0.5}));
  scene.add(pole);
});
// mikrobursty i termika
const mbMeshes = D.microbursts.map(mb=>{
  const g = new THREE.CylinderGeometry(mb.r, mb.r*1.3, 400, 40, 1, true);
  const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({color:0x8a2be2, transparent:true, opacity:0.15, side:THREE.DoubleSide, depthWrite:false}));
  scene.add(m); return [mb, m];
});
D.thermals.forEach(th=>{
  const m = new THREE.Mesh(new THREE.CylinderGeometry(th.r, th.r*0.8, 500, 32, 1, true), new THREE.MeshBasicMaterial({color:0xff9933, transparent:true, opacity:0.12, side:THREE.DoubleSide, depthWrite:false}));
  m.position.copy(P(th.n, th.e, 250)); scene.add(m);
});
if (D.icing && D.icing.lwc > 0) {
  const top = D.icing.cloud_top||0, base = D.icing.cloud_base||0;
  const cl = new THREE.Mesh(new THREE.BoxGeometry(size, Math.max(top-base,1), size), new THREE.MeshBasicMaterial({color:0xdde6ee, transparent:true, opacity:0.12, depthWrite:false}));
  cl.position.set(cE, (top+base)/2, -cN); scene.add(cl);
}

// ---------- model pojazdu (w układzie ciała FRD: x - nos, y - prawo, z - dół)
const veh = new THREE.Group();
const mat = (c)=>new THREE.MeshLambertMaterial({color:c});
function box(sx, sy, sz, x, y, z, c){ const m=new THREE.Mesh(new THREE.BoxGeometry(sx,sy,sz), mat(c)); m.position.set(x,y,z); veh.add(m); return m; }
if (D.mesh) {                                         // model z CAD
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(D.mesh.v, 3));
  g.setIndex(D.mesh.f);
  g.computeVertexNormals();
  veh.add(new THREE.Mesh(g, new THREE.MeshLambertMaterial({color:0xeef1f5, side:THREE.DoubleSide})));
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(g, 35), new THREE.LineBasicMaterial({color:0x5a6472}));
  veh.add(edges);
} else if (D.kind === 'fixed_wing') {
  box(1.2, 0.14, 0.14, 0, 0, 0, 0xf2f2f2);           // kadłub
  box(0.22, 2.0, 0.03, 0.05, 0, -0.03, 0xe8e8e8);     // skrzydło
  box(0.1, 0.4, 0.02, 0.05, 0.85, -0.03, 0xff3b30);   // końcówka prawa (czerwona)
  box(0.12, 0.6, 0.02, -0.62, 0, 0, 0xe8e8e8);        // statecznik poziomy
  box(0.14, 0.02, 0.25, -0.62, 0, -0.13, 0x1f6fb2);   // statecznik pionowy
  box(0.04, 0.3, 0.02, 0.62, 0, 0, 0x333333);         // śmigło
} else {
  box(0.22, 0.16, 0.08, 0, 0, 0, 0x2b2f36);
  box(0.08, 0.06, 0.04, 0.12, 0, 0, 0xff3b30);        // nos
  const L = 0.25;
  [[1,1],[-1,1],[-1,-1],[1,-1]].forEach(([sx,sy],i)=>{
    const x=sx*L/Math.SQRT2, y=sy*L/Math.SQRT2;
    const arm = new THREE.Mesh(new THREE.BoxGeometry(Math.hypot(x,y), 0.025, 0.025), mat(0x444a55));
    arm.position.set(x/2, y/2, 0); arm.rotation.z = Math.atan2(y, x); veh.add(arm);
    const disc = new THREE.Mesh(new THREE.CylinderGeometry(0.127, 0.127, 0.01, 24), new THREE.MeshLambertMaterial({color: sx>0?0xff6b5e:0x9ad1ff, transparent:true, opacity:0.6}));
    disc.rotation.x = Math.PI/2; disc.position.set(x, y, -0.05); veh.add(disc);
  });
}
const holder = new THREE.Group(); holder.add(veh); scene.add(holder);
const windArrow = new THREE.ArrowHelper(new THREE.Vector3(1,0,0), new THREE.Vector3(), 5, 0x2ad1c9, 1.5, 1.0);
scene.add(windArrow);

// kamera startowa: za pojazdem (tryb śledzenia); przycisk "cały lot" pokazuje całą trasę
const SC0 = Math.max(1, span/120);
const RAD = new THREE.Box3().setFromObject(veh).getBoundingSphere(new THREE.Sphere()).radius || 1;
function chaseView(){
  const yaw = Math.atan2(F.e[Math.min(5,N-1)]-F.e[0], F.n[Math.min(5,N-1)]-F.n[0]) || 0;
  const d = 9*RAD*SC0 + 3;
  camera.position.copy(P(F.n[0] - d*Math.cos(yaw), F.e[0] - d*Math.sin(yaw), F.h[0] + 0.45*d + 2));
  controls.target.copy(P(F.n[0], F.e[0], F.h[0]));
}
function overview(){
  document.getElementById('follow').checked = false;
  controls.target.set(cE, maxH*0.3, -cN);
  camera.position.set(cE - span*0.55, Math.max(maxH, 30) + span*0.55, -cN + span*0.75);
}
chaseView();

// ---------- profil wysokości
const prof = document.getElementById('prof'), pctx = prof.getContext('2d');
function drawProfile(i){
  const w = prof.width = prof.clientWidth*devicePixelRatio, h = prof.height = 60*devicePixelRatio;
  pctx.clearRect(0,0,w,h);
  const hm = Math.max(maxH, 1);
  pctx.strokeStyle='#5fb0ff'; pctx.lineWidth=1.5*devicePixelRatio; pctx.beginPath();
  for (let k=0;k<N;k++){ const x=F.t[k]/T_END*w, y=h-(F.h[k]/hm)*(h-6)-3; k?pctx.lineTo(x,y):pctx.moveTo(x,y); }
  pctx.stroke();
  D.events.forEach(([t,m])=>{ if(/AWARIA|KATASTROFA|FAILSAFE|PRZECI|Pitota/.test(m)){ pctx.fillStyle='#ff6b5e'; pctx.fillRect(t/T_END*w-1,0,2*devicePixelRatio,h);} });
  pctx.fillStyle='#ffd23f'; pctx.fillRect(F.t[i]/T_END*w-1, 0, 2*devicePixelRatio, h);
}

// ---------- HUD i zdarzenia
const evBox = document.getElementById('ev');
evBox.innerHTML = D.events.map(([t,m],k)=>`<div id="e${k}" class="${/AWARIA|KATASTROFA|FAILSAFE|PRZECI|Pitota|WYCZERP/.test(m)?'crit':''}">${t.toFixed(1)} s &nbsp; ${m}</div>`).join('') + `<div>koniec: ${D.end}</div>`;
const tbl = document.getElementById('tbl');
function row(k,v){return `<tr><td>${k}</td><td class="v">${v}</td></tr>`;}
function hud(i){
  const ws = Math.hypot(F.wn[i], F.we[i]);
  const wdir = ((Math.atan2(-F.we[i], -F.wn[i])*180/Math.PI)+360)%360;
  tbl.innerHTML = row('faza', F.phase[i]) + row('wysokość AGL', F.alt[i].toFixed(1)+' m') +
    row(D.kind==='fixed_wing'?'IAS (Pitot)':'prędkość powietrzna', (D.kind==='fixed_wing'?F.ias[i]:F.tas[i]).toFixed(1)+' m/s') +
    row('prędkość nad ziemią', F.gs[i].toFixed(1)+' m/s') +
    row('wiatr', ws.toFixed(1)+' m/s z '+wdir.toFixed(0)+'°' + (Math.abs(F.wd[i])>0.5?(' / '+(-F.wd[i]).toFixed(1)+' pion'):'')) +
    row('przechylenie / pochylenie', F.roll[i].toFixed(0)+'° / '+F.pitch[i].toFixed(0)+'°') +
    row('bateria', (F.soc[i]*100).toFixed(0)+'% · '+F.v[i].toFixed(1)+' V') +
    row('przepustnica', (F.thr[i]*100).toFixed(0)+'%') +
    row('temperatura', F.temp[i].toFixed(1)+' °C') +
    (F.ice[i]>0.01?row('oblodzenie', (F.ice[i]*100).toFixed(0)+'%'):'') +
    (F.gps[i]<0.5?row('<span style="color:#ff6b5e">GPS</span>','<span style="color:#ff6b5e">BRAK</span>'):'');
  D.events.forEach(([t],k)=>{ const el=document.getElementById('e'+k); if(el) el.classList.toggle('now', t<=F.t[i]); });
}

// ---------- odtwarzanie
let simT = 0, playing = false, last = performance.now(), idx = 0;
const slider = document.getElementById('slider'), playBtn = document.getElementById('play');
const speedSel = document.getElementById('speed'), follow = document.getElementById('follow'), big = document.getElementById('big');
document.getElementById('estp').onchange = e => estLine.visible = e.target.checked;
document.getElementById('ov').onclick = overview;
playBtn.onclick = () => { playing = !playing; if (simT >= T_END) simT = 0; playBtn.innerHTML = playing ? '&#10073;&#10073; Pauza' : '&#9654; Odtwórz'; };
slider.oninput = () => { simT = slider.value/1000*T_END; };
function findIdx(t){ let lo=0, hi=N-1; while(lo<hi){ const m=(lo+hi+1)>>1; if(F.t[m]<=t) lo=m; else hi=m-1; } return lo; }
function update(){
  idx = findIdx(simT);
  const i = idx, j = Math.min(i+1, N-1);
  const a = j>i ? (simT - F.t[i])/(F.t[j]-F.t[i]) : 0;
  const pos = P(F.n[i]+(F.n[j]-F.n[i])*a, F.e[i]+(F.e[j]-F.e[i])*a, F.h[i]+(F.h[j]-F.h[i])*a);
  holder.position.copy(pos);
  const q1 = new THREE.Quaternion(...F.q[i]), q2 = new THREE.Quaternion(...F.q[j]);
  holder.quaternion.copy(q1.slerp(q2, a));
  const sc = big.checked ? Math.max(1, span/120) : 1; veh.scale.setScalar(sc);
  const wv = new THREE.Vector3(F.we[i], -F.wd[i], -F.wn[i]); const wl = wv.length();
  windArrow.visible = wl > 0.3;
  if (wl > 0.3) { windArrow.position.copy(pos).add(new THREE.Vector3(0, 2.5*sc, 0)); windArrow.setDirection(wv.clone().normalize()); windArrow.setLength(Math.max(2, wl*sc*0.9), sc*0.6, sc*0.4); }
  doneGeo.setDrawRange(0, i+1);
  mbMeshes.forEach(([mb, m])=>{ const dt=Math.max(0, simT-mb.t0); m.visible = simT>=mb.t0; m.position.copy(P(mb.n+mb.dn*dt, mb.e+mb.de*dt, 200)); });
  if (follow.checked) { const d = pos.clone().sub(controls.target); camera.position.add(d); controls.target.copy(pos); }
  document.getElementById('time').textContent = simT.toFixed(1) + ' / ' + T_END.toFixed(0) + ' s';
  slider.value = Math.round(simT/T_END*1000);
  if (i !== lastIdx) { hud(i); drawProfile(i); lastIdx = i; }
}
function resize(){ const w=innerWidth, h=innerHeight; renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix(); }
let lastIdx = -1;
addEventListener('resize', () => { resize(); lastIdx = -1; }); resize();
function loop(now){
  const dt = Math.min(0.1, (now-last)/1000); last = now;
  if (playing) { simT += dt*parseFloat(speedSel.value); if (simT >= T_END) { simT = T_END; playing=false; playBtn.innerHTML='&#9654; Odtwórz'; } }
  update(); controls.update(); renderer.render(scene, camera);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
</script></body></html>
"""
