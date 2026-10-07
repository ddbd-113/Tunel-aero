'use strict';
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? '–' : Number(v).toFixed(d);

// ============================================================ worker (Python)
let worker = null, reqId = 0, ready = false;
const pending = new Map();
let progressCb = null;

function setRT(state, text) {
  const el = $('#rt');
  el.className = 'rt ' + state;
  $('#rt-text').textContent = text;
}

function startWorker() {
  ready = false;
  worker = new Worker('worker.js?v=__VERSION__');
  worker.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === 'status') setRT('busy', m.text);
    else if (m.type === 'ready') { ready = true; setRT('ready', 'Python gotowy (Pyodide)'); onReady(); }
    else if (m.type === 'fatal') setRT('error', 'Błąd uruchamiania: ' + m.error);
    else if (m.type === 'progress') { if (progressCb) progressCb(m.value); }
    else if (m.type === 'result') {
      const p = pending.get(m.id);
      if (p) { pending.delete(m.id); m.ok ? p.resolve(m.result) : p.reject(new Error(m.error)); }
    }
  };
}

function call(cmd, args = {}) {
  return new Promise((resolve, reject) => {
    const id = ++reqId;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, cmd, args });
  });
}

// ============================================================ stan
const DEFAULT_CFG = () => ({
  name: 'Mój samolot', units: 'mm', axes: { forward: '-x', up: '+z' },
  parts: {}, mass: { components: [] },
  surfaces: { wing: { airfoil: { cl_max: 1.3, zero_lift_alpha_deg: -2.1, cm0: -0.05 } } },
  propulsion: { kv: 1000, diameter_in: 10, pitch_in: 5 },
  battery: { cells: 3, capacity_ah: 2.2 },
});
const S = { cfg: DEFAULT_CFG(), files: {}, analysis: null, vmode: 'cad', catalog: null,
            scenario: null, info: null, fails: [], result: null };

function getPath(o, path) {
  return path.split('.').reduce((a, k) => (a == null ? undefined : a[k]), o);
}
function setPath(o, path, val) {
  const ks = path.split('.');
  let cur = o;
  ks.slice(0, -1).forEach((k, i) => {
    if (cur[k] == null || typeof cur[k] !== 'object') cur[k] = /^\d+$/.test(ks[i + 1]) ? [] : {};
    cur = cur[k];
  });
  const last = ks[ks.length - 1];
  if (val === undefined) { if (Array.isArray(cur)) cur[last] = null; else delete cur[last]; }
  else cur[last] = val;
}

// ============================================================ krok 1: pojazd
const ROLE_NAMES = { wing: 'skrzydło', htail: 'usterzenie poziome', vtail: 'usterzenie pionowe', fuselage: 'kadłub', other: 'inne (tylko masa/wygląd)' };

function guessRole(name) {
  const n = name.toLowerCase();
  if (/pion|vtail|v_tail|vstab|rudder|kierun|fin\b|fin[._-]|statecznik(?!.*poziom)/.test(n) && !/poziom|wysok/.test(n)) return 'vtail';
  if (/poziom|wysok|htail|h_tail|hstab|elevator|stabiliz/.test(n)) return 'htail';
  if (/skrzyd|wing|plat|płat/.test(n)) return 'wing';
  if (/kadl|kadł|fuse|body|korpus/.test(n)) return 'fuselage';
  return 'other';
}

function partName(file) {
  let base = file.replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_') || 'czesc';
  let n = base, i = 2;
  while (S.cfg.parts[n] && S.cfg.parts[n].file !== file) n = base + '_' + i++;
  return n;
}

async function addFiles(fileList) {
  for (const f of fileList) {
    const ext = f.name.split('.').pop().toLowerCase();
    if (ext === 'yaml' || ext === 'yml') {
      try {
        const cfg = await call('yaml_load', { text: await f.text() });
        S.cfg = Object.assign(DEFAULT_CFG(), cfg);
        S.cfg.mass = S.cfg.mass || { components: [] };
        S.cfg.mass.components = S.cfg.mass.components || [];
        for (const [k, p] of Object.entries(S.cfg.parts || {})) if (typeof p === 'string') S.cfg.parts[k] = { file: p };
      } catch (e) { alert('Nie udało się wczytać YAML: ' + e.message); }
    } else if (ext === 'stl' || ext === 'obj') {
      const bytes = new Uint8Array(await f.arrayBuffer());
      S.files[f.name] = bytes;
      let objects = [];
      if (ext === 'obj') {   // jeden OBJ z wieloma obiektami (Blender, FreeCAD) -> osobne części
        const txt = new TextDecoder().decode(bytes);
        objects = [...new Set([...txt.matchAll(/^[og]\s+(.+?)\s*$/gm)].map(m => m[1]))];
      }
      if (objects.length > 1) {
        for (const o of objects) {
          const file = f.name + '#' + o;
          if (!Object.values(S.cfg.parts).some(p => p.file === file)) S.cfg.parts[partName(o)] = { file, role: guessRole(o) };
        }
      } else if (!Object.values(S.cfg.parts).some(p => p.file === f.name)) {
        S.cfg.parts[partName(f.name)] = { file: f.name, role: guessRole(f.name) };
      }
    }
  }
  S.analysis = null;
  renderCad();
}

function renderParts() {
  const tb = $('#parts-tbl tbody');
  const rows = Object.entries(S.cfg.parts);
  if (!rows.length) { tb.innerHTML = '<tr><td colspan="5" class="muted">Brak plików - dodaj części samolotu.</td></tr>'; return; }
  tb.innerHTML = rows.map(([name, p]) => {
    const missing = !S.files[(p.file || '').split('#')[0]];
    return `<tr data-part="${esc(name)}">
      <td>${esc(p.file)}${missing ? ' <span class="err small">(brak pliku - dodaj go)</span>' : ''}</td>
      <td><select data-k="role">${Object.entries(ROLE_NAMES).map(([k, v]) => `<option value="${k}" ${p.role === k ? 'selected' : ''}>${v}</option>`).join('')}</select></td>
      <td><input data-k="mass" type="number" step="0.005" min="0" value="${p.mass ?? ''}" placeholder="auto"></td>
      <td><input data-k="shell" type="checkbox" ${p.shell ? 'checked' : ''} title="masa rozłożona na powierzchni (pianka, balsa, kompozyt)"></td>
      <td><button class="ghost small" data-k="del">usuń</button></td></tr>`;
  }).join('');
}

function renderComps() {
  const tb = $('#comp-tbl tbody');
  const comps = S.cfg.mass.components || [];
  tb.innerHTML = comps.map((c, i) => `<tr data-i="${i}">
    <td><input data-k="name" value="${esc(c.name || '')}"></td>
    <td><input data-k="mass" type="number" step="0.005" value="${c.mass ?? ''}"></td>
    ${[0, 1, 2].map(j => `<td><input data-k="p${j}" type="number" value="${(c.position || [])[j] ?? 0}"></td>`).join('')}
    <td><button class="ghost small" data-k="del">×</button></td></tr>`).join('') ||
    '<tr><td colspan="6" class="muted">Np. silnik, akumulator, serwa, odbiornik (pozycje w układzie CAD).</td></tr>';
}

function renderForm() {
  $$('[data-path]').forEach(el => {
    const v = getPath(S.cfg, el.dataset.path);
    el.value = v === undefined || v === null ? '' : v;
  });
}

function renderCad() {
  renderParts();
  renderComps();
  renderForm();
  const hasWing = Object.values(S.cfg.parts).some(p => p.role === 'wing') || getPath(S.cfg, 'surfaces.wing.sections');
  $('#analyze').disabled = !ready || !hasWing;
  $('#an-status').textContent = hasWing ? '' : 'Dodaj co najmniej skrzydło (rola: skrzydło).';
  if (!S.analysis) $('#analysis').classList.add('hidden');
}

function bindForm() {
  $$('[data-path]').forEach(el => el.addEventListener('change', () => {
    const path = el.dataset.path;
    let v = el.value;
    if (el.type === 'number') v = v === '' ? undefined : Number(v);
    if (path.startsWith('mass.cg.')) {
      const cg = S.cfg.mass.cg || [null, 0, null];
      cg[Number(path.split('.').pop())] = v === undefined ? null : v;
      if (cg[0] == null && cg[2] == null) delete S.cfg.mass.cg;
      else S.cfg.mass.cg = [cg[0] ?? 0, cg[1] ?? 0, cg[2] ?? 0];
      return;
    }
    setPath(S.cfg, path, v === '' ? undefined : v);
  }));
  $('#parts-tbl').addEventListener('change', (e) => {
    const tr = e.target.closest('tr[data-part]');
    if (!tr) return;
    const p = S.cfg.parts[tr.dataset.part];
    const k = e.target.dataset.k;
    if (k === 'role') p.role = e.target.value;
    if (k === 'mass') { if (e.target.value === '') delete p.mass; else p.mass = Number(e.target.value); }
    if (k === 'shell') p.shell = e.target.checked;
    renderCad();
  });
  $('#parts-tbl').addEventListener('click', (e) => {
    if (e.target.dataset.k !== 'del') return;
    const name = e.target.closest('tr').dataset.part;
    delete S.files[S.cfg.parts[name].file];
    delete S.cfg.parts[name];
    renderCad();
  });
  $('#comp-tbl').addEventListener('change', (e) => {
    const tr = e.target.closest('tr[data-i]');
    if (!tr) return;
    const c = S.cfg.mass.components[Number(tr.dataset.i)];
    const k = e.target.dataset.k;
    if (k === 'name') c.name = e.target.value;
    else if (k === 'mass') c.mass = Number(e.target.value);
    else if (k && k[0] === 'p') { c.position = c.position || [0, 0, 0]; c.position[Number(k[1])] = Number(e.target.value); }
  });
  $('#comp-tbl').addEventListener('click', (e) => {
    if (e.target.dataset.k !== 'del') return;
    S.cfg.mass.components.splice(Number(e.target.closest('tr').dataset.i), 1);
    renderComps();
  });
  $('#add-comp').onclick = () => { S.cfg.mass.components.push({ name: 'element', mass: 0.05, position: [0, 0, 0] }); renderComps(); };
}

function cleanCfg() {
  const c = JSON.parse(JSON.stringify(S.cfg));
  c.type = 'cad_aircraft';
  const prune = (o) => {
    if (Array.isArray(o)) return o.map(prune);
    if (o && typeof o === 'object') {
      for (const k of Object.keys(o)) {
        const v = o[k];
        if (v === null || v === '' || (typeof v === 'number' && Number.isNaN(v))) delete o[k];
        else o[k] = prune(v);
      }
    }
    return o;
  };
  return prune(c);
}

async function analyze() {
  $('#analyze').disabled = true;
  $('#an-status').textContent = 'Liczenie: przekroje z siatki, metoda siatki wirowej (VLM), masa i wyważenie…';
  try {
    const t0 = performance.now();
    S.analysis = await call('analyze', { cfg: JSON.stringify(cleanCfg()), files: S.files });
    $('#an-status').textContent = `Gotowe (${((performance.now() - t0) / 1000).toFixed(1)} s).`;
    renderAnalysis();
  } catch (e) {
    $('#an-status').innerHTML = '<span class="err">' + esc(e.message) + '</span>';
  } finally {
    $('#analyze').disabled = !ready;
  }
}

function renderAnalysis() {
  const a = S.analysis;
  $('#analysis').classList.remove('hidden');
  const lbl = { error: 'błąd', warn: 'uwaga', info: 'info', ok: 'ok' };
  $('#checks').innerHTML = a.checks.map(([l, m]) => `<div class="chk ${l}"><b>${lbl[l]}</b><span>${esc(m)}</span></div>`).join('');
  const s = a.summary, u = a.units;
  const kv = [
    [fmt(s.mass, 3) + ' kg', 'masa'], [fmt(s.span_mm, 0) + ' mm', 'rozpiętość'], [fmt(s.area_dm2, 1) + ' dm²', 'powierzchnia skrzydła'],
    [fmt(s.mac_mm, 0) + ' mm', 'średnia cięciwa'], [fmt(s.wing_loading, 0) + ' g/dm²', 'obciążenie powierzchni'],
    [fmt(s.static_margin, 1) + '%', 'zapas stateczności'], [fmt(s.cg_pct_mac, 0) + '% MAC', 'środek ciężkości'],
    [fmt(s.np_pct_mac, 0) + '% MAC', 'punkt neutralny'],
    [`x = ${fmt(a.cg_cad[0], 0)} ${u}`, 'CG w układzie CAD'],
    [`${fmt(Math.min(a.cg_range_cad[0][0], a.cg_range_cad[1][0]), 0)} … ${fmt(Math.max(a.cg_range_cad[0][0], a.cg_range_cad[1][0]), 0)} ${u}`, 'zalecany zakres CG (x)'],
    [fmt(s.v_stall, 1) + ' m/s', 'przeciągnięcie'], [fmt(s.cruise, 1) + ' m/s', 'przelot'],
    [fmt(s.trim_throttle, 0) + '% · ' + fmt(s.trim_power, 0) + ' W', 'moc w przelocie'],
    [fmt(s.trim_elevator, 1) + '°', 'ster wys. w trymie'], [fmt(s.tw, 2), 'ciąg statyczny / ciężar'],
    [fmt(s.CD0, 4), 'CD0'], [fmt(s.CLmax, 2), 'CL max'],
  ];
  $('#summary').innerHTML = kv.map(([v, l]) => `<div class="kv"><b>${esc(v)}</b><span>${esc(l)}</span></div>`).join('');
  const d = a.derivatives;
  const rows = [['CLa', 'CL_α'], ['Cma', 'Cm_α (stateczność podłużna, < 0)'], ['Cmq', 'Cm_q (tłumienie)'], ['Cnb', 'Cn_β (kierunkowa, > 0)'],
    ['Clb', 'Cl_β (wznios, < 0)'], ['Clp', 'Cl_p'], ['Cnr', 'Cn_r'], ['Clda', 'Cl_δa (lotki)'], ['Cmde', 'Cm_δe (ster wys.)'], ['Cndr', 'Cn_δr (ster kier.)']];
  $('#derivs').innerHTML = `<table class="tbl">${rows.map(([k, l]) => `<tr><td>${l}</td><td class="num">${fmt(d[k], 4)}</td></tr>`).join('')}</table>
    <table class="tbl"><tr><th>opór</th><th>CD</th></tr>${Object.entries(a.cd_parts).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${fmt(v, 4)}</td></tr>`).join('')}</table>`;
  drawPreview(a);
}

// ------------------------------------------------------------ podgląd 3D
let pv = null;
function drawPreview(a) {
  const box = $('#preview');
  if (typeof THREE === 'undefined') { box.innerHTML = '<p class="muted" style="padding:12px">Brak three.js (wymagany internet).</p>'; return; }
  if (!pv) {
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    box.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f1520);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.0));
    const dl = new THREE.DirectionalLight(0xffffff, 0.6); dl.position.set(2, 3, 1); scene.add(dl);
    const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 200);
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    const group = new THREE.Group(); scene.add(group);
    pv = { renderer, scene, camera, controls, group };
    const loop = () => { controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); };
    const resize = () => { const w = box.clientWidth, h = box.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); };
    new ResizeObserver(resize).observe(box); resize(); loop();
  }
  const { group, camera, controls } = pv;
  while (group.children.length) group.remove(group.children[0]);
  // ciało FRD -> three: x = y_ciała (prawo), y = -z (góra), z = -x (nos w stronę -z)
  const T = (x, y, z) => [y, -z, -x];
  if (a.mesh) {
    const v = a.mesh.v, pos = new Float32Array(v.length);
    for (let i = 0; i < v.length; i += 3) { const p = T(v[i], v[i + 1], v[i + 2]); pos[i] = p[0]; pos[i + 1] = p[1]; pos[i + 2] = p[2]; }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setIndex(a.mesh.f); g.computeVertexNormals();
    group.add(new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color: 0xd9dee5, side: THREE.DoubleSide, transparent: true, opacity: 0.85 })));
    group.add(new THREE.LineSegments(new THREE.EdgesGeometry(g, 30), new THREE.LineBasicMaterial({ color: 0x4a5462 })));
  }
  const R = new THREE.Box3().setFromObject(group).getBoundingSphere(new THREE.Sphere()).radius || 1;
  const sph = (r, c) => new THREE.Mesh(new THREE.SphereGeometry(r, 20, 14), new THREE.MeshBasicMaterial({ color: c, depthTest: false }));
  const cg = sph(R * 0.045, 0xffffff); cg.renderOrder = 2; group.add(cg);
  const np = sph(R * 0.04, 0xff4d4d); np.position.set(...T(a.np_body, 0, 0)); np.renderOrder = 2; group.add(np);
  const lo = a.np_body + 0.05 * a.mac, hi = a.np_body + 0.15 * a.mac;
  const band = new THREE.Mesh(new THREE.BoxGeometry(R * 0.5, R * 0.004, hi - lo),
    new THREE.MeshBasicMaterial({ color: 0x2ecc71, transparent: true, opacity: 0.55, depthTest: false }));
  band.position.set(...T((lo + hi) / 2, 0, 0)); band.renderOrder = 1; group.add(band);
  const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, -1), new THREE.Vector3(0, R * 0.25, 0), R * 0.7, 0x5aa9f0, R * 0.12, R * 0.06);
  group.add(arrow);
  const cv = document.createElement('canvas'); cv.width = 128; cv.height = 48;
  const cx = cv.getContext('2d'); cx.fillStyle = '#5aa9f0'; cx.font = 'bold 34px system-ui'; cx.fillText('NOS', 18, 36);
  const label = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), depthTest: false }));
  label.scale.set(R * 0.3, R * 0.11, 1); label.position.set(0, R * 0.35, -R * 0.8); group.add(label);
  camera.position.set(R * 1.3, R * 0.9, R * 1.5); controls.target.set(0, 0, 0); camera.near = R / 100; camera.far = R * 100; camera.updateProjectionMatrix();
}

// ============================================================ krok 2: warunki
const FAIL_DEFAULTS = {
  gps_loss: {}, link_loss: {}, pitot_blocked: {},
  gps_spoofing: { drift_north: 0.5, drift_east: 0 },
  mag_interference: { heading_error_deg: 25 },
  control_surface: { surface: 'aileron', position_deg: 3, detection_delay: 1.5 },
  motor: { motor: 1, efficiency: 0, detection_delay: 0.5 },
  battery_cell: { capacity_factor: 0.7, resistance_factor: 2 },
};
const FAIL_NAMES = { gps_loss: 'utrata GPS', gps_spoofing: 'spoofing GPS', link_loss: 'utrata łącza', mag_interference: 'zakłócenie magnetometru',
  pitot_blocked: 'zatkana rurka Pitota', control_surface: 'zablokowany ster', motor: 'awaria silnika', battery_cell: 'uszkodzone ogniwo',
  gps_degraded: 'pogorszony GPS', propeller_damage: 'uszkodzone śmigło', imu_vibration: 'drgania IMU' };

function vehicleKind() {
  if (S.vmode === 'cad') return 'fixed_wing';
  const v = S.catalog.vehicles.find(v => v.path === $('#preset').value);
  return v ? v.kind : 'multirotor';
}

function renderScenarioList() {
  const kind = vehicleKind();
  const list = S.catalog.scenarios.filter(s => s.kind === kind);
  const sel = $('#scenario');
  const prev = sel.value;
  sel.innerHTML = list.map(s => `<option value="${esc(s.path)}">${esc(s.name)}${s.expect_pass ? '' : ' (demonstracja porażki)'}</option>`).join('');
  if (list.some(s => s.path === prev)) sel.value = prev;
  loadScenarioInfo();
}

async function loadScenarioInfo() {
  const path = $('#scenario').value;
  if (!path) return;
  const i = await call('scenario_info', { path });
  S.info = i;
  $('#sc-desc').textContent = i.description;
  $('#c-wind').value = i.wind_speed; $('#c-dir').value = i.wind_direction; $('#c-terrain').value = i.terrain;
  $('#c-turb').value = typeof i.turbulence === 'number' ? 'moderate' : i.turbulence;
  $('#c-temp').value = i.temperature_c ?? 15; $('#c-elev').value = i.ground_elevation; $('#c-hum').value = Math.round((i.humidity ?? 0.5) * 100);
  $('#c-rain').value = i.rain_mm_h; $('#c-lwc').value = i.icing_lwc; $('#c-cb').value = i.icing_base; $('#c-ct').value = i.icing_top;
  $('#c-dur').value = i.duration; $('#c-seed').value = i.seed; $('#c-perfect').checked = i.navigation === 'perfect';
  $('#c-gusts').value = 0;
  S.fails = JSON.parse(JSON.stringify(i.failures || []));
  renderFails();
}

function renderFails() {
  $('#fails').innerHTML = S.fails.length ? S.fails.map((f, i) => {
    const p = Object.entries(f).filter(([k]) => !['type', 't', 'duration'].includes(k)).map(([k, v]) => `${k}=${v}`).join(', ');
    return `<div class="fail-row"><span><b>${esc(FAIL_NAMES[f.type] || f.type)}</b> od ${f.t} s${f.duration ? ` przez ${f.duration} s` : ''}${p ? ` · ${esc(p)}` : ''}</span>
      <button class="ghost small" data-i="${i}">usuń</button></div>`;
  }).join('') : '<p class="muted small">Brak awarii w tym teście.</p>';
}

function overrides() {
  const n = (id) => { const v = $(id).value; return v === '' ? undefined : Number(v); };
  const o = {
    duration: n('#c-dur'), seed: n('#c-seed'),
    'environment.wind.speed': n('#c-wind') ?? 0, 'environment.wind.direction': n('#c-dir') ?? 0,
    'environment.wind.terrain': $('#c-terrain').value, 'environment.wind.turbulence': $('#c-turb').value,
    'environment.temperature_c': n('#c-temp'), 'environment.ground_elevation': n('#c-elev') ?? 0,
    'environment.humidity': (n('#c-hum') ?? 50) / 100, 'environment.rain_mm_h': n('#c-rain') ?? 0,
    'environment.icing': { lwc: n('#c-lwc') ?? 0, cloud_base: n('#c-cb') ?? 0, cloud_top: n('#c-ct') ?? 0 },
    failures: S.fails, navigation: $('#c-perfect').checked ? 'perfect' : 'realistic',
  };
  const g = n('#c-gusts') ?? 0;
  if (g > 0) o['environment.wind.random_gusts'] = { rate_per_min: g, speed: [2, Math.max(2.5, n('#c-gustmax') ?? 6)], duration: [1, 4] };
  for (const k of Object.keys(o)) if (o[k] === undefined) delete o[k];
  return o;
}

async function runTest() {
  const btn = $('#run');
  btn.disabled = true;
  $('#prog').classList.remove('hidden');
  const t0 = performance.now();
  const dur = Number($('#c-dur').value) || 0;
  progressCb = (f) => {
    $('#prog-bar').style.width = (f * 100).toFixed(0) + '%';
    const el = (performance.now() - t0) / 1000;
    $('#prog-text').textContent = `${(f * 100).toFixed(0)}% · ${fmt(f * dur, 0)} s lotu w ${el.toFixed(0)} s`;
  };
  progressCb(0);
  try {
    let vehicle = '', files = null;
    if (S.vmode === 'cad') { vehicle = JSON.stringify(cleanCfg()); files = S.files; }
    else vehicle = JSON.stringify({ path: $('#preset').value });
    const r = await call('run', { scenario: $('#scenario').value, overrides: JSON.stringify(overrides()), vehicle, files });
    S.result = r;
    renderResult(r, (performance.now() - t0) / 1000);
    $('#tab3').disabled = false;
    showStep(3);
  } catch (e) {
    if (e.message !== 'przerwano') $('#prog-text').innerHTML = '<span class="err">' + esc(e.message) + '</span>';
  } finally {
    btn.disabled = !ready;
    progressCb = null;
    if (S.result) setTimeout(() => $('#prog').classList.add('hidden'), 400);
  }
}

function cancelRun() {
  worker.terminate();
  for (const [, p] of pending) p.reject(new Error('przerwano'));
  pending.clear();
  $('#prog-text').textContent = 'Przerwano - ponowne uruchamianie Pythona…';
  startWorker();
}

// ============================================================ krok 3: wyniki
function renderResult(r, wall) {
  const expect = r.expect_pass;
  $('#verdict').innerHTML = `<span class="badge ${r.passed ? 'pass' : 'fail'}">${r.passed ? 'PASS' : 'FAIL'}</span>
    <div><h2>${esc(r.name)}</h2><div class="muted">${esc(r.vehicle)} · ${esc(r.end_reason)} · obliczenia ${fmt(wall, 1)} s
    ${expect ? '' : ' · scenariusz demonstracyjny (oczekiwany FAIL)'}</div></div>`;
  $('#crit').innerHTML = '<tr><th>kryterium</th><th>wynik</th><th>próg</th><th></th></tr>' + r.criteria.map(c => {
    const v = typeof c.value === 'boolean' ? (c.value ? 'tak' : 'nie') : fmt(c.value, 2);
    const t = typeof c.threshold === 'boolean' ? (c.threshold ? 'tak' : '–') : c.threshold;
    return `<tr><td>${esc(c.label)}</td><td class="num">${v}</td><td class="num">${esc(t)}</td><td class="${c.passed ? 'ok' : 'bad'}">${c.passed ? 'OK' : 'NIE'}</td></tr>`;
  }).join('');
  const crit = /AWARIA|KATASTROFA|FAILSAFE|PRZECIĄGNIĘCIE|Pitota|WYCZERPANA/;
  $('#events').innerHTML = r.events.map(([t, m]) => `<li class="${crit.test(m) ? 'crit' : ''}">${fmt(t, 1)} s · ${esc(m)}</li>`).join('') || '<li>brak</li>';
  const m = r.metrics;
  const kv = [['flight_time', 'czas lotu', 's', 0], ['distance_km', 'dystans', 'km', 2], ['energy_wh', 'energia', 'Wh', 1],
    ['final_soc', 'bateria na końcu', '%', -1], ['max_track_error', 'maks. odchyłka od trasy', 'm', 1], ['p95_track_error', 'odchyłka p95', 'm', 1],
    ['min_altitude', 'min. wysokość', 'm', 1], ['max_tilt_deg', 'maks. przechył/pochylenie', '°', 0], ['max_wind', 'maks. wiatr', 'm/s', 1],
    ['min_airspeed', 'min. IAS', 'm/s', 1], ['min_stall_margin', 'zapas do przeciągnięcia', '×', 2], ['max_load_factor', 'maks. przeciążenie', 'g', 2],
    ['max_altitude_error', 'maks. błąd wysokości', 'm', 1], ['saturation_time', 'nasycenie silników', 's', 1], ['max_ice', 'oblodzenie', '%', -1]];
  $('#kvs').innerHTML = kv.filter(([k]) => k in m).map(([k, l, u, d]) =>
    `<div class="kv"><b>${d === -1 ? fmt(m[k] * 100, 0) : fmt(m[k], d)} ${u}</b><span>${l}</span></div>`).join('');
  if (S.replayUrl) URL.revokeObjectURL(S.replayUrl);
  S.replayUrl = URL.createObjectURL(new Blob([r.replay_html], { type: 'text/html' }));
  $('#replay').src = S.replayUrl;
  drawCharts(r);
}

const PALETTE = ['#1f6fb2', '#e76f51', '#2a9d8f', '#8a63d2', '#d9822b', '#7a8691'];
function drawCharts(r) {
  const s = r.series, t = s.t;
  const fw = r.kind === 'fixed_wing';
  const defs = [
    ['Wysokość [m]', [['rzeczywista', s.alt], ['barometryczna', s.est_alt], fw && ['zadana', s.h_sp]]],
    ['Prędkość [m/s]', [['TAS', s.tas], fw && ['IAS (Pitot)', s.est_ias], ['nad ziemią', s.gs], ['wiatr', s.wind_speed]]],
    ['Orientacja [°]', [['przechylenie', s.roll], ['pochylenie', s.pitch]]],
    ['Akumulator [%] i przepustnica [%]', [['SOC', s.batt_soc && s.batt_soc.map(v => v * 100)], ['przepustnica', s.throttle && s.throttle.map(v => v * 100)]]],
    ['Moc elektryczna [W]', [['moc', s.power]]],
    ['Turbulencja i prądy pionowe [m/s]', [['|turbulencja|', s.turb], ['prąd pionowy', s.updraft]]],
  ];
  $('#charts').innerHTML = defs.map((d, i) => `<div class="chart"><h4>${d[0]}</h4><div class="legend">${d[1].filter(x => x && x[1]).map((x, j) =>
    `<span><i style="background:${PALETTE[j]}"></i>${x[0]}</span>`).join('')}</div><canvas id="ch${i}"></canvas></div>`).join('');
  defs.forEach((d, i) => lineChart($('#ch' + i), t, d[1].filter(x => x && x[1]).map(x => x[1]), r.events));
}

function lineChart(cv, t, series, events) {
  const dpr = devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr;
  const c = cv.getContext('2d'); c.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement);
  const muted = css.getPropertyValue('--muted').trim(), line = css.getPropertyValue('--line').trim();
  let lo = Infinity, hi = -Infinity;
  series.forEach(s => s.forEach(v => { if (v != null && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
  if (!isFinite(lo)) return;
  if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const L = 44, Rr = 8, T = 6, B = 20, t0 = t[0], t1 = t[t.length - 1] || 1;
  const X = v => L + (v - t0) / (t1 - t0 || 1) * (W - L - Rr), Y = v => T + (hi - v) / (hi - lo) * (H - T - B);
  c.font = '11px system-ui'; c.fillStyle = muted; c.strokeStyle = line; c.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * i / 4, y = Y(v);
    c.beginPath(); c.moveTo(L, y); c.lineTo(W - Rr, y); c.stroke();
    c.fillText(Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1), 2, y + 4);
  }
  for (let i = 0; i <= 5; i++) { const v = t0 + (t1 - t0) * i / 5; c.fillText(v.toFixed(0) + ' s', X(v) - 10, H - 4); }
  (events || []).forEach(([te, m]) => {
    if (/AWARIA|KATASTROFA|FAILSAFE|PRZECIĄGNIĘCIE|Pitota/.test(m)) { c.strokeStyle = '#d6453d'; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(X(te), T); c.lineTo(X(te), H - B); c.stroke(); c.setLineDash([]); }
  });
  series.forEach((s, j) => {
    c.strokeStyle = PALETTE[j]; c.lineWidth = 1.5; c.beginPath();
    s.forEach((v, i) => { if (v == null) return; i ? c.lineTo(X(t[i]), Y(v)) : c.moveTo(X(t[i]), Y(v)); });
    c.stroke();
  });
}

// ============================================================ pobieranie
function download(name, data, type = 'text/plain') {
  const url = URL.createObjectURL(data instanceof Blob ? data : new Blob([data], { type }));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

// ============================================================ nawigacja
function showStep(n) {
  $$('.step').forEach(b => b.classList.toggle('active', b.dataset.step == n));
  $$('.panel').forEach(p => p.classList.toggle('active', p.id === 's' + n));
  if (n === 3 && S.result) setTimeout(() => drawCharts(S.result), 30);
  scrollTo({ top: 0, behavior: 'smooth' });
}

async function loadExample() {
  const ex = S.catalog.example;
  const base = 'data/examples/moj_samolot/';
  const files = [];
  for (const name of ex.files) {
    const r = await fetch(base + name);
    files.push(new File([await r.arrayBuffer()], name));
  }
  S.cfg = DEFAULT_CFG(); S.files = {};
  files.push(new File([ex.yaml], 'aircraft.yaml'));
  await addFiles(files);
}

async function onReady() {
  if (!S.catalog) {
    S.catalog = await call('catalog');
    $('#preset').innerHTML = S.catalog.vehicles.map(v => `<option value="${esc(v.path)}">${esc(v.name)}</option>`).join('');
    renderScenarioList();
  }
  $('#run').disabled = false;
  renderCad();
}

function init() {
  startWorker();
  bindForm();
  renderCad();
  $$('.step').forEach(b => b.onclick = () => !b.disabled && showStep(Number(b.dataset.step)));
  $$('input[name=vmode]').forEach(r => r.onchange = () => {
    S.vmode = r.value;
    $('#cad-box').classList.toggle('hidden', S.vmode !== 'cad');
    $('#preset-box').classList.toggle('hidden', S.vmode !== 'preset');
    $('#preset-next').classList.toggle('hidden', S.vmode !== 'preset');
    if (S.catalog) renderScenarioList();
  });
  $('#preset').onchange = () => renderScenarioList();
  const drop = $('#drop');
  ['dragenter', 'dragover'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(e => drop.addEventListener(e, () => drop.classList.remove('over')));
  drop.addEventListener('drop', ev => { ev.preventDefault(); addFiles([...ev.dataTransfer.files]); });
  $('#files').onchange = (e) => { const fl = [...e.target.files]; e.target.value = ''; addFiles(fl); };
  $('#load-example').onclick = () => S.catalog ? loadExample() : alert('Poczekaj na uruchomienie Pythona…');
  $('#clear-cad').onclick = () => { S.cfg = DEFAULT_CFG(); S.files = {}; S.analysis = null; renderCad(); };
  $('#save-yaml').onclick = async () => download('aircraft.yaml', await call('yaml_dump', { json: JSON.stringify(cleanCfg()) }), 'text/yaml');
  $('#analyze').onclick = analyze;
  $('#dl-vehicle').onclick = () => S.analysis && download('vehicle_generated.yaml', S.analysis.vehicle_yaml, 'text/yaml');
  $('#to-step2').onclick = () => showStep(2);
  $('#to-step2b').onclick = () => showStep(2);
  $('#scenario').onchange = loadScenarioInfo;
  $('#fails').onclick = (e) => { if (e.target.dataset.i) { S.fails.splice(Number(e.target.dataset.i), 1); renderFails(); } };
  $('#f-add').onclick = () => {
    const type = $('#f-type').value, t = Number($('#f-t').value) || 0, d = $('#f-dur').value;
    const f = Object.assign({ type, t }, FAIL_DEFAULTS[type] || {});
    if (d) f.duration = Number(d);
    S.fails.push(f); renderFails();
  };
  $('#run').onclick = runTest;
  $('#cancel').onclick = cancelRun;
  $('#dl-replay').onclick = () => S.result && download('replay.html', S.result.replay_html, 'text/html');
  $('#dl-csv').onclick = async () => download('log.csv', await call('csv'), 'text/csv');
  $('#dl-sc').onclick = async () => download('scenariusz.yaml', await call('scenario_yaml'), 'text/yaml');
  addEventListener('resize', () => { if ($('#s3').classList.contains('active') && S.result) drawCharts(S.result); });
}
init();
