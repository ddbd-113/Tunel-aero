// Web Worker: Python (Pyodide) + Tunel-aero liczą symulację w tle, interfejs pozostaje płynny.
const PYODIDE_URL = 'https://cdn.jsdelivr.net/pyodide/v0.29.5/full/';
const VERSION = '__VERSION__';
importScripts(PYODIDE_URL + 'pyodide.js');

let py = null, api = null;

function status(text) { postMessage({ type: 'status', text }); }

async function init() {
  status('Uruchamianie Pythona w przeglądarce (Pyodide)…');
  py = await loadPyodide({ indexURL: PYODIDE_URL });
  status('Ładowanie numpy i pyyaml…');
  await py.loadPackage(['numpy', 'pyyaml']);
  status('Ładowanie Tunel-aero…');
  const resp = await fetch('py/tunel_aero.zip?v=' + VERSION);
  if (!resp.ok) throw new Error('Nie znaleziono py/tunel_aero.zip (' + resp.status + ')');
  py.unpackArchive(await resp.arrayBuffer(), 'zip', { extractDir: '/home/pyodide/app' });
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide/app')");
  api = py.pyimport('tunel_aero.webapi');
}

const ready = init().then(() => postMessage({ type: 'ready' }))
  .catch(e => postMessage({ type: 'fatal', error: String(e && e.message || e) }));

function filesArg(files) {
  if (!files || !Object.keys(files).length) return null;
  return py.toPy(files);
}

onmessage = async (ev) => {
  const { id, cmd, args = {} } = ev.data;
  try {
    await ready;
    if (!api) throw new Error('Środowisko Pythona nie zostało uruchomione');
    let out;
    switch (cmd) {
      case 'catalog': out = api.catalog(); break;
      case 'scenario_info': out = api.scenario_info(args.path); break;
      case 'yaml_load': out = api.yaml_load(args.text); break;
      case 'yaml_dump': out = api.yaml_dump(args.json); break;
      case 'analyze': out = api.analyze_aircraft(args.cfg, filesArg(args.files)); break;
      case 'run': {
        let last = -1;
        const progress = (f) => {
          const p = Math.round(f * 100);
          if (p !== last) { last = p; postMessage({ type: 'progress', id, value: f }); }
        };
        out = api.run_test(args.scenario, args.overrides || '{}', args.vehicle || '', filesArg(args.files), progress);
        break;
      }
      case 'csv': out = JSON.stringify(api.last_csv()); break;
      case 'scenario_yaml': out = JSON.stringify(api.last_scenario_yaml()); break;
      default: throw new Error('Nieznana komenda: ' + cmd);
    }
    postMessage({ type: 'result', id, ok: true, result: cmd === 'yaml_dump' ? out : JSON.parse(out) });
  } catch (e) {
    let msg = String(e && e.message || e);
    const m = msg.match(/(\w*Error: [^\n]+)\s*$/);    // ostatnia linia wyjątku Pythona
    const clean = (m ? m[1] : msg).replace(/^(ValueError|RuntimeError|KeyError|FileNotFoundError): /, '');
    postMessage({ type: 'result', id, ok: false, error: clean });
  }
};
