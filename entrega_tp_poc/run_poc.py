"""POC TP SOFTWARE - corre el flujo completo: ingesta -> variables -> viento -> indicadores -> predicción -> visualización.

Uso:  python run_poc.py <carpeta_raw_tractrac> [carpeta_resultados]
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
from src.ingest import build_races
from src.wind import estimate_wind, validate_wind, WindSeries
from src.analytics import detect_maneuvers, build_snapshots, gain_loss_windows
from src.model import prepare, evaluate, feature_importance
from src.viz import make_figure

data_dir = Path(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else 'results')
out.mkdir(parents=True, exist_ok=True)
T0 = time.time()
log = lambda m: print(f"[{time.time()-T0:5.0f}s] {m}", flush=True)

# 1) ingesta + limpieza
rep = {}
races = build_races(data_dir, rep)
log(f"ingesta: {rep}")
clean = pd.concat([r.df.assign(race_id=r.race_id)[['race_id', 'competitor_id', 't', 'timestamp_unix_ms', 'latitude', 'longitude', 'x', 'y',
                                                    'speed_kn', 'cog', 'hdg', 'stable', 'leg']] for r in races.values()])
clean = clean.rename(columns={'t': 't_s_desde_largada', 'x': 'x_m', 'y': 'y_m', 'cog': 'rumbo_deg', 'hdg': 'rumbo_suavizado_deg', 'leg': 'pierna'})
clean.round({'t_s_desde_largada': 1, 'latitude': 6, 'longitude': 6, 'x_m': 2, 'y_m': 2, 'speed_kn': 2, 'rumbo_deg': 1, 'rumbo_suavizado_deg': 1}) \
     .to_csv(out / 'telemetria_limpia.csv.gz', index=False)
json.dump(rep, open(out / 'cleaning_report.json', 'w'), indent=1, ensure_ascii=False)

# 2-5) viento, maniobras, variables
val_rows, winds, mans, snaps = [], {}, [], []
for rid, r in races.items():
    v, est = validate_wind(r)
    val_rows.append(v)
    winds[rid] = WindSeries(est, r.W0)
    man = detect_maneuvers(r, winds[rid])
    mans.append(man)
    snaps.append(build_snapshots(r, winds[rid], man))
    log(f"{r.event[:28]} | {r.name[:16]} | maniobras {int(man.kind.isin(['tack','gybe']).sum())} | snapshots {len(snaps[-1])}")
val = pd.DataFrame(val_rows)
man = pd.concat(mans, ignore_index=True)
snap = pd.concat(snaps, ignore_index=True)
snap['event'] = snap.race_id.map({rid: r.event for rid, r in races.items()})
wd, tau = gain_loss_windows(snap, man)
val.round(2).to_csv(out / 'validacion_viento.csv', index=False)
man.round(2).to_csv(out / 'maniobras.csv.gz', index=False)
snap.round(3).to_csv(out / 'features_snapshots.csv.gz', index=False)
wd.round(2).to_csv(out / 'ventanas_ganancia_perdida.csv', index=False)
log(f"ventanas ganancia/pérdida: {len(wd)} (umbral ±{tau:.1f} m en 60 s)")

# 6) modelo
d = prepare(snap)
mets, oof_race = [], None
for scheme in ('race', 'event'):
    m, oof = evaluate(d, scheme)
    mets.append(m)
    if scheme == 'race':
        oof_race = oof
    log(f"evaluación '{scheme}':\n{m.round(3).to_string(index=False)}")
mets = pd.concat(mets)
mets.round(4).to_csv(out / 'metricas_modelo.csv', index=False)
fi = feature_importance(d)
fi.round(4).to_csv(out / 'importancia_variables.csv', index=False)
log(f"top variables:\n{fi.head(8).round(3).to_string(index=False)}")

# 7) visualización: una página por regata + índice con selector de regata
d['pred'] = oof_race['M-conviento (estado + acción + viento)']
snap = snap.merge(d[['race_id', 'competitor_id', 'g', 'pred']], on=['race_id', 'competitor_id', 'g'], how='left')
boats = pd.read_csv(data_dir / 'boats.csv')
names = {(r.race_id, r.competitor_id): r.boat_short_name for r in boats.itertuples()}
import re, html as _html
vdir = out / 'trayectorias'
vdir.mkdir(exist_ok=True)
index = []
for rid, r in races.items():
    try:
        fig = make_figure(r, snap, man, winds[rid], {c: names.get((rid, c), c[:6]) for c in snap.competitor_id.unique()}, pred=snap.pred.values)
    except Exception as e:                                   # regata sin datos suficientes para graficar
        log(f"viz omitida ({r.event} / {r.name}): {e}")
        continue
    fname = re.sub(r'[^A-Za-z0-9]+', '_', f"{r.event}_{r.name}").strip('_')[:80] + '.html'
    fig.write_html(vdir / fname, include_plotlyjs='cdn')
    nb = snap[snap.race_id == rid].competitor_id.nunique()
    index.append((r.event, r.name, nb, fname))
index.sort()
opts, last = [], None
for ev, nm, nb, f in index:
    if ev != last:
        opts.append(f'<optgroup label="{_html.escape(ev)}">'if last is None else f'</optgroup><optgroup label="{_html.escape(ev)}">'); last = ev
    opts.append(f'<option value="trayectorias/{f}">{_html.escape(nm)} ({nb} barcos)</option>')
opts.append('</optgroup>')
page = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Trayectorias por regata</title>
<style>body{{margin:0;font-family:system-ui,sans-serif}}header{{padding:10px 16px;background:#f4f6fa;border-bottom:1px solid #d5dae3;display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
select{{font-size:15px;padding:6px 8px;min-width:360px}}iframe{{border:0;width:100%;height:calc(100vh - 58px)}}small{{color:#666}}</style></head><body>
<header><b>Regata:</b><select id="r" onchange="document.getElementById('f').src=this.value">{''.join(opts)}</select>
<small>{len(index)} regatas · elegí la regata acá y el barco en el desplegable del gráfico (requiere internet para cargar Plotly)</small></header>
<iframe id="f" src="{'trayectorias/' + index[0][3]}"></iframe></body></html>"""
(out / 'trayectoria_verde_rojo.html').write_text(page, encoding='utf-8')
log(f"viz: {len(index)} regatas en {vdir}/ (abrir trayectoria_verde_rojo.html)")
log("LISTO")
