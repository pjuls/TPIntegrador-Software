"""ETAPA 7 - Visualización: trayectoria coloreada de verde (ganó) a rojo (perdió) contra la flota."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

CLIP = 60.0   # m: saturación de la escala de color


NBINS = 15
NOREF = 'rgb(175,175,175)'


def make_figure(race, snap, man, wind, names, pred=None, n_boats=None):
    """n_boats=None -> todos los barcos de la regata (ordenados por llegada)."""
    s = snap[snap.race_id == race.race_id].copy()
    if pred is not None:
        s['pred'] = pred[snap.race_id.eq(race.race_id).values] if len(pred) == len(snap) else np.nan
    fin = race.df[race.df.leg == race.K - 1].groupby('competitor_id').t.min().sort_values()
    sel = list(fin.index) if n_boats is None else [fin.index[i] for i in np.unique(np.linspace(0, len(fin) - 1, n_boats).astype(int))]
    sel = [c for c in sel if (s.competitor_id == c).any()]
    fig = go.Figure()
    per_boat = []
    bins = np.linspace(0, 1, NBINS)
    for comp in sel:
        d = s[s.competitor_id == comp].sort_values('g').reset_index(drop=True)
        first = len(fig.data)
        # tramos agrupados por color: un trace por color (15 + gris) en vez de uno por tramo
        xs = {i: [] for i in range(NBINS + 1)}
        ys = {i: [] for i in range(NBINS + 1)}
        for i in range(1, len(d)):
            a_, b_ = d.iloc[i - 1], d.iloc[i]
            if b_.g - a_.g > 20:
                continue
            gv = b_.gain_past60
            j = NBINS if pd.isna(gv) else int(np.argmin(np.abs(bins - (np.clip(gv, -CLIP, CLIP) + CLIP) / (2 * CLIP))))
            xs[j] += [round(a_.x, 1), round(b_.x, 1), None]
            ys[j] += [round(a_.y, 1), round(b_.y, 1), None]
        for j in range(NBINS + 1):
            if xs[j]:
                col = NOREF if j == NBINS else _rg(bins[j])
                fig.add_trace(go.Scatter(x=xs[j], y=ys[j], mode='lines', line=dict(color=col, width=5),
                                         hoverinfo='skip', showlegend=False, visible=False))
        has_pred = 'pred' in d
        gstr = [f"{v:+.0f} m" if pd.notna(v) else "sin referencia (ya terminó parte de la flota)" for v in d.gain_past60]
        pstr = [(f"<br>predicción próx. 60 s: {v:+.0f} m" if pd.notna(v) else "") for v in d.pred] if has_pred else [""] * len(d)
        cdata = [[round(g / 60, 1), int(l) + 1, gs, round(rk * 100), round(w), round(c, 2), round(tw), round(sp, 1), ps]
                 for g, l, gs, rk, w, c, tw, sp, ps in zip(d.g, d.leg, gstr, d.rank_pct, d.wind, d.wind_conf, d.twa, d.speed_kn, pstr)]
        fig.add_trace(go.Scatter(x=d.x.round(1), y=d.y.round(1), mode='markers', marker=dict(size=7, opacity=0), customdata=cdata,
                                 hovertemplate=("t=%{customdata[0]:.1f} min | pierna %{customdata[1]}<br>ganancia últ. 60 s: %{customdata[2]}"
                                                "<br>posición en flota: %{customdata[3]}%<br>viento est.: %{customdata[4]}° (conf %{customdata[5]:.2f}) | TWA %{customdata[6]:+d}°"
                                                "<br>vel: %{customdata[7]:.1f} kn%{customdata[8]}<extra></extra>"),
                                 showlegend=False, visible=False))
        mv = man[(man.race_id == race.race_id) & (man.competitor_id == comp) & man.kind.isin(['tack', 'gybe'])]
        if len(mv):
            px = np.interp(mv.t, d.g, d.x); py = np.interp(mv.t, d.g, d.y)
            fig.add_trace(go.Scatter(x=px.round(1), y=py.round(1), mode='markers', marker=dict(symbol='x', size=8, color='black'),
                                     text=[f"{'virada' if k == 'tack' else 'trasluchada'} | costo {c:.1f} m" if pd.notna(c) else k for k, c in zip(mv.kind, mv.cost_m)],
                                     hoverinfo='text', name='virada / trasluchada', showlegend=False, visible=False))
        per_boat.append((first, len(fig.data)))
    base = len(fig.data)
    _add_course(fig, race)
    Wm = float(np.median(s.wind))
    _add_wind_arrow(fig, race, Wm)
    fig.add_trace(go.Scatter(                      # escala de color (barra lateral)
        x=[None], y=[None], mode='markers', hoverinfo='skip', showlegend=False,
        marker=dict(size=0.001, color=[0], cmin=-CLIP, cmax=CLIP, showscale=True,
                    colorscale=[[0, 'rgb(215,48,39)'], [.5, 'rgb(215,200,39)'], [1, 'rgb(26,152,80)']],
                    colorbar=dict(title=dict(text='Ganancia vs flota (últimos 60 s)', side='right'),
                                  tickvals=[-CLIP, 0, CLIP], ticktext=[f'-{int(CLIP)} m (perdió)', '0', f'+{int(CLIP)} m (ganó)'],
                                  len=0.45, x=1.01, y=0.5, thickness=16))))
    static_end = len(fig.data)
    buttons = []
    for bi, comp in enumerate(sel):
        vis = [False] * len(fig.data)
        a, b = per_boat[bi]
        for j in range(a, b):
            vis[j] = True
        for j in range(base, static_end):
            vis[j] = True
        pos = list(fin.index).index(comp) + 1
        buttons.append(dict(label=f"{names.get(comp, comp[:6])} (llegó {pos}º de {len(fin)})", method='update', args=[{'visible': vis}]))
    a, b = per_boat[0]
    for j in list(range(a, b)) + list(range(base, static_end)):
        fig.data[j].visible = True
    fig.update_layout(
        title=dict(text=f"<b>{race.event} · {race.name}</b><br><sup>Recorrido: {_course_summary(race)}</sup>", x=0.01),
        updatemenus=[dict(buttons=buttons, direction='down', x=0.01, y=1.0, xanchor='left', yanchor='bottom')],
        xaxis=dict(title='este (m)', scaleanchor='y', scaleratio=1, zeroline=False),
        yaxis=dict(title='norte (m)', zeroline=False),
        height=820, template='plotly_white', margin=dict(t=130, b=150, r=130),
        annotations=list(fig.layout.annotations) + [dict(
            xref='paper', yref='paper', x=0, y=-0.10, xanchor='left', yanchor='top', showarrow=False, align='left',
            text=("<b>Cómo leerlo:</b> la línea gruesa es la trayectoria del barco elegido: <span style='color:#1a9850'><b>verde</b></span> = estaba ganando terreno "
                  "contra la flota, <span style='color:#d73027'><b>rojo</b></span> = perdiéndolo (ventana de 60 s).<br>"
                  "Los <b>rombos de color</b> son las boyas (al pasar el mouse se ve su nombre original de TracTrac). "
                  "La línea punteada gris es el recorrido oficial. <b>✕</b> = viradas / trasluchadas.<br>"
                  "<span style='color:#888'><b>Tramo gris</b></span> = sin referencia confiable (ya terminó más del 25% de la flota; comparar ahí daría falsas ganancias)."))])

    return fig


def _rg(frac):
    """0 -> rojo, 0.5 -> amarillo, 1 -> verde."""
    if frac < 0.5:
        k = frac / 0.5
        r, g, b = 215, int(48 + (200 - 48) * k), 39
    else:
        k = (frac - 0.5) / 0.5
        r, g, b = int(215 - (215 - 26) * k), int(200 - (200 - 152) * k), int(39 + (80 - 39) * k)
    return f'rgb({r},{g},{b})'


# ----------------------------------------------------------------------------- recorrido y boyas
ROLE_STYLE = {'Salida': '#1f77b4', 'Barlovento': '#ff7f0e', 'Través': '#9467bd', 'Sotavento (puerta)': '#17becf', 'Llegada': '#222222'}


def _key(xy):
    return (round(xy[0] / 3), round(xy[1] / 3))


def _roles(race):
    """Función de cada control de la ruta, según DÓNDE está la boya respecto del viento:
    la primera y la última son salida y llegada; el resto se clasifica por su posición sobre el eje del viento
    (arriba = barlovento, abajo = sotavento/puerta, en el medio = través). Una misma boya siempre tiene el mismo rol."""
    K = len(race.marks)
    k0, kl = _key(race.marks[0]), _key(race.marks[-1])
    w = np.deg2rad(race.W0)
    proj = lambda xy: xy[0] * np.sin(w) + xy[1] * np.cos(w)          # posición a lo largo del eje del viento
    mid = [proj(race.marks[k]) for k in range(1, K - 1) if _key(race.marks[k]) not in (k0, kl)]
    lo, hi = (min(mid), max(mid)) if mid else (0.0, 1.0)
    out = []
    for k in range(K):
        key = _key(race.marks[k])
        if k == 0 or key == k0:
            out.append('Salida')
        elif k == K - 1 or key == kl:
            out.append('Llegada')
        else:
            f = (proj(race.marks[k]) - lo) / max(hi - lo, 1e-9)
            out.append('Barlovento' if f >= 0.65 else 'Sotavento (puerta)' if f <= 0.35 else 'Través')
    return out


def _course_summary(race):
    r = _roles(race)
    seq = [r[0]] + [x for i, x in enumerate(r[1:], 1) if x != r[i - 1]]
    return ' → '.join(seq)


def _add_course(fig, race):
    marks, names, roles = race.marks, race.mark_names, _roles(race)
    fig.add_trace(go.Scatter(x=marks[:, 0], y=marks[:, 1], mode='lines', line=dict(color='#9aa0a6', width=2, dash='dot'),
                             hoverinfo='skip', showlegend=False))
    # boyas únicas (por posición); el rol es el de su primera llegada
    uniq = {}
    for k, (xy, nm, ro) in enumerate(zip(marks, names, roles)):
        key = (round(xy[0] / 3), round(xy[1] / 3))
        if key not in uniq:
            uniq[key] = dict(x=xy[0], y=xy[1], role=ro, name=nm, steps=[k + 1])
        else:
            uniq[key]['steps'].append(k + 1)
    pts = list(uniq.values())
    fig.add_trace(go.Scatter(
        x=[p['x'] for p in pts], y=[p['y'] for p in pts], mode='markers', showlegend=False,
        marker=dict(symbol='diamond', size=15, color=[ROLE_STYLE[p['role']] for p in pts], line=dict(color='white', width=2)),
        text=[f"<b>{p['role']}</b><br>nombre en TracTrac: «{p['name']}»<br>se pasa en los pasos: {', '.join(map(str, p['steps']))}" for p in pts],
        hoverinfo='text'))
    # una etiqueta por grupo de boyas del mismo rol y cercanas (ej. las dos boyas de una puerta)
    groups = []
    for p in pts:
        for g in groups:
            if g['role'] == p['role'] and np.hypot(g['x'] - p['x'], g['y'] - p['y']) < 250:
                g['members'].append(p); g['x'] = np.mean([m['x'] for m in g['members']]); g['y'] = np.mean([m['y'] for m in g['members']])
                break
        else:
            groups.append(dict(role=p['role'], x=p['x'], y=p['y'], members=[p]))
    # si un mismo rol tiene varios grupos separados (ej. dos boyas de barlovento lejos entre sí), se numeran por orden de paso
    for g in groups:
        g['step'] = min(st for m in g['members'] for st in m['steps'])
    for role in {g['role'] for g in groups}:
        same = sorted([g for g in groups if g['role'] == role], key=lambda g: g['step'])
        for i, g in enumerate(same, 1):
            g['num'] = f' {i}' if len(same) > 1 else ''
    label = lambda g: g['role'] + g['num'] + (' · 2 boyas' if len(g['members']) > 1 and g['role'] != 'Salida' else '')
    fig.add_trace(go.Scatter(
        x=[g['x'] for g in groups], y=[g['y'] + 0.03 * np.ptp(marks[:, 1]) for g in groups], mode='text', showlegend=False, hoverinfo='skip',
        text=[f"<b>{label(g)}</b>" for g in groups], textposition='top center',
        textfont=dict(size=13, color=[ROLE_STYLE[g['role']] for g in groups])))


def _add_wind_arrow(fig, race, wind_from):
    """Flecha de viento en la esquina del gráfico, apuntando hacia donde SOPLA (el viento viene de wind_from)."""
    xs, ys = race.marks[:, 0], race.marks[:, 1]
    L = 0.18 * max(np.ptp(xs), np.ptp(ys))
    x0, y0 = xs.min() - 0.15 * np.ptp(xs), ys.max() + 0.02 * np.ptp(ys)       # esquina superior izquierda
    dx, dy = -L * np.sin(np.deg2rad(wind_from)), -L * np.cos(np.deg2rad(wind_from))
    pts = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO']
    card = pts[int(((wind_from % 360) + 22.5) // 45) % 8]
    fig.add_annotation(x=x0 + dx, y=y0 + dy, ax=x0, ay=y0, xref='x', yref='y', axref='x', ayref='y', showarrow=True,
                       arrowhead=3, arrowwidth=4, arrowcolor='#1f77b4',
                       text=f'<b>Viento</b><br>viene del {card} ({wind_from:.0f}°)', font=dict(color='#1f77b4', size=13), yshift=14)
