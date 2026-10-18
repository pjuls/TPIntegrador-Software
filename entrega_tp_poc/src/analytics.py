"""ETAPAS 2, 4 y 5 - Variables relativas, maniobras (virada/trasluchada), costo y ventanas de ganancia/pérdida."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .common import angdiff, bearing, MS_PER_KN
from .wind import WindSeries

STEP = 15            # s entre snapshots de decisión
HORIZON = 60         # s hacia adelante para el target / hacia atrás para 'ganancia observada'
K_H = HORIZON // STEP
TURN_DEG = 55.0      # cambio de rumbo mínimo para considerar maniobra
MIN_FLEET = 8        # mínimo de barcos activos para calcular métricas relativas


# ------------------------------------------------------------------ maniobras
def detect_maneuvers(race, wind: WindSeries):
    rows = []
    mk = race.marks
    for comp, d in race.df[(race.df.leg >= 0) & (race.df.leg <= race.K - 2)].groupby('competitor_id'):
        if len(d) < 30:
            continue
        t, x, y, leg = d.t.values, d.x.values, d.y.values, d.leg.values
        spd = d.speed_kn.values
        s, c = np.sin(np.deg2rad(d.cog.values)), np.cos(np.deg2rad(d.cog.values))
        ps = pd.Series(s).rolling(4).mean().values
        pc = pd.Series(c).rolling(4).mean().values
        past = np.rad2deg(np.arctan2(ps, pc)) % 360
        fut = np.full_like(past, np.nan)
        fut[:-4] = past[4:]                    # ventana de las 4 muestras siguientes
        turn = np.abs(angdiff(fut, past))
        span = np.full_like(t, np.inf)
        span[3:-4] = t[7:] - t[:-7]
        cand = (turn > TURN_DEG) & (span < 30) & (spd > 1.0)
        idx = np.where(cand)[0]
        if len(idx) == 0:
            continue
        runs = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)
        for r in runs:
            i = r[np.argmax(turn[r])]
            te = t[min(i + 2, len(t) - 1)]
            j = min(i + 2, len(t) - 1)
            W = wind.at(te)[0][0]
            hb, ha = past[i], fut[i]
            db, da = angdiff(hb, W), angdiff(ha, W)
            if abs(db) < 90 and abs(da) < 90 and np.sign(db) != np.sign(da):
                kind = 'tack'
            elif abs(db) > 90 and abs(da) > 90 and np.sign(angdiff(hb, W + 180)) != np.sign(angdiff(ha, W + 180)):
                kind = 'gybe'
            else:
                kind = 'other'
            # cerca de una marca (redondeo) => no cuenta como maniobra táctica
            nxt = mk[min(leg[j] + 1, race.K - 1)]
            near_mark = np.hypot(x[j] - nxt[0], y[j] - nxt[1]) < 60 or (j > 0 and leg[min(j + 6, len(leg) - 1)] != leg[max(j - 6, 0)])
            if near_mark and kind != 'tack':
                kind = 'rounding'
            elif near_mark:
                kind = 'rounding'
            cost_m = cost_s = np.nan
            if kind in ('tack', 'gybe') and te - 30 >= t[0] and te + 35 <= t[-1]:
                # costo = avance hacia barlovento (tack) / sotavento (gybe) que se esperaba en 25 s [te-10, te+15] menos el real.
                # Avance esperado = velocidad media proyectada sobre el eje del viento en tramos estables antes y después.
                ang = np.deg2rad(W if kind == 'tack' else W + 180.0)
                ax = np.array([np.sin(ang), np.cos(ang)])
                ref = ((t >= te - 30) & (t <= te - 10)) | ((t >= te + 15) & (t <= te + 35))
                if ref.sum() >= 6:
                    vx, vy = np.gradient(x, t), np.gradient(y, t)
                    v_ref = float(np.mean(vx[ref] * ax[0] + vy[ref] * ax[1]))
                    p0 = np.array([np.interp(te - 10, t, x), np.interp(te - 10, t, y)])
                    p1 = np.array([np.interp(te + 15, t, x), np.interp(te + 15, t, y)])
                    cost_m = v_ref * 25.0 - float((p1 - p0) @ ax)
                    cost_s = cost_m / v_ref if v_ref > 0.3 else np.nan
            rows.append(dict(race_id=race.race_id, competitor_id=comp, t=float(te), kind=kind, turn_deg=float(turn[i]),
                             wind=float(W), leg=int(leg[j]), leg_type=race.leg_type[int(leg[j])],
                             cost_m=cost_m, cost_s=cost_s))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ snapshots + features
def build_snapshots(race, wind: WindSeries, man: pd.DataFrame):
    mk, K = race.marks, race.K
    grid = np.arange(0, race.df.t.max(), STEP)
    parts = []
    for comp, d in race.df[(race.df.leg >= 0) & (race.df.leg <= K - 2)].groupby('competitor_id'):
        t = d.t.values
        idx = np.searchsorted(t, grid, side='right') - 1
        ok = (idx >= 0) & (grid - t[np.clip(idx, 0, None)] <= 6) & (grid >= t[0])
        if ok.sum() == 0:
            continue
        s = d.iloc[idx[ok]][['t', 'x', 'y', 'speed_kn', 'hdg', 'leg']].copy()
        s['g'] = grid[ok]
        s['k'] = (grid[ok] // STEP).astype(int)
        s['competitor_id'] = comp
        parts.append(s)
    s = pd.concat(parts, ignore_index=True)
    leg = s.leg.values
    m0, m1 = mk[leg], mk[leg + 1]
    v = m1 - m0
    vl = np.hypot(v[:, 0], v[:, 1])
    p = s[['x', 'y']].values
    to = m1 - p
    s['dist_next'] = np.hypot(to[:, 0], to[:, 1])
    s['brg_mark'] = bearing(to[:, 0], to[:, 1])
    s['remaining'] = s.dist_next + race.tail[leg + 1]
    s['frac_leg'] = np.clip(1 - s.dist_next / vl, 0, 1)
    s['xt_m'] = -(v[:, 0] * (p[:, 1] - m0[:, 1]) - v[:, 1] * (p[:, 0] - m0[:, 0])) / vl   # + = a la derecha del eje de la pierna
    s['leg_type'] = pd.Categorical(np.array(race.leg_type)[leg], categories=['U', 'R', 'D']).codes
    s['vmg_kn'] = s.speed_kn * np.cos(np.deg2rad(angdiff(s.hdg, s.brg_mark)))

    # viento disponible en el instante de decisión (causal)
    W, C, SRC = wind.at(s.g.values)
    W5, _, _ = wind.at(np.maximum(s.g.values - 300, 0))
    s['wind'], s['wind_conf'], s['wind_est'] = W, C, SRC
    s['wind_shift_5m'] = angdiff(W, W5)
    s['twa'] = angdiff(s.hdg, W)
    s['abs_twa'] = s.twa.abs()
    s['mark_vs_wind'] = angdiff(s.brg_mark, W)

    # métricas de flota por instante
    s = s.sort_values(['g', 'remaining']).reset_index(drop=True)
    grp = s.groupby('g')
    s['n_active'] = grp.remaining.transform('size')
    s['gap_median_m'] = s.remaining - grp.remaining.transform('median')       # + = detrás de la mediana
    s['gap_leader_m'] = s.remaining - grp.remaining.transform('min')
    s['rank_pct'] = grp.remaining.rank(pct=True)
    s['xt_rel_m'] = s.xt_m - s.groupby(['g', 'leg']).xt_m.transform('mean')
    near = np.zeros(len(s), dtype=int)
    for _, ix in grp.indices.items():
        pts = s.loc[ix, ['x', 'y']].values
        near[ix] = [len(q) - 1 for q in cKDTree(pts).query_ball_point(pts, 100.0)]
    s['n_near100'] = near
    s = s[s.n_active >= MIN_FLEET].copy()

    # ganancia observada (pasado) y target (futuro) vs. mediana de la flota; + = ganó terreno
    key = s.set_index(['competitor_id', 'k']).gap_median_m
    kk = list(zip(s.competitor_id, s.k))
    s['gain_past60'] = [key.get((c, k - K_H), np.nan) - key.get((c, k), np.nan) for c, k in kk]
    s['y_gain_fut60'] = [key.get((c, k), np.nan) - key.get((c, k + K_H), np.nan) for c, k in kk]
    s['t_race_s'] = s.g

    # acción tomada y contexto de maniobras (por barco)
    ev = man[man.kind.isin(['tack', 'gybe'])]
    for col in ['act_tack_30', 'act_gybe_30', 'n_man_2min', 'sec_since_man']:
        s[col] = 0.0
    for comp, ix in s.groupby('competitor_id').indices.items():
        e = ev[ev.competitor_id == comp]
        gg = s.g.values[ix]
        for kind, col in (('tack', 'act_tack_30'), ('gybe', 'act_gybe_30')):
            te = np.sort(e[e.kind == kind].t.values)
            s.iloc[ix, s.columns.get_loc(col)] = (np.searchsorted(te, gg + 30, 'right') - np.searchsorted(te, gg, 'right') > 0).astype(float)
        te = np.sort(e.t.values)
        if len(te) == 0:
            s.iloc[ix, s.columns.get_loc('sec_since_man')] = 600
            continue
        s.iloc[ix, s.columns.get_loc('n_man_2min')] = np.searchsorted(te, gg, 'right') - np.searchsorted(te, gg - 120, 'right')
        last = np.searchsorted(te, gg, 'right') - 1
        s.iloc[ix, s.columns.get_loc('sec_since_man')] = np.where(last >= 0, np.minimum(gg - te[np.clip(last, 0, None)], 600), 600)
    s.insert(0, 'race_id', race.race_id)
    return s.sort_values(['competitor_id', 'g']).reset_index(drop=True)


# ------------------------------------------------------------------ ventanas de ganancia / pérdida
def gain_loss_windows(snap: pd.DataFrame, man: pd.DataFrame, tau=None):
    s = snap.dropna(subset=['gain_past60']).copy()
    tau = tau if tau is not None else 0.5 * float(s.gain_past60.std())
    s['state'] = np.where(s.gain_past60 > tau, 'gain', np.where(s.gain_past60 < -tau, 'loss', 'neutral'))
    out = []
    for (rid, comp), d in s.groupby(['race_id', 'competitor_id']):
        d = d.sort_values('g')
        blk = ((d.state != d.state.shift()) | (d.g.diff() > STEP * 1.5)).cumsum()
        for _, b in d.groupby(blk):
            if b.state.iloc[0] == 'neutral' or len(b) < 2:
                continue
            t0, t1 = b.g.iloc[0] - HORIZON, b.g.iloc[-1]
            m = man[(man.race_id == rid) & (man.competitor_id == comp) & (man.t >= t0) & (man.t <= t1) & man.kind.isin(['tack', 'gybe'])]
            out.append(dict(race_id=rid, competitor_id=comp, state=b.state.iloc[0], t_start=t0, t_end=t1,
                            leg=int(b.leg.iloc[0]), net_gain_m=float(b.gain_past60.iloc[np.argmax(np.abs(b.gain_past60.values))]),
                            n_tacks=int((m.kind == 'tack').sum()), n_gybes=int((m.kind == 'gybe').sum()),
                            maneuver_cost_m=float(m.cost_m.sum(skipna=True))))
    return pd.DataFrame(out), tau
