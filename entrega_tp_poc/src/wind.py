"""ETAPA 3 - Estimador de viento (módulo separado del modelo predictivo).

Idea: en ceñida los barcos navegan en dos bordadas simétricas respecto del viento (rumbos W±β) y en
popa en dos bordadas W+180±γ. En una ventana de tiempo tomamos los rumbos estables de la flota, los
separamos en dos grupos (babor/estribor) respecto del eje de viento vigente y el viento es la
BISECTRIZ de las medias de ambos grupos. La separación entre grupos (2β) es la prueba de consistencia:
en Optimist debe ser ~80-100°.

Es CAUSAL: la estimación en el instante t usa solo datos de (t-win, t].
"""
import numpy as np
import pandas as pd
from .common import angdiff, circmean

WIN_S, STEP_S = 240, 30
MIN_PER_SIDE, MIN_BOATS_SIDE = 40, 4
BETA_UP = (15, 30, 52, 70)         # (cero, pleno-bajo, pleno-alto, cero)  semi-ángulo ceñida plausible (°)
BETA_DN = (12, 20, 45, 60)         # semi-ángulo popa plausible (°)
MAX_AGE_S = 600                   # antigüedad máxima de una estimación para seguir usándola


def _plaus(b, lo0, lo1, hi1, hi0):
    if b <= lo0 or b >= hi0:
        return 0.0
    if b < lo1:
        return (b - lo0) / (lo1 - lo0)
    if b > hi1:
        return (hi0 - b) / (hi0 - hi1)
    return 1.0


def _side_estimate(h, comp, ref, beta_rng):
    """Bisectriz de dos grupos de rumbos alrededor de la referencia 'ref'. Devuelve (viento, n_min, beta, conf)."""
    d = angdiff(h, ref)
    m = np.abs(d) < 90
    h, comp, d = h[m], comp[m], d[m]
    p, s = d < 0, d >= 0
    if p.sum() < MIN_PER_SIDE or s.sum() < MIN_PER_SIDE:
        return None
    if len(np.unique(comp[p])) < MIN_BOATS_SIDE or len(np.unique(comp[s])) < MIN_BOATS_SIDE:
        return None
    mu_p, mu_s = circmean(h[p]), circmean(h[s])
    beta = abs(float(angdiff(mu_s, mu_p))) / 2.0
    bis = circmean([mu_p, mu_s])
    n = int(min(p.sum(), s.sum()))
    if beta < beta_rng[0]:      # sin estructura de dos bordadas (la flota va derecho): no hay información de viento
        return None
    conf = min(1.0, n / 150.0) * _plaus(beta, *beta_rng)
    return bis, n, beta, conf


def estimate_wind(df, W0, leg_type, K, comp_subset=None):
    """df: telemetría de UNA regata. Devuelve DataFrame (t, wind, conf, beta_up, beta_dn, n_up, n_dn)."""
    d = df[df.stable & (df.speed_kn > 1.5) & (df.leg >= 0) & (df.leg <= K - 2)]
    if comp_subset is not None:
        d = d[d.competitor_id.isin(comp_subset)]
    lt = np.array(leg_type)[d.leg.values]
    d = d[(lt == 'U') | (lt == 'D')]
    lt = lt[(lt == 'U') | (lt == 'D')]
    d = d.assign(is_up=(lt == 'U'))
    d = d.sort_values('t')
    t, h, up = d.t.values, d.hdg.values, d.is_up.values
    comp = pd.factorize(d.competitor_id)[0]
    rows, wref = [], W0
    for g in np.arange(0, max(t.max(), 1), STEP_S) if len(t) else []:
        i0, i1 = np.searchsorted(t, g - WIN_S), np.searchsorted(t, g, side='right')
        if i1 - i0 < 2 * MIN_PER_SIDE:
            continue
        hh, cc, uu = h[i0:i1], comp[i0:i1], up[i0:i1]
        res = {}
        for _ in range(2):  # 2 pasadas: la segunda re-centra el reparto babor/estribor con el viento recién estimado
            ru = _side_estimate(hh[uu], cc[uu], wref, BETA_UP)
            rd = _side_estimate(hh[~uu], cc[~uu], (wref + 180) % 360, BETA_DN)
            ests = []
            if ru: ests.append((ru[0], ru[1], ru[3], 'u', ru))
            if rd: ests.append(((rd[0] + 180) % 360, rd[1], rd[3], 'd', rd))
            if not ests:
                break
            wref = circmean([e[0] for e in ests], [e[1] for e in ests])
            res = {e[3]: e for e in ests}
        if not res:
            continue
        ests = list(res.values())
        w = circmean([e[0] for e in ests], [e[1] for e in ests])
        conf = float(np.average([e[2] for e in ests], weights=[e[1] for e in ests]))
        if len(ests) == 2:   # coherencia entre ceñida y popa
            conf *= 1.0 - min(1.0, abs(float(angdiff(ests[0][0], ests[1][0]))) / 40.0)
        else:
            conf *= 0.8
        rows.append(dict(t=g, wind=w, conf=conf,
                         beta_up=res['u'][4][2] if 'u' in res else np.nan,
                         beta_dn=res['d'][4][2] if 'd' in res else np.nan,
                         n_up=res['u'][1] if 'u' in res else 0, n_dn=res['d'][1] if 'd' in res else 0))
        wref = w
    return pd.DataFrame(rows)


class WindSeries:
    """Serie causal: en t devuelve la última estimación (<10 min). Si no hay, cae al eje del recorrido con conf=0."""

    def __init__(self, est: pd.DataFrame, W0: float):
        self.t = est.t.values if len(est) else np.array([])
        self.w = est.wind.values if len(est) else np.array([])
        self.c = est.conf.values if len(est) else np.array([])
        self.W0 = W0

    def at(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        W, C, SRC = np.full(len(t), self.W0), np.zeros(len(t)), np.zeros(len(t), dtype=int)   # SRC 1 = estimado
        if len(self.t):
            i = np.searchsorted(self.t, t, side='right') - 1
            ok = (i >= 0)
            ok[ok] &= (t[ok] - self.t[i[ok]]) <= MAX_AGE_S
            W[ok], C[ok], SRC[ok] = self.w[i[ok]], self.c[i[ok]], 1
        return W, C, SRC


def validate_wind(race, seed=0):
    """Protocolo Fase 0: (a) consistencia split-half de barcos, (b) cobertura, (c) semi-ángulo de bordada."""
    df = race.df
    comps = np.array(list(df[(df.leg >= 0) & (df.leg < race.K - 1)].competitor_id.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(comps)
    a, b = comps[: len(comps) // 2], comps[len(comps) // 2:]
    ea = estimate_wind(df, race.W0, race.leg_type, race.K, a)
    eb = estimate_wind(df, race.W0, race.leg_type, race.K, b)
    full = estimate_wind(df, race.W0, race.leg_type, race.K)
    out = dict(event=race.event, race=race.name, boats=len(comps), windows_valid=len(full),
               windows_total=int(max(df.t.max(), 1) // STEP_S), median_conf=float(full.conf.median()) if len(full) else np.nan,
               beta_up_med=float(full.beta_up.median()) if len(full) else np.nan,
               beta_dn_med=float(full.beta_dn.median()) if len(full) else np.nan,
               wind_vs_course_axis_deg=float(np.median(np.abs(angdiff(full.wind, race.W0)))) if len(full) else np.nan)
    m = ea.merge(eb, on='t', suffixes=('_a', '_b'))
    out['split_half_n'] = len(m)
    out['split_half_median_abs_diff_deg'] = float(np.median(np.abs(angdiff(m.wind_a, m.wind_b)))) if len(m) else np.nan
    out['split_half_p90_abs_diff_deg'] = float(np.quantile(np.abs(angdiff(m.wind_a, m.wind_b)), .9)) if len(m) else np.nan
    return out, full
