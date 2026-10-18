"""ETAPA 6 - Modelo predictivo (separado del estimador de viento).

Target : y_gain_fut60 = metros que el barco gana (+) o pierde (-) contra la mediana de la flota en los próximos 60 s.
Entrada: estado del barco/flota/recorrido + viento estimado disponible en t + acción (virada/trasluchada en los próximos 30 s).
No se afirma causalidad: se compara el resultado esperado de acciones en situaciones similares.
Evaluación: siempre sobre regatas completas no vistas (GroupKFold por regata) y, más estricto, dejando afuera un evento entero.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score
from .common import angdiff

STATE = ['leg', 'leg_type', 'dist_next', 'remaining', 'frac_leg', 'speed_kn', 'vmg_kn', 'hdg_vs_mark', 'gap_median_m',
         'gap_leader_m', 'rank_pct', 'xt_m', 'xt_rel_m', 'n_near100', 'n_active', 'gain_past60', 'n_man_2min',
         'sec_since_man', 't_race_s']
WIND = ['twa', 'abs_twa', 'wind_conf', 'wind_est', 'wind_shift_5m', 'mark_vs_wind']
ACTION = ['act_tack_30', 'act_gybe_30']
TARGET = 'y_gain_fut60'


def prepare(snap: pd.DataFrame) -> pd.DataFrame:
    d = snap.copy()
    d['hdg_vs_mark'] = angdiff(d.hdg, d.brg_mark)
    return d.dropna(subset=[TARGET, 'gain_past60']).reset_index(drop=True)


def _metrics(y, p, tau):
    big = np.abs(y) > tau
    return dict(MAE=mean_absolute_error(y, p), R2=r2_score(y, p), spearman=float(spearmanr(y, p)[0]) if np.std(p) > 0 else 0.0,
                dir_acc_big_moves=float(np.mean(np.sign(y[big]) == np.sign(p[big]))) if big.sum() else np.nan)


def _gbm():
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=250, min_samples_leaf=80,
                                         l2_regularization=1.0, random_state=0)


def evaluate(d: pd.DataFrame, scheme: str):
    """scheme: 'race' (GroupKFold por regata) o 'event' (se deja afuera un evento entero)."""
    y = d[TARGET].values
    lo, hi = np.quantile(y, [0.01, 0.99])
    tau = 0.5 * float(np.std(y))
    groups = d.race_id.values if scheme == 'race' else d.event.values
    splitter = GroupKFold(n_splits=5) if scheme == 'race' else LeaveOneGroupOut()
    variants = {
        'B0 constante (media de entrenamiento)': None,
        'B1 persistencia (ganancia últimos 60 s)': ['gain_past60'],
        'M-sinviento (estado + acción)': STATE + ACTION,
        'M-conviento (estado + acción + viento)': STATE + WIND + ACTION,
    }
    oof = {k: np.full(len(d), np.nan) for k in variants}
    for tr, te in splitter.split(d, y, groups):
        ytr = np.clip(y[tr], lo, hi)
        for name, cols in variants.items():
            if cols is None:
                oof[name][te] = ytr.mean()
            elif name.startswith('B1'):
                m = LinearRegression().fit(d.loc[tr, cols], ytr)
                oof[name][te] = m.predict(d.loc[te, cols])
            else:
                m = _gbm().fit(d.loc[tr, cols], ytr)
                oof[name][te] = m.predict(d.loc[te, cols])
    rows = [dict(scheme=scheme, model=k, n=len(d), **_metrics(y, v, tau)) for k, v in oof.items()]
    return pd.DataFrame(rows), oof


def feature_importance(d: pd.DataFrame, n_repeats=3):
    """Importancia por permutación en regatas no vistas (hold-out: 1 de cada 5 regatas)."""
    from sklearn.inspection import permutation_importance
    races = d.race_id.unique()
    te_r = races[::5]
    tr, te = ~d.race_id.isin(te_r), d.race_id.isin(te_r)
    cols = STATE + WIND + ACTION
    y = np.clip(d.loc[tr, TARGET], *np.quantile(d[TARGET], [0.01, 0.99]))
    m = _gbm().fit(d.loc[tr, cols], y)
    pi = permutation_importance(m, d.loc[te, cols], d.loc[te, TARGET], n_repeats=n_repeats, random_state=0,
                                scoring='neg_mean_absolute_error')
    return pd.DataFrame({'feature': cols, 'mae_increase': pi.importances_mean}).sort_values('mae_increase', ascending=False)
