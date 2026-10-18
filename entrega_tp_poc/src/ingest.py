"""ETAPA 1 - Ingesta, limpieza y normalización.

Entrada : carpeta raw TracTrac (telemetry.csv.gz, routes.csv, control_passings.csv, ...)
Salida  : dict race_id -> Race (tabla limpia en metros locales + recorrido + piernas)
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .common import KN, MS_PER_KN, EARTH_M, to_xy, bearing, angdiff

MAX_SPEED_KN = 15.0      # un Optimist no pasa de ~12 kn (planeo); arriba es error de sensor
MAX_JUMP_KN = 20.0       # velocidad implícita entre dos posiciones consecutivas
MIN_ROWS_BOAT = 200      # barcos con muy poca telemetría se descartan
UP_TOL, DOWN_TOL = 40.0, 140.0   # tipo de pierna según ángulo entre su rumbo y el eje de viento inicial


@dataclass
class Race:
    race_id: str
    event: str
    name: str
    df: pd.DataFrame            # telemetría limpia
    marks: np.ndarray           # (K,2) xy de cada control de la ruta, en orden
    mark_names: list
    leg_bearing: np.ndarray     # (K-1,)
    leg_len: np.ndarray         # (K-1,)
    leg_type: list              # 'U' upwind, 'D' downwind, 'R' través
    tail: np.ndarray            # distancia restante desde el inicio de cada pierna hasta la meta
    W0: float                   # eje de viento inicial = rumbo salida -> primera marca
    lat0: float
    lon0: float
    K: int


def clean_telemetry(tel: pd.DataFrame, report: dict) -> pd.DataFrame:
    report['rows_raw'] = int(len(tel))
    tel = tel.drop_duplicates(['race_id', 'competitor_id', 'timestamp_unix_ms'])
    report['dup_removed'] = report['rows_raw'] - len(tel)
    n = len(tel)
    tel = tel.dropna(subset=['latitude', 'longitude', 'speed_kmh', 'course_over_ground_deg'])
    report['nan_removed'] = n - len(tel)

    tel = tel.assign(speed_kn=tel.speed_kmh / KN, cog=tel.course_over_ground_deg % 360.0)  # rumbo con signo -> [0,360)
    n = len(tel)
    tel = tel[tel.speed_kn <= MAX_SPEED_KN]
    report['speed_outliers_removed'] = n - len(tel)

    g = tel.groupby('race_id')
    lat0, lon0 = g.latitude.transform('median'), g.longitude.transform('median')
    tel['x'], tel['y'] = to_xy(tel.latitude, tel.longitude, lat0, lon0)
    tel['lat0'], tel['lon0'] = lat0, lon0
    tel = tel.sort_values(['race_id', 'competitor_id', 'timestamp_unix_ms']).reset_index(drop=True)

    removed_jump = 0
    for _ in range(3):   # saltos de GPS: velocidad implícita imposible (se repite porque al borrar cambian los vecinos)
        key = [tel.race_id, tel.competitor_id]
        dt = tel.groupby(key).timestamp_unix_ms.diff() / 1000.0
        dist = np.hypot(tel.groupby(key).x.diff(), tel.groupby(key).y.diff())
        bad = (dt > 0) & (dist / dt / MS_PER_KN > MAX_JUMP_KN)
        if not bad.any():
            break
        removed_jump += int(bad.sum())
        tel = tel[~bad].reset_index(drop=True)
    report['gps_jumps_removed'] = removed_jump

    cnt = tel.groupby(['race_id', 'competitor_id']).timestamp_unix_ms.transform('size')
    report['boats_dropped_short'] = int(tel[cnt < MIN_ROWS_BOAT].groupby(['race_id', 'competitor_id']).ngroups)
    tel = tel[cnt >= MIN_ROWS_BOAT].reset_index(drop=True)
    report['rows_clean'] = int(len(tel))
    return tel


def _smooth_heading(df: pd.DataFrame) -> pd.DataFrame:
    """Rumbo suavizado (media circular, 5 muestras = 15 s) y bandera 'stable' (no está virando)."""
    s, c = np.sin(np.deg2rad(df.cog)), np.cos(np.deg2rad(df.cog))
    key = df.competitor_id
    ss = s.groupby(key).transform(lambda v: v.rolling(5, center=True, min_periods=3).mean())
    cc = c.groupby(key).transform(lambda v: v.rolling(5, center=True, min_periods=3).mean())
    df['hdg'] = np.rad2deg(np.arctan2(ss, cc)) % 360.0
    df['stable'] = np.abs(angdiff(df.cog, df.hdg)) < 25.0
    return df


def build_races(data_dir, report: dict) -> dict:
    d = Path(data_dir)
    tel = pd.read_csv(d / 'telemetry.csv.gz', usecols=[
        'race_id', 'competitor_id', 'timestamp_unix_ms', 'latitude', 'longitude',
        'speed_kmh', 'course_over_ground_deg'])
    tel = clean_telemetry(tel, report)
    tel = _smooth_heading(tel)

    races = pd.read_csv(d / 'races.csv').set_index('race_id')
    routes = pd.read_csv(d / 'routes.csv')
    pas = pd.read_csv(d / 'control_passings.csv')
    cpos = pd.read_csv(d / 'control_positions.csv.gz',
                       usecols=['race_id', 'positioned_item_id', 'control_id', 'latitude', 'longitude'])
    imed = cpos.groupby(['race_id', 'positioned_item_id'])[['latitude', 'longitude']].median()
    cmed = cpos.groupby(['race_id', 'control_id'])[['latitude', 'longitude']].median()

    out = {}
    for rid, df in tel.groupby('race_id'):
        r = routes[routes.race_id == rid].sort_values('sequence')
        if len(r) < 2:
            continue
        lat0, lon0 = df.lat0.iloc[0], df.lon0.iloc[0]
        mk = []
        for _, row in r.iterrows():
            items = [i.strip() for i in str(row.positioned_item_ids).split('|')]
            pts = [imed.loc[(rid, i)].values for i in items if (rid, i) in imed.index]   # fuente principal: ítems del control
            if not pts and (rid, row.control_id) in cmed.index:
                pts = [cmed.loc[(rid, row.control_id)].values]                             # respaldo: por control_id
            mk.append(to_xy(*np.mean(pts, axis=0), lat0, lon0) if pts else (np.nan, np.nan))
        mk = np.array(mk).reshape(-1, 2)
        if np.isnan(mk).any():
            report.setdefault('races_skipped_no_marks', []).append(rid)
            continue
        K = len(r)
        P = pas[pas.race_id == rid].pivot_table(index='competitor_id', columns='sequence',
                                                values='timestamp_unix_ms', aggfunc='first')
        P = P.reindex(columns=range(K))
        t_start = float(P[0].median())      # hora de largada = paso mediano por la línea de salida

        df = df.copy()
        df['t'] = (df.timestamp_unix_ms - t_start) / 1000.0        # segundos desde la largada
        leg = np.full(len(df), -1, dtype=int)                      # -1 = pre-largada, K-1 = terminó
        comp_codes = df.competitor_id.values
        ts = df.timestamp_unix_ms.values
        for comp in np.unique(comp_codes):
            if comp not in P.index:
                continue
            m = comp_codes == comp
            L = np.full(m.sum(), -1)
            for k in range(K):
                pk = P.loc[comp, k]
                if not np.isnan(pk):
                    L = np.maximum(L, np.where(ts[m] >= pk, k, -1))
            leg[m] = L
        df['leg'] = leg
        df = df[df.competitor_id.isin(P.index)]

        v = np.diff(mk, axis=0)
        lb, ll = bearing(v[:, 0], v[:, 1]), np.hypot(v[:, 0], v[:, 1])
        if ll[0] < 250 or ll.min() < 60:          # sanidad de recorrido: piernas absurdamente cortas => marcas mal definidas
            report.setdefault('races_skipped_bad_course', []).append(f"{races.loc[rid].event_name} / {races.loc[rid].race_name}")
            continue
        W0 = float(lb[0])
        lt = ['U' if abs(angdiff(b, W0)) <= UP_TOL else 'D' if abs(angdiff(b, W0)) >= DOWN_TOL else 'R' for b in lb]
        tail = np.concatenate([np.cumsum(ll[::-1])[::-1], [0.0]])    # tail[k] = largo de piernas k..K-2
        meta = races.loc[rid]
        out[rid] = Race(rid, str(meta.event_name), str(meta.race_name), df.reset_index(drop=True), mk,
                        list(r.control_name.fillna('').str.strip()), lb, ll, lt, tail, W0, lat0, lon0, K)
    report['races'] = len(out)
    return out
