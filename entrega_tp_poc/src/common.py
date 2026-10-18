"""Utilidades geométricas compartidas."""
import numpy as np

KN = 1.852            # km/h por nudo
EARTH_M = 6371000.0   # radio terrestre (m)
MS_PER_KN = 0.514444


def angdiff(a, b):
    """a - b envuelto a [-180, 180)."""
    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0


def circmean(deg, w=None):
    """Media circular en grados [0, 360)."""
    r = np.deg2rad(np.asarray(deg, dtype=float))
    w = np.ones_like(r) if w is None else np.asarray(w, dtype=float)
    return float(np.rad2deg(np.arctan2((w * np.sin(r)).sum(), (w * np.cos(r)).sum())) % 360.0)


def bearing(dx, dy):
    """Rumbo (0=N, 90=E) de un vector (dx=este, dy=norte)."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def to_xy(lat, lon, lat0, lon0):
    k = np.pi / 180.0 * EARTH_M
    return (np.asarray(lon) - lon0) * np.cos(np.radians(lat0)) * k, (np.asarray(lat) - lat0) * k
