#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster-first, route-second (trial) using Google Distance Matrix
+ Cluster-by-cluster full From-To export (distance_km + duration_min)
+ Excel OR CSV input (CSV'de kolon adları TR/EN olabilir)
+ Default dosya/depot + API key'i otomatik bulma
+ ALL günleri çalıştırma (day=ALL)
+ Route görselleştirme (cluster bazlı png + opsiyonel folium html)

DEFAULTLAR (senin veri setine göre)
- Varsayılan CSV adı: data_cleaned.csv (bulunursa otomatik okur)
- Varsayılan depot: Latitude=40.920153, Longitude=29.348245
- API key otomatik aranır:
  1) Environment variable: GOOGLE_MAPS_API_KEY
  2) .env dosyası (cwd veya script klasörü)
  3) google_maps_api_key.txt / gmaps_key.txt / apikey.txt / api_key.txt
  4) config.json içinde {"GOOGLE_MAPS_API_KEY": "..."}

EN kısa komut (Mon örneği):
    python cluster_route_google.py --day Mon --use_traffic --export_cluster_ft --plot_routes

Hepsi için (day=ALL, nrows=-1 yani limit yok):
    python cluster_route_google.py --day ALL --nrows -1 --use_traffic --export_cluster_ft --plot_routes

Kurulum:
    pip install pandas numpy openpyxl scikit-learn matplotlib googlemaps
Opsiyonel harita:
    pip install folium
"""

import os
import math
import time
import json
import argparse
import threading
import datetime as dt
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import googlemaps


DEFAULT_CSV_NAME = "data_cleaned.csv"
DEFAULT_DEPOT_LAT = 40.920153
DEFAULT_DEPOT_LON = 29.348245
API_ENV_VAR = "GOOGLE_MAPS_API_KEY"


# ------------------------- Progress indicator -------------------------

class ProgressTicker:
    """Prints 'running...' every `interval` seconds until stopped."""
    def __init__(self, interval: int = 10, msg: str = "running..."):
        self.interval = interval
        self.msg = msg
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            print(self.msg, flush=True)

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=1)


# ------------------------- API key auto-find -------------------------

def _read_text_file_first_line(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    return s
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin1") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    return s
    except Exception:
        return None
    return None

def _parse_dotenv(path: str) -> Dict[str, str]:
    env = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin1") as f:
            lines = f.readlines()
    except Exception:
        return env

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        env[k] = v
    return env

def _resolve_candidates(filename: str) -> List[str]:
    """cwd + script dir"""
    candidates = [
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
    ]
    out, seen = [], set()
    for p in candidates:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def get_google_maps_api_key(env_var: str = API_ENV_VAR) -> str:
    # 1) environment
    key = os.getenv(env_var)
    if key and key.strip():
        return key.strip()

    # 2) .env
    for env_path in _resolve_candidates(".env"):
        if os.path.exists(env_path):
            d = _parse_dotenv(env_path)
            if env_var in d and d[env_var].strip():
                return d[env_var].strip()

    # 3) key text files
    for fname in ["google_maps_api_key.txt", "gmaps_key.txt", "apikey.txt", "api_key.txt"]:
        for fp in _resolve_candidates(fname):
            if os.path.exists(fp):
                s = _read_text_file_first_line(fp)
                if s:
                    return s.strip()

    # 4) config.json
    for fp in _resolve_candidates("config.json"):
        if os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if isinstance(cfg, dict) and env_var in cfg and str(cfg[env_var]).strip():
                    return str(cfg[env_var]).strip()
            except Exception:
                pass

    raise SystemExit(
        f"{env_var} bulunamadı.\n"
        "Çözüm seçenekleri:\n"
        "  1) Environment variable olarak ayarla (önerilir):\n"
        "     PowerShell:  $env:GOOGLE_MAPS_API_KEY=\"...\"\n"
        "     Kalıcı yapmak için: setx GOOGLE_MAPS_API_KEY \"...\"  (sonra yeni terminal aç)\n"
        "  2) Aynı klasöre .env koy: GOOGLE_MAPS_API_KEY=...\n"
        "  3) Aynı klasöre google_maps_api_key.txt koy (ilk satır key)\n"
        "  4) config.json içine yaz: {\"GOOGLE_MAPS_API_KEY\": \"...\"}\n"
    )


# ------------------------- Input helpers -------------------------

def _norm(s: str) -> str:
    return str(s).strip().lower()

def read_excel_safely(excel_path: str, target_sheet_hint: str) -> pd.DataFrame:
    """Trim+lower ile sheet eşleştirerek okur (sondaki boşluklar sorun çıkarmaz)."""
    xls = pd.ExcelFile(excel_path)
    hint = _norm(target_sheet_hint)

    sheet = None
    for s in xls.sheet_names:
        if _norm(s) == hint:
            sheet = s
            break
    if sheet is None:
        for s in xls.sheet_names:
            if hint in _norm(s):
                sheet = s
                break
    if sheet is None:
        raise ValueError(f"Sheet not found. Available sheets: {xls.sheet_names}")
    return pd.read_excel(excel_path, sheet_name=sheet)

def read_depot_latlon_from_excel(excel_path: str, depot_sheet_hint: str = "IstAnadolu Aktarma Koordinat") -> Tuple[float, float]:
    depot_raw = read_excel_safely(excel_path, depot_sheet_hint)

    col0 = depot_raw.columns[0]
    col_last = depot_raw.columns[-1]

    def pick(label: str) -> Optional[float]:
        m = depot_raw[depot_raw[col0].astype(str).str.strip().str.lower() == label.lower()]
        if len(m) == 0:
            return None
        return float(m.iloc[0][col_last])

    lat = pick("Latitude")
    lon = pick("Longitude")
    if lat is None or lon is None:
        raise ValueError("Depot Latitude/Longitude not found in depot sheet.")
    return lat, lon

def read_data_csv_auto(
    csv_path: str,
    sep: Optional[str],
    decimal: str,
    encoding: str,
) -> pd.DataFrame:
    """
    encoding="auto" ise: utf-8 dene, olmazsa cp1254/cp1252/latin1 sırayla dener.
    """
    encodings = [encoding] if encoding and encoding.lower() != "auto" else ["utf-8", "cp1254", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            if sep is None:
                return pd.read_csv(csv_path, sep=None, engine="python", decimal=decimal, encoding=enc)
            return pd.read_csv(csv_path, sep=sep, decimal=decimal, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"CSV okunamadı. Denenen encoding'ler: {encodings}. Son hata: {last_err}")

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    TR/EN kolonları tek standarda çeker:
      Durak Günü, Durak No, Latitude, Longitude, Paket Adedi (opsiyonel)

    Sende gelen (data_cleaned.csv):
      StopNum -> Durak No
      Day     -> Durak Günü
      NumOfPackages -> Paket Adedi
    """
    colmap = {}
    for c in df.columns:
        cn = _norm(c)
        if cn in ["durak günü", "durak gunu", "day"]:
            colmap[c] = "Durak Günü"
        elif cn in ["durak no", "durak no.", "stopnum", "stop num", "stop_no", "stop no", "stopnumber", "stop number"]:
            colmap[c] = "Durak No"
        elif cn in ["latitude", "lat", "enlem"]:
            colmap[c] = "Latitude"
        elif cn in ["longitude", "lon", "boylam", "lng"]:
            colmap[c] = "Longitude"
        elif cn in ["paket adedi", "numofpackages", "numberofpackages", "packages", "packagecount", "num packages"]:
            colmap[c] = "Paket Adedi"
    return df.rename(columns=colmap).copy()

def validate_columns(df: pd.DataFrame) -> None:
    required = ["Durak Günü", "Durak No", "Latitude", "Longitude"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolon(lar): {missing}. Gerekli: {required} (+ opsiyonel: Paket Adedi)")

def _resolve_default_path(filename: str) -> Optional[str]:
    """cwd + script dir"""
    for p in _resolve_candidates(filename):
        if os.path.exists(p):
            return p
    return None

def load_input(args) -> Tuple[pd.DataFrame, float, float]:
    """
    Girdi seçimi:
    - Eğer --excel verilmişse onu kullan
    - Yoksa CSV (args.csv veya DEFAULT_CSV_NAME) bulunursa onu kullan
    """
    # Excel explicit
    if args.excel:
        if not os.path.exists(args.excel):
            raise SystemExit(f"Excel bulunamadı: {args.excel}")
        df_raw = read_excel_safely(args.excel, "Data")
        df = standardize_columns(df_raw)
        validate_columns(df)
        depot_lat, depot_lon = read_depot_latlon_from_excel(args.excel)
        return df, depot_lat, depot_lon

    # CSV explicit or default
    csv_path = None
    if args.csv:
        csv_path = args.csv if os.path.exists(args.csv) else _resolve_default_path(os.path.basename(args.csv))
    if not csv_path:
        csv_path = _resolve_default_path(DEFAULT_CSV_NAME)

    if not csv_path:
        raise SystemExit(
            "Girdi bulunamadı.\n"
            f"Çözüm:\n"
            f"  - CSV dosyanı script ile aynı klasöre koy ve adı '{DEFAULT_CSV_NAME}' olsun, veya\n"
            "  - komutta açıkça ver: --csv \"data_cleaned.csv\" , veya\n"
            "  - Excel için: --excel \"data.xlsx\""
        )

    df_raw = read_data_csv_auto(csv_path, sep=args.csv_sep, decimal=args.csv_decimal, encoding=args.csv_encoding)
    df = standardize_columns(df_raw)
    validate_columns(df)

    depot_lat = float(args.depot_lat)
    depot_lon = float(args.depot_lon)
    return df, depot_lat, depot_lon


# ------------------------- Google Distance Matrix (batching + retry) -------------------------

def _fmt_latlon(lat: float, lon: float) -> str:
    return f"{lat:.6f},{lon:.6f}"

def _dm_call_with_retry(
    gmaps_client: googlemaps.Client,
    origins: List[str],
    destinations: List[str],
    mode: str,
    departure_time,
    language: str,
    units: str,
    max_retries: int = 5,
    base_sleep: float = 1.2,
):
    last_err = None
    for attempt in range(max_retries):
        try:
            kwargs = dict(origins=origins, destinations=destinations, mode=mode, units=units, language=language)
            if departure_time is not None:
                kwargs["departure_time"] = departure_time
            return gmaps_client.distance_matrix(**kwargs)
        except Exception as e:
            last_err = e
            time.sleep(base_sleep * (2 ** attempt))
    raise RuntimeError(f"Google DM call failed after retries. Last error: {last_err}")

def google_distance_duration_matrices(
    gmaps_client: googlemaps.Client,
    latlon_deg: np.ndarray,
    mode: str = "driving",
    departure_time=None,
    language: str = "tr",
    units: str = "metric",
    chunk_size: int = 10,
    pause_s: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      dist_m: NxN meters (asymmetric olabilir)
      dur_s:  NxN seconds
      dur_traffic_s: NxN seconds (yoksa NaN)
    """
    n = latlon_deg.shape[0]
    dist_m = np.full((n, n), np.nan, dtype=float)
    dur_s = np.full((n, n), np.nan, dtype=float)
    dur_tr_s = np.full((n, n), np.nan, dtype=float)

    locs = [_fmt_latlon(float(latlon_deg[i, 0]), float(latlon_deg[i, 1])) for i in range(n)]
    if departure_time is None:
        # "now" avoids timezone / system clock issues and enables traffic when requested
        departure_time = "now"

    for i0 in range(0, n, chunk_size):
        origins = locs[i0:i0 + chunk_size]
        for j0 in range(0, n, chunk_size):
            destinations = locs[j0:j0 + chunk_size]

            resp = _dm_call_with_retry(
                gmaps_client,
                origins=origins,
                destinations=destinations,
                mode=mode,
                departure_time=departure_time,
                language=language,
                units=units,
            )

            rows = resp.get("rows", [])
            for oi, row in enumerate(rows):
                elems = row.get("elements", [])
                for dj, e in enumerate(elems):
                    ii = i0 + oi
                    jj = j0 + dj
                    if ii >= n or jj >= n:
                        continue
                    if e.get("status") != "OK":
                        continue
                    if "distance" in e:
                        dist_m[ii, jj] = float(e["distance"]["value"])
                    if "duration" in e:
                        dur_s[ii, jj] = float(e["duration"]["value"])
                    if "duration_in_traffic" in e:
                        dur_tr_s[ii, jj] = float(e["duration_in_traffic"]["value"])

            time.sleep(pause_s)

    return dist_m, dur_s, dur_tr_s


# ------------------------- Routing heuristics (NN + 2-opt) -------------------------

def nn_route(cost: np.ndarray) -> List[int]:
    n = cost.shape[0]
    unvisited = set(range(1, n))
    route = [0]
    cur = 0
    while unvisited:
        def c(j):
            v = cost[cur, j]
            return v if np.isfinite(v) else 1e18
        nxt = min(unvisited, key=c)
        unvisited.remove(nxt)
        route.append(nxt)
        cur = nxt
    route.append(0)
    return route

def route_cost(route: List[int], cost: np.ndarray) -> float:
    total = 0.0
    for i in range(len(route) - 1):
        v = cost[route[i], route[i+1]]
        if not np.isfinite(v):
            return float("inf")
        total += float(v)
    return total

def two_opt(route: List[int], cost: np.ndarray, max_iter: int = 3500) -> Tuple[List[int], float]:
    best = route[:]
    best_len = route_cost(best, cost)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(1, len(best) - 2):
            for k in range(i + 1, len(best) - 1):
                if k - i == 1:
                    continue
                new = best[:i] + best[i:k+1][::-1] + best[k+1:]
                new_len = route_cost(new, cost)
                if new_len + 1e-6 < best_len:
                    best, best_len = new, new_len
                    improved = True
                    break
            if improved:
                break
    return best, best_len


# ------------------------- Cluster + route -------------------------

def choose_k(n_stops: int, target_cluster_size: int = 20, max_k: int = 10) -> int:
    return max(1, min(max_k, math.ceil(n_stops / target_cluster_size)))

def cluster_kmeans(df_day: pd.DataFrame, k: int) -> np.ndarray:
    coords = df_day[["Latitude", "Longitude"]].to_numpy()
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(coords)


def _euclid2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between Nx2 and Kx2 arrays -> NxK (fast, fine for clustering)."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=2)

def cluster_kmeans_balanced(
    df_day: pd.DataFrame,
    k: int,
    min_size: int,
    max_size: int,
) -> np.ndarray:
    """
    Capacity-constrained clustering (heuristic):
      1) Run KMeans to get centroids
      2) Assign points to clusters with capacity [min_size, max_size]
         - Fill each cluster up to min_size (nearest-first)
         - Assign remaining to nearest cluster with remaining capacity

    Guarantees:
      - If k*min_size <= n <= k*max_size, this aims to satisfy bounds.
      - For pathological geometry, it can still struggle; we raise a clear error if it can't.
    """
    coords = df_day[["Latitude", "Longitude"]].to_numpy(dtype=float)
    n = coords.shape[0]
    if k * min_size > n:
        raise ValueError(f"Impossible: k*min_size={k*min_size} > n={n}. Increase n or reduce k/min_size.")
    if k * max_size < n:
        raise ValueError(f"Impossible: k*max_size={k*max_size} < n={n}. Increase k/max_size.")

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(coords)
    centers = km.cluster_centers_

    d2 = _euclid2(coords, centers)  # NxK
    # for each cluster, points sorted by closeness
    order = np.argsort(d2, axis=0)  # NxK (column j is indices sorted for cluster j)

    labels = np.full(n, -1, dtype=int)
    unassigned = np.ones(n, dtype=bool)
    filled = np.zeros(k, dtype=int)
    cap = np.full(k, max_size, dtype=int)
    ptr = np.zeros(k, dtype=int)

    # Phase A: ensure minimum (round-robin)
    # iterate until all clusters reach min_size
    changed = True
    while changed and np.any(filled < min_size):
        changed = False
        for j in range(k):
            while filled[j] < min_size:
                # advance pointer to next unassigned candidate
                while ptr[j] < n and (not unassigned[order[ptr[j], j]]):
                    ptr[j] += 1
                if ptr[j] >= n:
                    # no candidates left for this cluster
                    raise RuntimeError("Min-fill failed: ran out of candidates. Try different k or seed.")
                i = int(order[ptr[j], j])
                labels[i] = j
                unassigned[i] = False
                filled[j] += 1
                cap[j] -= 1
                ptr[j] += 1
                changed = True
                if cap[j] < 0:
                    raise RuntimeError("Capacity error during min-fill. Check max_size >= min_size.")

    # Phase B: assign remaining to nearest cluster with capacity
    remaining_idx = np.where(unassigned)[0]
    if remaining_idx.size > 0:
        # process points in order of "how clear" their best choice is (optional but helps)
        best = np.min(d2[remaining_idx, :], axis=1)
        remaining_idx = remaining_idx[np.argsort(best)]

        for i in remaining_idx:
            # clusters with remaining cap
            valid = np.where(cap > 0)[0]
            if valid.size == 0:
                raise RuntimeError("No capacity left to assign remaining points. Increase k or max_size.")
            # choose nearest valid cluster
            j = int(valid[np.argmin(d2[i, valid])])
            labels[i] = j
            cap[j] -= 1
            filled[j] += 1

    # Final check
    sizes = np.bincount(labels, minlength=k)
    if np.any(sizes < min_size) or np.any(sizes > max_size):
        raise RuntimeError(
            "Balanced clustering produced out-of-bound cluster sizes. "
            f"min={min_size}, max={max_size}, sizes={sizes.tolist()}. "
            "Try changing k, target_cluster_size, or increase max_k."
        )
    return labels

def build_tables_from_matrices(
    labels: List[str],
    dist_m: np.ndarray,
    dur_s: np.ndarray,
    dur_tr_s: np.ndarray,
    use_traffic: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dist_km = pd.DataFrame(dist_m / 1000.0, index=labels, columns=labels)
    dur_use = np.where(np.isnan(dur_tr_s), dur_s, dur_tr_s) if use_traffic else dur_s
    dur_min = pd.DataFrame(dur_use / 60.0, index=labels, columns=labels)
    dur_tr_min = pd.DataFrame(dur_tr_s / 60.0, index=labels, columns=labels)
    return dist_km, dur_min, dur_tr_min

def solve_cluster_route_google(
    cdf: pd.DataFrame,
    depot_lat: float,
    depot_lon: float,
    gmaps_client: googlemaps.Client,
    mode: str = "driving",
    use_traffic: bool = True,
    chunk_size: int = 10,
    pause_s: float = 0.15,
) -> Dict[str, Any]:
    pts = np.vstack([[depot_lat, depot_lon], cdf[["Latitude", "Longitude"]].to_numpy()])
    dist_m, dur_s, dur_tr_s = google_distance_duration_matrices(
        gmaps_client,
        pts,
        mode=mode,
        departure_time=None,
        chunk_size=chunk_size,
        pause_s=pause_s,
    )

    # route optimization cost = time (traffic tercih)
    cost_time = np.where(np.isnan(dur_tr_s), dur_s, dur_tr_s) if use_traffic else dur_s
    r0 = nn_route(cost_time)
    r, _ = two_opt(r0, cost_time)

    total_m = route_cost(r, dist_m)
    total_s = route_cost(r, cost_time)

    labels = ["DEPOT"] + [str(x) for x in cdf["Durak No"].tolist()]
    return {
        "route": r,
        "total_km": float(total_m / 1000.0),
        "total_min": float(total_s / 60.0),
        "dist_m": dist_m,
        "dur_s": dur_s,
        "dur_tr_s": dur_tr_s,
        "labels": labels,
        "pts": pts,  # depot + stops latlon
    }


def _find_col_case_insensitive(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_norm = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in cols_norm:
            return cols_norm[key]
    return None

def get_demands(
    df_part: pd.DataFrame,
    delivery_col: Optional[str] = None,
    pickup_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (delivery, pickup) arrays for rows in df_part.
    If no delivery column is found, tries 'Paket Adedi' as delivery.
    If no pickup column is found, assumes 0.
    """
    dcol = delivery_col
    pcol = pickup_col

    if dcol is None or dcol not in df_part.columns:
        dcol = _find_col_case_insensitive(df_part, ["delivery", "deliveries", "teslimat", "del", "delivery_qty", "deliverycount"])
        if dcol is None and "Paket Adedi" in df_part.columns:
            dcol = "Paket Adedi"
    if pcol is None or pcol not in df_part.columns:
        pcol = _find_col_case_insensitive(df_part, ["pickup", "pickups", "alim", "alım", "pickup_qty", "pickupcount"])

    delivery = df_part[dcol].to_numpy(dtype=float) if dcol is not None else np.zeros(len(df_part), dtype=float)
    pickup = df_part[pcol].to_numpy(dtype=float) if pcol is not None else np.zeros(len(df_part), dtype=float)

    # NaN -> 0
    delivery = np.nan_to_num(delivery, nan=0.0)
    pickup = np.nan_to_num(pickup, nan=0.0)

    # Negative values not expected
    if np.any(delivery < 0) or np.any(pickup < 0):
        raise ValueError("Delivery/Pickup values must be >= 0.")
    return delivery, pickup

def _dm_one_to_many(
    gmaps_client: googlemaps.Client,
    origin_lat: float,
    origin_lon: float,
    dest_latlon: np.ndarray,  # Mx2
    mode: str,
    language: str,
    units: str,
    use_traffic: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (dist_m, dur_s_used) arrays length M.
    Uses departure_time="now" for traffic when enabled.
    """
    origin = [_fmt_latlon(origin_lat, origin_lon)]
    dests = [_fmt_latlon(float(lat), float(lon)) for lat, lon in dest_latlon]

    # Google limit: origins*destinations <= 100 elements per request.
    # Here origins=1, so M must be <=100.
    if len(dests) > 100:
        raise ValueError("Too many destinations in one DM request (max 100). Lower candidate_limit.")

    resp = _dm_call_with_retry(
        gmaps_client,
        origins=origin,
        destinations=dests,
        mode=mode,
        departure_time="now" if use_traffic else None,
        language=language,
        units=units,
    )

    elems = resp["rows"][0]["elements"]
    dist = np.full(len(dests), np.nan, dtype=float)
    dur = np.full(len(dests), np.nan, dtype=float)

    for j, e in enumerate(elems):
        if e.get("status") != "OK":
            continue
        if "distance" in e:
            dist[j] = float(e["distance"]["value"])
        if use_traffic and "duration_in_traffic" in e:
            dur[j] = float(e["duration_in_traffic"]["value"])
        elif "duration" in e:
            dur[j] = float(e["duration"]["value"])

    return dist, dur

def solve_cluster_route_edge_greedy_google(
    cdf: pd.DataFrame,
    depot_lat: float,
    depot_lon: float,
    gmaps_client: googlemaps.Client,
    vehicle_cap: float,
    delivery_col: Optional[str],
    pickup_col: Optional[str],
    mode: str = "driving",
    use_traffic: bool = True,
    candidate_limit: int = 60,
    alpha_load: float = 0.0,
) -> Dict[str, Any]:
    """
    Greedy route builder with dynamic load feasibility.
    - No full N×N matrix (fast, fewer API calls).
    - At each step, pick the feasible next stop with minimum (time + alpha*load_penalty)

    Load model:
      load_after = load_before - delivery_i + pickup_i
      constraints: 0 <= load_after <= vehicle_cap

    Start load:
      load0 = sum(deliveries of assigned stops)
      (typical for VRP delivery-from-depot; if your model differs, adjust here.)
    """
    n = len(cdf)
    coords = cdf[["Latitude", "Longitude"]].to_numpy(dtype=float)
    delivery, pickup = get_demands(cdf, delivery_col=delivery_col, pickup_col=pickup_col)

    load = float(np.sum(delivery))
    if load > vehicle_cap + 1e-9:
        raise RuntimeError(f"Cluster infeasible at start: total_delivery={load:.2f} > vehicle_cap={vehicle_cap}. Increase k or rebalance.")

    target = vehicle_cap / 2.0

    remaining = list(range(n))  # indices in cdf
    route_nodes = [0]  # 0=DEPOT, then 1..n mapped as idx+1
    legs = []  # per move: (from, to, dist_m, dur_s, load_before, load_after)

    cur_lat, cur_lon = depot_lat, depot_lon

    while remaining:
        # Feasible set by load constraint
        feas = []
        for idx in remaining:
            load_after = load - float(delivery[idx]) + float(pickup[idx])
            if 0.0 <= load_after <= vehicle_cap + 1e-9:
                feas.append(idx)

        if not feas:
            raise RuntimeError("No feasible next stop found under load constraints. Consider different clustering or allow depot reload/unload logic.")

        # Reduce candidates (nearest in Euclidean lat/lon) to limit API
        feas_coords = coords[feas]
        # squared distance in lat/lon space (ok for pruning)
        d2 = (feas_coords[:,0] - cur_lat)**2 + (feas_coords[:,1] - cur_lon)**2
        order = np.argsort(d2)
        cand = [feas[i] for i in order[:min(candidate_limit, len(feas))]]

        dest_latlon = coords[cand]
        dist_m, dur_s = _dm_one_to_many(
            gmaps_client,
            origin_lat=cur_lat,
            origin_lon=cur_lon,
            dest_latlon=dest_latlon,
            mode=mode,
            language="tr",
            units="metric",
            use_traffic=use_traffic,
        )

        # Choose best candidate
        best_idx = None
        best_obj = float("inf")
        best_dist = None
        best_dur = None

        for j, idx in enumerate(cand):
            if not np.isfinite(dur_s[j]):
                continue
            load_after = load - float(delivery[idx]) + float(pickup[idx])
            obj = float(dur_s[j])
            if alpha_load > 0:
                obj += alpha_load * abs(load_after - target)
            if obj < best_obj:
                best_obj = obj
                best_idx = idx
                best_dist = float(dist_m[j]) if np.isfinite(dist_m[j]) else float("nan")
                best_dur = float(dur_s[j])

        if best_idx is None:
            raise RuntimeError("All DM times were invalid for candidates. Check coordinates or API limits.")

        load_before = load
        load_after = load_before - float(delivery[best_idx]) + float(pickup[best_idx])

        legs.append(("CUR", int(best_idx), best_dist, best_dur, load_before, load_after))
        load = load_after

        # move
        cur_lat, cur_lon = float(coords[best_idx,0]), float(coords[best_idx,1])
        route_nodes.append(int(best_idx) + 1)  # map to 1..n
        remaining.remove(best_idx)

    # return to depot
    dist_back, dur_back = _dm_one_to_many(
        gmaps_client,
        origin_lat=cur_lat,
        origin_lon=cur_lon,
        dest_latlon=np.array([[depot_lat, depot_lon]], dtype=float),
        mode=mode,
        language="tr",
        units="metric",
        use_traffic=use_traffic,
    )
    legs.append(("LAST", "DEPOT", float(dist_back[0]), float(dur_back[0]), load, load))  # load unchanged at depot
    route_nodes.append(0)

    total_km = float(np.nansum([x[2] for x in legs]) / 1000.0)
    total_min = float(np.nansum([x[3] for x in legs]) / 60.0)

    labels = ["DEPOT"] + [str(x) for x in cdf["Durak No"].tolist()]
    return {
        "route": route_nodes,
        "total_km": total_km,
        "total_min": total_min,
        "labels": labels,
        "delivery": delivery,
        "pickup": pickup,
        "legs": legs,
    }

def top_k_pairs(mat: pd.DataFrame, k: int = 7, smallest: bool = True) -> pd.DataFrame:
    labels = mat.index.tolist()
    rows = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            v = float(mat.iat[i, j]) if np.isfinite(mat.iat[i, j]) else np.nan
            rows.append((labels[i], labels[j], v))
    dfp = pd.DataFrame(rows, columns=["From", "To", "Value"]).dropna()
    dfp = dfp.sort_values("Value", ascending=smallest).head(k).reset_index(drop=True)
    return dfp


# ------------------------- Visualizations -------------------------

def plot_clusters_points(df_day: pd.DataFrame, depot_lat: float, depot_lon: float, out_png: str):
    plt.figure()
    for cid, sub in df_day.groupby("cluster_id"):
        plt.scatter(sub["Longitude"], sub["Latitude"], s=15, label=f"cluster {cid}")
    plt.scatter([depot_lon], [depot_lat], marker="x", s=80, label="DEPOT")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc="best")
    plt.title("Clusters")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_route_for_cluster(
    cdf: pd.DataFrame,
    depot_lat: float,
    depot_lon: float,
    route: List[int],
    out_png: str,
    title: str,
):
    """
    route: [0, ..., 0] indexler (0=DEPOT, 1..n stops)
    """
    pts = np.vstack([[depot_lat, depot_lon], cdf[["Latitude", "Longitude"]].to_numpy()])
    lons = pts[:, 1]
    lats = pts[:, 0]

    xs = [lons[i] for i in route]
    ys = [lats[i] for i in route]

    plt.figure()
    plt.scatter(lons[1:], lats[1:], s=22)
    plt.scatter([lons[0]], [lats[0]], marker="x", s=90)

    plt.plot(xs, ys, linewidth=1.6)  # polyline

    # annotate sequence number for stops (skip depot occurrences)
    seq_no = 1
    for idx in route:
        if idx == 0:
            continue
        plt.text(lons[idx], lats[idx], str(seq_no), fontsize=8)
        seq_no += 1

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def try_export_folium_map(
    df_day: pd.DataFrame,
    depot_lat: float,
    depot_lon: float,
    routes_by_cluster: Dict[int, Tuple[pd.DataFrame, List[int]]],
    out_html: str,
):
    """
    Folium yüklüyse tek bir HTML haritası üretir:
    - cluster renkli marker (folium default)
    - her cluster route polyline
    """
    try:
        import folium  # type: ignore
    except Exception:
        print("folium kurulu değil -> HTML harita üretimi atlandı. (pip install folium)", flush=True)
        return

    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=10, control_scale=True)

    folium.Marker(
        [depot_lat, depot_lon],
        tooltip="DEPOT",
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
    ).add_to(m)

    # Markers
    for _, row in df_day.iterrows():
        cid = int(row["cluster_id"])
        folium.CircleMarker(
            location=[float(row["Latitude"]), float(row["Longitude"])],
            radius=4,
            tooltip=f"Durak {int(row['Durak No'])} | cluster {cid}",
            fill=True,
        ).add_to(m)

    # Routes
    for cid, (cdf, route) in routes_by_cluster.items():
        pts = np.vstack([[depot_lat, depot_lon], cdf[["Latitude", "Longitude"]].to_numpy()])
        poly = [[float(pts[i,0]), float(pts[i,1])] for i in route]
        folium.PolyLine(poly, weight=4, opacity=0.8, tooltip=f"cluster {cid}").add_to(m)

    m.save(out_html)


# ------------------------- Main workflow -------------------------

def process_one_day(
    day_label: str,
    df: pd.DataFrame,
    depot_lat: float,
    depot_lon: float,
    args,
    gmaps_client: googlemaps.Client,
    outdir: str,
) -> None:
    # Filter day
    df_day = df[df["Durak Günü"].astype(str).str.strip().str.lower() == day_label.lower()].copy()
    if len(df_day) == 0:
        print(f"[SKIP] day={day_label} için satır yok.", flush=True)
        return

    # Apply nrows limit if asked (nrows <= 0 => no limit)
    if args.nrows is not None and args.nrows > 0:
        df_day = df_day.head(args.nrows).copy()

    # Cluster
    n = len(df_day)
    # Feasible K range from stop constraints
    k_min = math.ceil(n / max(1, args.max_stops))
    k_max = max(1, math.floor(n / max(1, args.min_stops)))
    if k_min > k_max:
        raise SystemExit(f"Infeasible constraints: n={n}, min_stops={args.min_stops}, max_stops={args.max_stops} -> k_min={k_min} > k_max={k_max}")

    if args.k is not None:
        k = int(args.k)
        if not (k_min <= k <= k_max):
            print(f"[WARN] manual k={k} violates constraints -> feasible range [{k_min}, {k_max}]. Will still run but clusters may violate min/max.", flush=True)
        k_mode = "manual"
    else:
        # auto choose close to target_cluster_size but within feasible range and max_k
        k_guess = choose_k(n, target_cluster_size=args.target_cluster_size, max_k=10**9)
        k = int(min(max(k_guess, k_min), min(k_max, args.max_k)))
        # if user left max_k too small, lift it to feasible minimum
        if args.max_k < k_min:
            print(f"[WARN] max_k={args.max_k} < k_min={k_min}. Using k={k_min} to satisfy max_stops.", flush=True)
            k = int(k_min)
        k_mode = "auto"

    print(f"[INFO] day={day_label} | n_stops={n} | feasible_k=[{k_min},{k_max}] | k_used={k} ({k_mode})", flush=True)
    # Cluster assignment
    if args.balanced_clusters:
        df_day["cluster_id"] = cluster_kmeans_balanced(df_day, k, args.min_stops, args.max_stops)
    else:
        df_day["cluster_id"] = cluster_kmeans(df_day, k)
    # Quick cluster size summary
    sizes = df_day["cluster_id"].value_counts().sort_index()
    sizes_str = ", ".join([f"{int(cid)}:{int(cnt)}" for cid, cnt in sizes.items()])
    print(f"[INFO] day={day_label} | cluster_sizes (cluster:count) -> {sizes_str}", flush=True)
    clustered_path = os.path.join(outdir, f"clustered_stops_{day_label}_n{len(df_day)}.csv")
    df_day.to_csv(clustered_path, index=False)

    routes_rows = []
    summary_rows = []
    routes_by_cluster = {}  # for folium

    for cid in sorted(df_day["cluster_id"].unique()):
        cdf = df_day[df_day["cluster_id"] == cid].copy().reset_index(drop=True)
        # Progress / rough API-call estimate (Distance Matrix request count)
        n_c = len(cdf) + 1  # + depot
        blocks = math.ceil(n_c / max(1, args.chunk_size))
        est_calls = blocks * blocks
        print(f"[INFO] day={day_label} | processing cluster={int(cid)} | n_points(incl depot)={n_c} | approx_DM_calls={est_calls}", flush=True)

        if args.route_strategy == "full_matrix" or args.export_cluster_ft:
            # full N×N matrices (expensive) required for from-to exports
            sol = solve_cluster_route_google(
                cdf, depot_lat, depot_lon, gmaps_client,
                mode=args.mode, use_traffic=args.use_traffic,
                chunk_size=args.chunk_size, pause_s=args.pause_s
            )
        else:
            # fast greedy routing with dynamic load, no full matrix
            sol = solve_cluster_route_edge_greedy_google(
                cdf, depot_lat, depot_lon, gmaps_client,
                vehicle_cap=args.vehicle_cap,
                delivery_col=args.delivery_col,
                pickup_col=args.pickup_col,
                mode=args.mode,
                use_traffic=args.use_traffic,
                candidate_limit=min(100, max(1, args.candidate_limit)),
                alpha_load=args.alpha_load,
            )
        route = sol["route"]
        routes_by_cluster[int(cid)] = (cdf, route)

        # route rows
        for seq, node in enumerate(route):
            if node == 0:
                routes_rows.append({
                    "day": day_label, "cluster_id": int(cid), "seq": int(seq),
                    "Durak No": "DEPOT", "Longitude": depot_lon, "Latitude": depot_lat,
                    "Paket Adedi": np.nan,
                    "Delivery": np.nan,
                    "Pickup": np.nan,
                })
            else:
                row = cdf.iloc[node - 1]
                routes_rows.append({
                    "day": day_label, "cluster_id": int(cid), "seq": int(seq),
                    "Durak No": int(row["Durak No"]),
                    "Longitude": float(row["Longitude"]),
                    "Latitude": float(row["Latitude"]),
                    "Paket Adedi": float(row["Paket Adedi"]) if ("Paket Adedi" in cdf.columns and pd.notna(row.get("Paket Adedi"))) else np.nan,
                    "Delivery": float(sol["delivery"][node-1]) if ("delivery" in sol) else np.nan,
                    "Pickup": float(sol["pickup"][node-1]) if ("pickup" in sol) else np.nan
                })

        summary_rows.append({
            "day": day_label,
            "cluster_id": int(cid),
            "n_stops": int(len(cdf)),
            "total_km": float(sol["total_km"]),
            "total_min": float(sol["total_min"]),
            "mode": args.mode,
            "use_traffic": bool(args.use_traffic),
            "vehicle_cap": float(args.vehicle_cap),
            "route_strategy": str(args.route_strategy),
        })

        # cluster-by-cluster FULL from-to export (reuse matrices)
        if args.export_cluster_ft:
            dist_km_df, dur_min_df, dur_tr_min_df = build_tables_from_matrices(
                sol["labels"], sol["dist_m"], sol["dur_s"], sol["dur_tr_s"], use_traffic=args.use_traffic
            )
            ft_cluster_xlsx = os.path.join(outdir, f"from_to_google_{day_label}_cluster{cid}_n{len(cdf)}.xlsx")
            with pd.ExcelWriter(ft_cluster_xlsx, engine="openpyxl") as w:
                dist_km_df.to_excel(w, sheet_name="distance_km")
                dur_min_df.to_excel(w, sheet_name="duration_min")
                dur_tr_min_df.to_excel(w, sheet_name="duration_in_traffic_min")

            top_dist = top_k_pairs(dist_km_df, k=args.topk, smallest=True).rename(columns={"Value": "Distance_km"})
            top_dur = top_k_pairs(dur_min_df, k=args.topk, smallest=True).rename(columns={"Value": "Duration_min"})
            pd.merge(top_dist, top_dur, on=["From", "To"], how="outer").to_csv(
                os.path.join(outdir, f"top{args.topk}_pairs_{day_label}_cluster{cid}_n{len(cdf)}.csv"), index=False
            )

        # route plot per cluster
        if args.plot_routes:
            out_png = os.path.join(outdir, f"route_{day_label}_cluster{cid}_n{len(cdf)}.png")
            plot_route_for_cluster(
                cdf, depot_lat, depot_lon, route,
                out_png=out_png,
                title=f"Route | day={day_label} | cluster={cid} | n={len(cdf)}"
            )

    # Save routes + summary
    routes_df = pd.DataFrame(routes_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(["day", "cluster_id"])

    routes_path = os.path.join(outdir, f"routes_{day_label}_n{len(df_day)}.csv")
    summary_path = os.path.join(outdir, f"routes_summary_{day_label}_n{len(df_day)}.csv")
    routes_df.to_csv(routes_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # points plot
    points_plot = os.path.join(outdir, f"clusters_{day_label}_n{len(df_day)}.png")
    plot_clusters_points(df_day, depot_lat, depot_lon, points_plot)

    # Sample from-to export (small sanity check)
    sample_df = df_day.head(min(args.sample_n, len(df_day))).copy()
    pts = np.vstack([[depot_lat, depot_lon], sample_df[["Latitude", "Longitude"]].to_numpy()])
    dist_m, dur_s, dur_tr_s = google_distance_duration_matrices(
        gmaps_client, pts, mode=args.mode, departure_time=None,
        chunk_size=args.chunk_size, pause_s=args.pause_s
    )
    labels = ["DEPOT"] + [str(x) for x in sample_df["Durak No"].tolist()]
    dist_km_df, dur_min_df, dur_tr_min_df = build_tables_from_matrices(labels, dist_m, dur_s, dur_tr_s, use_traffic=args.use_traffic)

    ft_xlsx = os.path.join(outdir, f"from_to_google_{day_label}_sample{len(sample_df)}.xlsx")
    with pd.ExcelWriter(ft_xlsx, engine="openpyxl") as w:
        dist_km_df.to_excel(w, sheet_name="distance_km")
        dur_min_df.to_excel(w, sheet_name="duration_min")
        dur_tr_min_df.to_excel(w, sheet_name="duration_in_traffic_min")

    top_dist = top_k_pairs(dist_km_df, k=args.topk, smallest=True).rename(columns={"Value": "Distance_km"})
    top_dur = top_k_pairs(dur_min_df, k=args.topk, smallest=True).rename(columns={"Value": "Duration_min"})
    pd.merge(top_dist, top_dur, on=["From", "To"], how="outer").to_csv(
        os.path.join(outdir, f"top{args.topk}_pairs_{day_label}_sample{len(sample_df)}.csv"), index=False
    )

    # Folium map (opsiyonel)
    if args.export_map:
        out_html = os.path.join(outdir, f"map_{day_label}_n{len(df_day)}.html")
        try_export_folium_map(df_day, depot_lat, depot_lon, routes_by_cluster, out_html)

    print(f"[OK] day={day_label} | n={len(df_day)} | routes:{routes_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default=None, help='Excel path (opsiyonel). Örn: "data.xlsx"')
    ap.add_argument("--csv", default=None, help=f'CSV path (opsiyonel). Default: "{DEFAULT_CSV_NAME}" bulunursa otomatik')

    ap.add_argument("--csv_sep", default=None, help='CSV separator. Örn ";" (boş bırakırsan auto-detect)')
    ap.add_argument("--csv_decimal", default=".", help='CSV decimal separator. Örn ","')
    ap.add_argument("--csv_encoding", default="auto", help='CSV encoding: "auto" (önerilir), "utf-8", "cp1254" ...')

    ap.add_argument("--depot_lat", type=float, default=DEFAULT_DEPOT_LAT, help="Depot latitude (default: Istanbul Anadolu)")
    ap.add_argument("--depot_lon", type=float, default=DEFAULT_DEPOT_LON, help="Depot longitude (default: Istanbul Anadolu)")

    ap.add_argument("--outdir", default="vrp_trial_outputs_google", help="Output folder")

    ap.add_argument("--day", default="Mon", help='Day filter (Mon/Tue/...) or ALL')
    ap.add_argument("--nrows", type=int, default=-1, help="Use first N rows for that day. <=0 means ALL rows for that day")

    ap.add_argument("--k", type=int, default=None, help="Number of clusters/vehicles (optional)")
    ap.add_argument("--target_cluster_size", type=int, default=20, help="Auto-K heuristic")
    ap.add_argument("--max_k", type=int, default=10000, help="Max K if auto")

    ap.add_argument("--min_stops", type=int, default=80, help="Vehicle stop minimum per cluster (default: 80)")
    ap.add_argument("--max_stops", type=int, default=110, help="Vehicle stop maximum per cluster (default: 110)")
    ap.set_defaults(balanced_clusters=True)
    ap.add_argument("--no_balanced_clusters", action="store_false", dest="balanced_clusters", help="Disable capacity-constrained clustering (default is enabled)")
    ap.add_argument("--sample_n", type=int, default=10, help="Sample size for from-to export (Google)")
    ap.add_argument("--mode", default="driving", choices=["driving", "walking", "bicycling", "transit"])
    ap.add_argument("--use_traffic", action="store_true", help="Use duration_in_traffic if available")

    ap.add_argument("--vehicle_cap", type=float, default=350.0, help="Vehicle capacity (packages), default 350")
    ap.add_argument("--delivery_col", default=None, help="Delivery column name (optional). If not given, tries common names or Paket Adedi")
    ap.add_argument("--pickup_col", default=None, help="Pickup column name (optional). If not given, tries common names; else 0")
    ap.add_argument("--route_strategy", default="edge_greedy", choices=["edge_greedy", "full_matrix"], help="Routing strategy: edge_greedy (fast) or full_matrix (expensive)")
    ap.add_argument("--candidate_limit", type=int, default=60, help="For edge_greedy: max candidate stops per step (<=100)")
    ap.add_argument("--alpha_load", type=float, default=0.0, help="For edge_greedy: load-balance penalty weight (seconds per package distance to cap/2)")

    ap.add_argument("--chunk_size", type=int, default=10, help="DM batching (10->100 elements/request)")
    ap.add_argument("--pause_s", type=float, default=0.15, help="Sleep between DM requests")
    ap.add_argument("--topk", type=int, default=7, help="Top K directed pairs export")

    ap.add_argument("--export_cluster_ft", action="store_true",
                    help="Export full from-to xlsx for EACH cluster (cluster cluster)")
    ap.add_argument("--plot_routes", action="store_true", help="Export route PNG per cluster")
    ap.add_argument("--export_map", action="store_true", help="Export folium HTML map (requires folium)")

    args = ap.parse_args()

    # API key'i otomatik bul
    api_key = get_google_maps_api_key()
    gmaps_client = googlemaps.Client(key=api_key)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    ticker = ProgressTicker(interval=10, msg="running...")
    ticker.start()

    # Load input (Excel or CSV) -> standardized columns
    df, depot_lat, depot_lon = load_input(args)

    # Day selection
    if str(args.day).strip().lower() == "all":
        days = sorted(df["Durak Günü"].astype(str).str.strip().unique().tolist())
    else:
        days = [args.day]

    for d in days:
        process_one_day(
            day_label=str(d).strip(),
            df=df,
            depot_lat=depot_lat,
            depot_lon=depot_lon,
            args=args,
            gmaps_client=gmaps_client,
            outdir=outdir,
        )

    ticker.stop()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
