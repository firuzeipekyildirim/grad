#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Cluster-first, route-second (trial) using Google Distance Matrix (no Gurobi).

Fixes vs older versions:
- Accepts: --plot_routes (your command will work)
- Avoids MAX_ELEMENTS_EXCEEDED by removing the old "sample full-matrix export" that sent >100 elements in one request
- Enforces Google per-request limit by capping chunk_size<=10 (full matrix) and candidate_limit<=100 (greedy)
- Uses your "Type" column:
    Type=Delivery  -> delivery = NumOfPackages, pickup = 0
    Type=Pickup    -> pickup   = NumOfPackages, delivery = 0
  (If Type missing, defaults to delivery=NumOfPackages)

Main idea:
- Balanced clustering to keep each cluster stop-count in [min_stops, max_stops]
- Route per cluster with dynamic load feasibility:
    load_after = load_before - delivery_i + pickup_i
    0 <= load_after <= cap_pkgs
  Start load is sum(deliveries in that cluster).

Outputs (in --outdir):
- clustered_stops_<day>.csv
- routes_<day>.csv (sequence with depot + stops)
- routes_summary_<day>.csv
- clusters_<day>.png
- route_<day>_cluster<cid>.png   (when --plot_routes)

Notes:
- Google Distance Matrix hard limit (commonly) is elements = origins*destinations <= 100 per request.
  That's why:
    chunk_size is capped at 10 (10*10=100)
    candidate_limit is capped at 100 (1*100=100)

Minimal run:
  python .\kod\cluster_route_google_nogurobi.py --input .\data_cleaned.csv --days Mon --use_traffic --split_overcap_stops --plot_routes

If you still see API-limit errors, reduce candidate_limit:
  ... --candidate_limit 25

Dependencies:
  pip install pandas numpy scikit-learn matplotlib googlemaps openpyxl
"""

import os
import math
import time
import json
import argparse
import threading
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import googlemaps


# ------------------------- Defaults -------------------------

DEFAULT_DEPOT_LAT = 40.920153
DEFAULT_DEPOT_LON = 29.348245
API_ENV_VAR = "GOOGLE_MAPS_API_KEY"


# ------------------------- Progress indicator -------------------------

class ProgressTicker:
    """Prints a heartbeat message every `interval` seconds until stopped."""
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
    for enc in ("utf-8", "cp1254", "cp1252", "latin1"):
        try:
            with open(path, "r", encoding=enc) as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#"):
                        return s
            return None
        except Exception:
            continue
    return None

def _parse_dotenv(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for enc in ("utf-8", "cp1254", "cp1252", "latin1"):
        try:
            with open(path, "r", encoding=enc) as f:
                lines = f.readlines()
            break
        except Exception:
            lines = []
    for line in lines:
        s = line.strip()
        if (not s) or s.startswith("#") or ("=" not in s):
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out

def _resolve_candidates(filename: str) -> List[str]:
    """cwd + script dir"""
    here = os.path.dirname(os.path.abspath(__file__))
    return [os.path.join(os.getcwd(), filename), os.path.join(here, filename)]

def get_google_maps_api_key(env_var: str = API_ENV_VAR) -> str:
    key = os.getenv(env_var)
    if key and key.strip():
        return key.strip()

    # .env
    for env_path in _resolve_candidates(".env"):
        if os.path.exists(env_path):
            d = _parse_dotenv(env_path)
            if env_var in d and d[env_var].strip():
                return d[env_var].strip()

    # key files
    for fname in ("google_maps_api_key.txt", "gmaps_key.txt", "apikey.txt", "api_key.txt"):
        for fp in _resolve_candidates(fname):
            if os.path.exists(fp):
                s = _read_text_file_first_line(fp)
                if s:
                    return s.strip()

    # config.json
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
        "Çözüm:\n"
        "  1) PowerShell:  $env:GOOGLE_MAPS_API_KEY=\"...\"\n"
        "     Kalıcı: setx GOOGLE_MAPS_API_KEY \"...\"  (sonra yeni terminal)\n"
        "  2) .env: GOOGLE_MAPS_API_KEY=...\n"
        "  3) google_maps_api_key.txt: ilk satır key\n"
        "  4) config.json: {\"GOOGLE_MAPS_API_KEY\": \"...\"}\n"
    )


# ------------------------- Input helpers -------------------------

def _norm(s: str) -> str:
    return str(s).strip().lower()

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts your raw columns and maps them to:
      StopNum, Type, Day, Latitude, Longitude, NumOfPackages
    """
    colmap: Dict[str, str] = {}
    for c in df.columns:
        cn = _norm(c)
        if cn in ("stopnum", "stop num", "durak no", "durak no.", "stop_no", "stopnumber"):
            colmap[c] = "StopNum"
        elif cn in ("type", "tip"):
            colmap[c] = "Type"
        elif cn in ("day", "durak günü", "durak gunu"):
            colmap[c] = "Day"
        elif cn in ("latitude", "lat", "enlem"):
            colmap[c] = "Latitude"
        elif cn in ("longitude", "lon", "lng", "boylam"):
            colmap[c] = "Longitude"
        elif cn in ("numofpackages", "paket adedi", "packages", "packagecount"):
            colmap[c] = "NumOfPackages"
    out = df.rename(columns=colmap).copy()

    # Some CSVs may contain "Long" / "Lat" etc. If still missing, show helpful error.
    return out

def validate_columns(df: pd.DataFrame) -> None:
    req = ["StopNum", "Day", "Latitude", "Longitude", "NumOfPackages"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}. Beklenen en az: {req} (+ opsiyonel Type)")

def load_input_csv(path: str) -> pd.DataFrame:
    # encoding auto
    last = None
    for enc in ("utf-8", "cp1254", "cp1252", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            return df
        except Exception as e:
            last = e
    raise RuntimeError(f"CSV okunamadı: {path}. Son hata: {last}")


# ------------------------- Stop splitting -------------------------

def split_overcap_stops(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """
    If any single stop has NumOfPackages > cap, split into multiple pseudo-stops
    at the same coordinates. StopNum becomes like 2126_1, 2126_2, ...
    """
    rows = []
    for _, r in df.iterrows():
        pk = float(r["NumOfPackages"])
        if pk <= cap + 1e-9:
            rr = r.copy()
            rr["StopKey"] = str(r["StopNum"])
            rows.append(rr)
            continue

        # split
        base = str(r["StopNum"])
        n_parts = int(math.ceil(pk / cap))
        part = pk / n_parts
        for j in range(n_parts):
            rr = r.copy()
            rr["NumOfPackages"] = part
            rr["StopKey"] = f"{base}_{j+1}"
            rows.append(rr)

    out = pd.DataFrame(rows).reset_index(drop=True)
    if "StopKey" not in out.columns:
        out["StopKey"] = out["StopNum"].astype(str)
    return out


# ------------------------- Demand model from Type -------------------------

def build_delivery_pickup(df_part: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uses:
      Type=Delivery -> delivery = NumOfPackages
      Type=Pickup   -> pickup   = NumOfPackages
    Type missing or unknown -> treat as Delivery.
    """
    pk = df_part["NumOfPackages"].to_numpy(dtype=float)
    pk = np.nan_to_num(pk, nan=0.0)

    if "Type" not in df_part.columns:
        return pk, np.zeros(len(df_part), dtype=float)

    t = df_part["Type"].astype(str).str.strip().str.lower().to_numpy()
    delivery = np.zeros(len(df_part), dtype=float)
    pickup = np.zeros(len(df_part), dtype=float)

    for i, tt in enumerate(t):
        if "pick" in tt or "al" in tt:   # Pickup / Alım
            pickup[i] = pk[i]
        else:
            delivery[i] = pk[i]
    return delivery, pickup


# ------------------------- Balanced clustering -------------------------

def _euclid2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=2)

def cluster_kmeans_balanced(
    coords: np.ndarray,
    k: int,
    min_size: int,
    max_size: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Heuristic capacity-constrained clustering:
    - KMeans centroids
    - Min-fill each cluster to min_size
    - Then assign remaining points to nearest cluster with remaining capacity
    """
    n = coords.shape[0]
    if k * min_size > n:
        raise ValueError(f"Impossible: k*min_size={k*min_size} > n={n}")
    if k * max_size < n:
        raise ValueError(f"Impossible: k*max_size={k*max_size} < n={n}")

    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(coords)
    centers = km.cluster_centers_
    d2 = _euclid2(coords, centers)  # NxK
    order = np.argsort(d2, axis=0)  # NxK (each col sorted indices)

    labels = np.full(n, -1, dtype=int)
    unassigned = np.ones(n, dtype=bool)
    filled = np.zeros(k, dtype=int)
    cap = np.full(k, max_size, dtype=int)
    ptr = np.zeros(k, dtype=int)

    # Phase A: min-fill round robin
    while np.any(filled < min_size):
        progressed = False
        for j in range(k):
            while filled[j] < min_size:
                while ptr[j] < n and (not unassigned[order[ptr[j], j]]):
                    ptr[j] += 1
                if ptr[j] >= n:
                    raise RuntimeError("Min-fill failed: ran out of candidates. Try different k.")
                i = int(order[ptr[j], j])
                labels[i] = j
                unassigned[i] = False
                filled[j] += 1
                cap[j] -= 1
                ptr[j] += 1
                progressed = True
                if cap[j] < 0:
                    raise RuntimeError("Capacity error during min-fill.")
        if not progressed:
            break

    # Phase B: assign remaining
    remaining = np.where(unassigned)[0]
    if remaining.size > 0:
        # process easiest-to-place first
        best = np.min(d2[remaining, :], axis=1)
        remaining = remaining[np.argsort(best)]
        for i in remaining:
            valid = np.where(cap > 0)[0]
            if valid.size == 0:
                raise RuntimeError("No capacity left for remaining points. Increase k or max_size.")
            j = int(valid[np.argmin(d2[i, valid])])
            labels[i] = j
            cap[j] -= 1
            filled[j] += 1

    sizes = np.bincount(labels, minlength=k)
    if np.any(sizes < min_size) or np.any(sizes > max_size):
        raise RuntimeError(f"Balanced clustering out-of-bounds. sizes={sizes.tolist()}")
    return labels


def pick_k_auto(n: int, min_stops: int, max_stops: int, target_cluster_size: int, max_k: int) -> Tuple[int, int, int]:
    k_min = int(math.ceil(n / max(1, max_stops)))
    k_max = int(max(1, math.floor(n / max(1, min_stops))))
    if k_min > k_max:
        raise SystemExit(f"Infeasible stop bounds: n={n}, min={min_stops}, max={max_stops} -> k_min={k_min} > k_max={k_max}")
    k_guess = int(math.ceil(n / max(1, target_cluster_size)))
    k = int(min(max(k_guess, k_min), min(k_max, max_k)))
    k = max(k, k_min)
    return k, k_min, k_max


# ------------------------- Google Distance Matrix calls -------------------------

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

def dm_one_to_many(
    gmaps_client: googlemaps.Client,
    origin_lat: float,
    origin_lon: float,
    dest_latlon: np.ndarray,
    mode: str,
    use_traffic: bool,
    language: str = "tr",
    units: str = "metric",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns dist_m, dur_s_used arrays length M.
    Per-request element limit: 1*M <= 100, so M must be <=100.
    """
    origin = [_fmt_latlon(origin_lat, origin_lon)]
    dests = [_fmt_latlon(float(lat), float(lon)) for lat, lon in dest_latlon]
    if len(dests) > 100:
        raise ValueError("Too many destinations in one DM request. Use smaller --candidate_limit (<=100).")

    resp = _dm_call_with_retry(
        gmaps_client,
        origins=origin,
        destinations=dests,
        mode=mode,
        departure_time="now" if use_traffic else None,
        language=language,
        units=units,
    )
    elems = resp.get("rows", [{}])[0].get("elements", [])
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


# ------------------------- Routing: load-feasible greedy -------------------------

def solve_cluster_route_edge_greedy_google(
    cdf: pd.DataFrame,
    depot_lat: float,
    depot_lon: float,
    gmaps_client: googlemaps.Client,
    cap_pkgs: float,
    mode: str,
    use_traffic: bool,
    candidate_limit: int,
    alpha_load: float,
) -> Dict[str, Any]:
    """
    Greedy route builder with load feasibility.
    Select next stop among feasible ones by minimal (DM duration + alpha*|load_after - cap/2|).

    Start load = sum(deliveries).
    End load   = sum(pickups).
    """
    coords = cdf[["Latitude", "Longitude"]].to_numpy(dtype=float)
    stop_keys = cdf["StopKey"].astype(str).tolist()

    delivery, pickup = build_delivery_pickup(cdf)
    load = float(np.sum(delivery))
    if load > cap_pkgs + 1e-9:
        raise RuntimeError(
            f"Cluster infeasible at start: sum(delivery)={load:.1f} > cap={cap_pkgs:.1f}. "
            "Increase k, or enable --split_overcap_stops, or relax bounds."
        )

    target = cap_pkgs / 2.0
    remaining = list(range(len(cdf)))  # indices
    route_nodes = [0]  # 0=DEPOT, then 1..n
    legs: List[Tuple[str, str, float, float, float, float]] = []

    cur_lat, cur_lon = depot_lat, depot_lon

    # Safety caps for Google
    candidate_limit = int(max(1, min(100, candidate_limit)))

    while remaining:
        feas = []
        for idx in remaining:
            load_after = load - float(delivery[idx]) + float(pickup[idx])
            if (0.0 <= load_after <= cap_pkgs + 1e-9):
                feas.append(idx)

        if not feas:
            # Greedy may get stuck. This is a real feasibility issue (or bad order).
            raise RuntimeError(
                "No feasible next stop under load constraints. "
                "Try larger k, or increase --alpha_load, or reduce cluster sizes."
            )

        # prune candidates by euclidean closeness in lat/lon space
        feas_coords = coords[feas]
        d2 = (feas_coords[:, 0] - cur_lat) ** 2 + (feas_coords[:, 1] - cur_lon) ** 2
        order = np.argsort(d2)
        cand = [feas[i] for i in order[:min(candidate_limit, len(feas))]]

        dist_m, dur_s = dm_one_to_many(
            gmaps_client,
            origin_lat=cur_lat, origin_lon=cur_lon,
            dest_latlon=coords[cand],
            mode=mode,
            use_traffic=use_traffic,
        )

        best_idx = None
        best_obj = float("inf")
        best_dist = float("nan")
        best_dur = float("nan")

        for j, idx in enumerate(cand):
            if not np.isfinite(dur_s[j]):
                continue
            load_after = load - float(delivery[idx]) + float(pickup[idx])
            obj = float(dur_s[j])
            if alpha_load > 0:
                obj += float(alpha_load) * abs(load_after - target)
            if obj < best_obj:
                best_obj = obj
                best_idx = idx
                best_dist = float(dist_m[j]) if np.isfinite(dist_m[j]) else float("nan")
                best_dur = float(dur_s[j])

        if best_idx is None:
            raise RuntimeError("All DM candidates returned invalid duration. Check coordinates or API status.")

        load_before = load
        load_after = load_before - float(delivery[best_idx]) + float(pickup[best_idx])

        legs.append((
            "DEPOT" if route_nodes[-1] == 0 else stop_keys[route_nodes[-1]-1],
            stop_keys[best_idx],
            best_dist, best_dur,
            load_before, load_after
        ))

        load = load_after
        cur_lat, cur_lon = float(coords[best_idx, 0]), float(coords[best_idx, 1])
        route_nodes.append(int(best_idx) + 1)
        remaining.remove(best_idx)

    # return to depot
    dist_back, dur_back = dm_one_to_many(
        gmaps_client,
        origin_lat=cur_lat, origin_lon=cur_lon,
        dest_latlon=np.array([[depot_lat, depot_lon]], dtype=float),
        mode=mode,
        use_traffic=use_traffic,
    )
    legs.append((
        stop_keys[route_nodes[-1]-1] if route_nodes[-1] != 0 else "DEPOT",
        "DEPOT",
        float(dist_back[0]), float(dur_back[0]),
        load, load
    ))
    route_nodes.append(0)

    total_km = float(np.nansum([x[2] for x in legs]) / 1000.0)
    total_min = float(np.nansum([x[3] for x in legs]) / 60.0)

    return {
        "route": route_nodes,
        "legs": legs,
        "total_km": total_km,
        "total_min": total_min,
        "delivery": delivery,
        "pickup": pickup,
        "stop_keys": stop_keys,
    }


# ------------------------- Visualizations -------------------------

def plot_clusters_points(df_day: pd.DataFrame, depot_lat: float, depot_lon: float, out_png: str):
    plt.figure()
    for cid, sub in df_day.groupby("cluster_id"):
        plt.scatter(sub["Longitude"], sub["Latitude"], s=10, label=f"cluster {int(cid)}")
    plt.scatter([depot_lon], [depot_lat], marker="x", s=80, label="DEPOT")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc="best", fontsize=8)
    plt.title("Clusters")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_route_for_cluster(
    cdf: pd.DataFrame,
    depot_lat: float,
    depot_lon: float,
    route: List[int],
    out_png: str,
    title: str,
    annotate_every: int = 5,
):
    pts = np.vstack([[depot_lat, depot_lon], cdf[["Latitude", "Longitude"]].to_numpy()])
    lons = pts[:, 1]
    lats = pts[:, 0]
    xs = [lons[i] for i in route]
    ys = [lats[i] for i in route]

    plt.figure()
    plt.scatter(lons[1:], lats[1:], s=18)
    plt.scatter([lons[0]], [lats[0]], marker="x", s=90)
    plt.plot(xs, ys, linewidth=1.3)

    if annotate_every and annotate_every > 0:
        seq_no = 1
        for idx in route:
            if idx == 0:
                continue
            if seq_no % annotate_every == 0:
                plt.text(lons[idx], lats[idx], str(seq_no), fontsize=7)
            seq_no += 1

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=240)
    plt.close()


# ------------------------- Day processing -------------------------

def parse_days(days_arg: str) -> List[str]:
    s = str(days_arg).strip()
    if s.lower() == "all":
        return ["ALL"]
    parts = [x.strip() for x in s.split(",") if x.strip()]
    return parts if parts else ["Mon"]

def process_one_day(
    day_label: str,
    df: pd.DataFrame,
    depot_lat: float,
    depot_lon: float,
    args,
    gmaps_client: googlemaps.Client,
    outdir: str,
) -> None:
    df_day = df[df["Day"].astype(str).str.strip().str.lower() == day_label.lower()].copy()
    if len(df_day) == 0:
        print(f"[SKIP] day={day_label} için satır yok.", flush=True)
        return

    if args.nrows > 0:
        df_day = df_day.head(args.nrows).copy()

    coords = df_day[["Latitude", "Longitude"]].to_numpy(dtype=float)
    n = len(df_day)

    if args.k is None:
        k, k_min, k_max = pick_k_auto(n, args.min_stops, args.max_stops, args.target_cluster_size, args.max_k)
        k_mode = "auto"
    else:
        k = int(args.k)
        # still compute feasible range for info
        k_min = int(math.ceil(n / max(1, args.max_stops)))
        k_max = int(max(1, math.floor(n / max(1, args.min_stops))))
        k_mode = "manual"

    print(f"[INFO] day={day_label} | n_stops={n} | feasible_k=[{k_min},{k_max}] | k_used={k} ({k_mode})", flush=True)

    # (Optional) Increase k if clusters cannot satisfy start/end cap sums.
    # This is a heuristic safety net.
    k_try = k
    best_labels = None
    while True:
        labels = cluster_kmeans_balanced(coords, k_try, args.min_stops, args.max_stops, seed=42)
        df_day["cluster_id"] = labels

        # cap checks: sum(delivery) <= cap AND sum(pickup) <= cap
        ok = True
        for cid in sorted(df_day["cluster_id"].unique()):
            cdf = df_day[df_day["cluster_id"] == cid]
            delivery, pickup = build_delivery_pickup(cdf)
            if float(np.sum(delivery)) > args.cap_pkgs + 1e-9:
                ok = False
                break
            if float(np.sum(pickup)) > args.cap_pkgs + 1e-9:
                ok = False
                break

        if ok:
            best_labels = labels
            break

        if k_try >= k_max:
            print(
                f"[WARN] Cap check failed but cannot increase k further (k_max={k_max}). "
                "Will continue; some clusters may be infeasible by capacity.",
                flush=True
            )
            best_labels = labels
            break

        k_try += 1

    df_day["cluster_id"] = best_labels
    sizes = df_day["cluster_id"].value_counts().sort_index()
    sizes_str = ", ".join([f"{int(cid)}:{int(cnt)}" for cid, cnt in sizes.items()])
    print(f"[INFO] day={day_label} | cluster_sizes (cluster:count) -> {sizes_str}", flush=True)

    os.makedirs(outdir, exist_ok=True)
    clustered_path = os.path.join(outdir, f"clustered_stops_{day_label}_n{len(df_day)}.csv")
    df_day.to_csv(clustered_path, index=False)

    # per cluster routes
    routes_rows = []
    summary_rows = []

    for cid in sorted(df_day["cluster_id"].unique()):
        cdf = df_day[df_day["cluster_id"] == cid].copy().reset_index(drop=True)

        # ensure StopKey exists
        if "StopKey" not in cdf.columns:
            cdf["StopKey"] = cdf["StopNum"].astype(str)

        n_c = len(cdf) + 1
        print(f"[INFO] day={day_label} | cluster={int(cid)} | n_stops={len(cdf)} | greedy_DM_calls~{len(cdf)+1}", flush=True)

        sol = solve_cluster_route_edge_greedy_google(
            cdf=cdf,
            depot_lat=depot_lat, depot_lon=depot_lon,
            gmaps_client=gmaps_client,
            cap_pkgs=float(args.cap_pkgs),
            mode=str(args.mode),
            use_traffic=bool(args.use_traffic),
            candidate_limit=int(args.candidate_limit),
            alpha_load=float(args.alpha_load),
        )
        route = sol["route"]

        # route rows
        for seq, node in enumerate(route):
            if node == 0:
                routes_rows.append({
                    "day": day_label, "cluster_id": int(cid), "seq": int(seq),
                    "StopKey": "DEPOT",
                    "StopNum": "DEPOT",
                    "Type": "DEPOT",
                    "Longitude": float(depot_lon),
                    "Latitude": float(depot_lat),
                    "NumOfPackages": np.nan,
                    "Delivery": np.nan,
                    "Pickup": np.nan,
                })
            else:
                row = cdf.iloc[node - 1]
                routes_rows.append({
                    "day": day_label, "cluster_id": int(cid), "seq": int(seq),
                    "StopKey": str(row.get("StopKey", row["StopNum"])),
                    "StopNum": row["StopNum"],
                    "Type": row.get("Type", ""),
                    "Longitude": float(row["Longitude"]),
                    "Latitude": float(row["Latitude"]),
                    "NumOfPackages": float(row["NumOfPackages"]),
                    "Delivery": float(sol["delivery"][node - 1]),
                    "Pickup": float(sol["pickup"][node - 1]),
                })

        summary_rows.append({
            "day": day_label,
            "cluster_id": int(cid),
            "n_stops": int(len(cdf)),
            "total_km": float(sol["total_km"]),
            "total_min": float(sol["total_min"]),
            "cap_pkgs": float(args.cap_pkgs),
            "candidate_limit": int(min(100, max(1, args.candidate_limit))),
            "alpha_load": float(args.alpha_load),
            "mode": str(args.mode),
            "use_traffic": bool(args.use_traffic),
        })

        if args.plot_routes:
            out_png = os.path.join(outdir, f"route_{day_label}_cluster{int(cid)}_n{len(cdf)}.png")
            plot_route_for_cluster(
                cdf=cdf,
                depot_lat=depot_lat, depot_lon=depot_lon,
                route=route,
                out_png=out_png,
                title=f"Route | day={day_label} | cluster={int(cid)} | n={len(cdf)}",
                annotate_every=int(args.annotate_every),
            )

    routes_df = pd.DataFrame(routes_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(["day", "cluster_id"])

    routes_path = os.path.join(outdir, f"routes_{day_label}_n{len(df_day)}.csv")
    summary_path = os.path.join(outdir, f"routes_summary_{day_label}_n{len(df_day)}.csv")
    routes_df.to_csv(routes_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    points_plot = os.path.join(outdir, f"clusters_{day_label}_n{len(df_day)}.png")
    plot_clusters_points(df_day, depot_lat, depot_lon, points_plot)

    print(f"[OK] day={day_label} | outputs -> {outdir}", flush=True)


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV path, ex: .\\data_cleaned.csv")
    ap.add_argument("--outdir", default="vrp_trial_outputs_google", help="Output folder")
    ap.add_argument("--days", default="Mon", help='Day filter: "Mon" or "Mon,Tue" or "ALL"')
    ap.add_argument("--nrows", type=int, default=-1, help="Use first N rows for each day. <=0 means ALL")

    ap.add_argument("--depot_lat", type=float, default=DEFAULT_DEPOT_LAT)
    ap.add_argument("--depot_lon", type=float, default=DEFAULT_DEPOT_LON)

    ap.add_argument("--min_stops", type=int, default=80)
    ap.add_argument("--max_stops", type=int, default=110)
    ap.add_argument("--target_cluster_size", type=int, default=95)
    ap.add_argument("--max_k", type=int, default=10000)
    ap.add_argument("--k", type=int, default=None)

    ap.add_argument("--cap_pkgs", type=float, default=350.0, help="Vehicle capacity (packages)")
    ap.add_argument("--split_overcap_stops", action="store_true", help="Split any single stop with packages > cap into pseudo-stops")

    ap.add_argument("--mode", default="driving", choices=["driving", "walking", "bicycling", "transit"])
    ap.add_argument("--use_traffic", action="store_true", help="Use duration_in_traffic when available")
    ap.add_argument("--candidate_limit", type=int, default=60, help="For greedy routing: destinations per DM call (<=100)")
    ap.add_argument("--alpha_load", type=float, default=0.0, help="Load-balance penalty weight")

    ap.add_argument("--plot_routes", action="store_true", help="Export route PNG per cluster")
    ap.add_argument("--annotate_every", type=int, default=5, help="Annotate every Nth stop on route plots (0=none)")

    ap.add_argument("--ticker_s", type=int, default=10, help="Progress ticker interval seconds")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input bulunamadı: {args.input}")

    api_key = get_google_maps_api_key()
    gmaps_client = googlemaps.Client(key=api_key)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    ticker = ProgressTicker(interval=int(max(1, args.ticker_s)), msg="running...")
    ticker.start()

    df_raw = load_input_csv(args.input)
    df = standardize_columns(df_raw)
    validate_columns(df)

    # Normalize types
    if "Type" in df.columns:
        df["Type"] = df["Type"].astype(str).str.strip()

    # Ensure numeric
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["NumOfPackages"] = pd.to_numeric(df["NumOfPackages"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["Latitude", "Longitude"]).copy()

    # Stop splitting
    if args.split_overcap_stops:
        df = split_overcap_stops(df, cap=float(args.cap_pkgs))
    else:
        df["StopKey"] = df["StopNum"].astype(str)

    # Days
    dlist = parse_days(args.days)
    if dlist == ["ALL"]:
        days = sorted(df["Day"].astype(str).str.strip().unique().tolist())
    else:
        days = dlist

    for d in days:
        process_one_day(
            day_label=str(d).strip(),
            df=df,
            depot_lat=float(args.depot_lat),
            depot_lon=float(args.depot_lon),
            args=args,
            gmaps_client=gmaps_client,
            outdir=outdir,
        )

    ticker.stop()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
