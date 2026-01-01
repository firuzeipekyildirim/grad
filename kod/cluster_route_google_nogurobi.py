#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_route_google_nogurobi.py
--------------------------------
Cluster-first, route-second prototype (NO GUROBI) using Google Distance Matrix.

Input (CSV):
  StopNum, Type, Day, Longitude, Latitude, NumOfPackages

- Type: "Delivery" or "Pickup"
- Day : e.g., Mon/Tue/...

What this script does
1) Load CSV (tries Turkish-friendly encodings)
2) For each chosen day:
   a) Choose k (clusters/vehicles) to satisfy stop-count constraints
   b) KMeans clustering on (lat,lon)
   c) Rebalance cluster sizes to respect min_stops <= size <= max_stops
3) For each cluster:
   Build a feasible route with a greedy heuristic that uses Google travel time/distance:
   - Starts at depot, ends at depot
   - Maintains dynamic load with capacity cap_pkgs (packages)
   - Allows returning to depot to (re)load deliveries and unload pickups when needed
   - Uses Distance Matrix in a safe way (1 origin -> <=25 destinations per request) to avoid MAX_DIMENSIONS_EXCEEDED
4) Exports:
   - clustered_<DAY>.csv
   - routes_<DAY>.csv
   - routes_summary_<DAY>.csv
   - from_to_google_sample<N>_<DAY>.xlsx  (distance_km + duration_min (+ traffic if enabled))
   - cluster_plot_<DAY>.png (clusters)
   - route_plot_sample_<DAY>.png (routes for a few clusters, optional)

Notes about capacity model (prototype)
- Each row is a stop that is either Delivery or Pickup (not both).
- We keep two quantities:
    delivery_stock: how many delivery packages currently loaded to deliver
    pickup_load   : how many pickup packages collected and still onboard
  total_load = delivery_stock + pickup_load  must be <= cap_pkgs
- At the depot, we set pickup_load = 0 and refill delivery_stock up to cap.
- If a stop requires more than cap_pkgs packages, we can split it into multiple pseudo-stops with the same coordinates.

Install:
  pip install pandas numpy scikit-learn matplotlib googlemaps openpyxl

Run (example, from grad/):
  python .\kod\cluster_route_google_nogurobi.py --input .\data_cleaned.csv --days Mon --use_traffic

If you want ALL days:
  python .\kod\cluster_route_google_nogurobi.py --input .\data_cleaned.csv --days all --use_traffic

"""

from __future__ import annotations

import os
import math
import time
import argparse
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import googlemaps


# ------------------------- Progress indicator -------------------------

class ProgressTicker:
    """Prints a heartbeat message every `interval` seconds until stopped."""
    def __init__(self, interval: int = 10, msg: str = "running..."):
        self.interval = int(interval)
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


# ------------------------- Utilities -------------------------

def read_csv_flexible(path: Path) -> pd.DataFrame:
    """Try common encodings used in TR/Windows exports."""
    encs = ["utf-8-sig", "utf-8", "cp1254", "cp1252", "iso-8859-9", "latin1"]
    last = None
    for enc in encs:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last = e
    raise last


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize expected columns."""
    cols = {c.strip(): c for c in df.columns}
    required = ["StopNum", "Type", "Day", "Longitude", "Latitude", "NumOfPackages"]
    miss = [c for c in required if c not in cols]
    if miss:
        raise ValueError(f"CSV columns missing: {miss}. Found: {list(df.columns)}")

    out = df[[cols[c] for c in required]].copy()
    out.columns = required

    out["StopNum"] = pd.to_numeric(out["StopNum"], errors="coerce").astype("Int64")
    out["Longitude"] = pd.to_numeric(out["Longitude"], errors="coerce")
    out["Latitude"] = pd.to_numeric(out["Latitude"], errors="coerce")
    out["NumOfPackages"] = pd.to_numeric(out["NumOfPackages"], errors="coerce")

    out["Type"] = out["Type"].astype(str).str.strip()
    out["Day"] = out["Day"].astype(str).str.strip()

    out = out.dropna(subset=["StopNum", "Longitude", "Latitude", "NumOfPackages", "Day", "Type"]).copy()
    out["StopNum"] = out["StopNum"].astype(int)
    out["NumOfPackages"] = out["NumOfPackages"].astype(float)

    return out


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Haversine distance in meters."""
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))


def safe_filename(name: str) -> str:
    """Make a LaTeX/OS friendly filename."""
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "file"


# ------------------------- Google Distance Matrix helpers -------------------------

def _dm_call_with_retry(
    gmaps_client: googlemaps.Client,
    origins: list[str],
    destinations: list[str],
    mode: str,
    language: str,
    units: str,
    departure_time,
    max_retries: int = 5,
    pause_s: float = 0.25,
):
    last_err = None
    for r in range(max_retries):
        try:
            resp = gmaps_client.distance_matrix(
                origins=origins,
                destinations=destinations,
                mode=mode,
                departure_time=departure_time,
                units=units,
                language=language,
            )
            return resp
        except Exception as e:
            last_err = e
            time.sleep(pause_s * (2**r))
    raise RuntimeError(f"Google DM call failed after retries. Last error: {last_err}")


def dm_one_to_many(
    gmaps_client: googlemaps.Client,
    origin_lat: float,
    origin_lon: float,
    dest_latlons: list[tuple[float, float]],
    mode: str = "driving",
    use_traffic: bool = False,
    chunk_size: int = 25,
    pause_s: float = 0.1,
    language: str = "tr",
    units: str = "metric",
):
    """
    Safe Distance Matrix:
      1 origin -> up to 25 destinations per request (chunked)
    Returns arrays:
      dist_m, dur_s, (optional) dur_tr_s
    """
    if chunk_size > 25:
        # To avoid MAX_DIMENSIONS_EXCEEDED
        chunk_size = 25

    origin = f"{origin_lat},{origin_lon}"
    # Traffic requires driving + departure_time now/future
    departure_time = "now" if (use_traffic and mode == "driving") else None

    dists = []
    durs = []
    durs_tr = []

    for i in range(0, len(dest_latlons), chunk_size):
        chunk = dest_latlons[i:i+chunk_size]
        dests = [f"{la},{lo}" for la, lo in chunk]

        resp = _dm_call_with_retry(
            gmaps_client,
            origins=[origin],
            destinations=dests,
            mode=mode,
            language=language,
            units=units,
            departure_time=departure_time,
            pause_s=max(0.05, pause_s),
        )

        row = resp["rows"][0]["elements"]
        for e in row:
            if e.get("status") != "OK":
                # If Google cannot route, mark as inf
                dists.append(float("inf"))
                durs.append(float("inf"))
                if use_traffic:
                    durs_tr.append(float("inf"))
                continue
            dists.append(float(e["distance"]["value"]))
            durs.append(float(e["duration"]["value"]))
            if use_traffic:
                durs_tr.append(float(e.get("duration_in_traffic", e["duration"])["value"]))

        time.sleep(pause_s)

    dist_m = np.array(dists, dtype=float)
    dur_s = np.array(durs, dtype=float)
    if use_traffic:
        dur_tr_s = np.array(durs_tr, dtype=float)
        return dist_m, dur_s, dur_tr_s
    return dist_m, dur_s, None


def build_from_to_google_sample(
    gmaps_client: googlemaps.Client,
    depot_lat: float,
    depot_lon: float,
    df_points: pd.DataFrame,
    mode: str,
    use_traffic: bool,
    chunk_size: int,
    pause_s: float,
) -> dict[str, pd.DataFrame]:
    """
    Build a from-to chart (DEPOT + sample points).
    Uses chunked calls to respect API limits:
      origins_chunk <= 25, destinations_chunk <= 25
    Returns DataFrames:
      distance_km, duration_min, (optional) duration_in_traffic_min
    """
    pts = [(depot_lat, depot_lon)] + list(zip(df_points["Latitude"].tolist(), df_points["Longitude"].tolist()))
    labels = ["DEPOT"] + [str(x) for x in df_points["StopNum"].tolist()]
    n = len(pts)

    dist = np.full((n, n), np.inf, dtype=float)
    dur = np.full((n, n), np.inf, dtype=float)
    dur_tr = np.full((n, n), np.inf, dtype=float) if use_traffic else None

    # chunk origins and destinations by 25
    for oi in range(0, n, 25):
        o_chunk = pts[oi:oi+25]
        o_str = [f"{la},{lo}" for la, lo in o_chunk]
        for dj in range(0, n, 25):
            d_chunk = pts[dj:dj+25]
            d_str = [f"{la},{lo}" for la, lo in d_chunk]

            departure_time = "now" if (use_traffic and mode == "driving") else None
            resp = _dm_call_with_retry(
                gmaps_client,
                origins=o_str,
                destinations=d_str,
                mode=mode,
                language="tr",
                units="metric",
                departure_time=departure_time,
                pause_s=max(0.05, pause_s),
            )
            rows = resp["rows"]
            for r_idx, row in enumerate(rows):
                for c_idx, e in enumerate(row["elements"]):
                    if e.get("status") != "OK":
                        continue
                    dist[oi+r_idx, dj+c_idx] = float(e["distance"]["value"])
                    dur[oi+r_idx, dj+c_idx] = float(e["duration"]["value"])
                    if use_traffic and dur_tr is not None:
                        dur_tr[oi+r_idx, dj+c_idx] = float(e.get("duration_in_traffic", e["duration"])["value"])

            time.sleep(pause_s)

    out = {
        "distance_km": pd.DataFrame(dist/1000.0, index=labels, columns=labels),
        "duration_min": pd.DataFrame(dur/60.0, index=labels, columns=labels),
    }
    if use_traffic and dur_tr is not None:
        out["duration_in_traffic_min"] = pd.DataFrame(dur_tr/60.0, index=labels, columns=labels)
    return out


# ------------------------- Clustering (KMeans + rebalance sizes) -------------------------

def feasible_k_range(n_stops: int, min_stops: int, max_stops: int) -> tuple[int, int]:
    k_min = math.ceil(n_stops / max_stops)  # need at least this many vehicles to not exceed max
    k_max = n_stops // min_stops            # at most this many vehicles to keep min
    return k_min, k_max


def choose_k_auto(n_stops: int, min_stops: int, max_stops: int) -> int:
    k_min, k_max = feasible_k_range(n_stops, min_stops, max_stops)
    if k_min > k_max:
        raise ValueError(
            f"Stop-count constraints infeasible: n={n_stops}, min={min_stops}, max={max_stops}. "
            f"Feasible k would require k in [{k_min},{k_max}] but empty."
        )
    # Aim for mid-size clusters
    target = (min_stops + max_stops) / 2.0
    k_guess = int(round(n_stops / target))
    return int(min(k_max, max(k_min, k_guess)))


def kmeans_labels(df_day: pd.DataFrame, k: int) -> np.ndarray:
    X = df_day[["Latitude", "Longitude"]].to_numpy()
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(X)


def rebalance_cluster_sizes(
    df_day: pd.DataFrame,
    labels: np.ndarray,
    min_size: int,
    max_size: int,
    max_iter: int = 20000,
) -> np.ndarray:
    """
    Rebalance cluster sizes by moving points from oversized clusters to undersized ones.
    Greedy based on centroid distances (fast & usually good enough for a prototype).
    """
    X = df_day[["Latitude", "Longitude"]].to_numpy()
    labels = labels.copy()

    k = int(labels.max() + 1)
    for _ in range(max_iter):
        sizes = np.bincount(labels, minlength=k)
        unders = np.where(sizes < min_size)[0]
        overs = np.where(sizes > max_size)[0]
        if len(unders) == 0 and len(overs) == 0:
            return labels

        # compute centroids
        centroids = np.zeros((k, 2), dtype=float)
        for cid in range(k):
            idx = np.where(labels == cid)[0]
            if len(idx) == 0:
                centroids[cid] = X.mean(axis=0)
            else:
                centroids[cid] = X[idx].mean(axis=0)

        if len(overs) > 0 and len(unders) > 0:
            # take one oversized cluster, move one point to closest undersized cluster
            o = int(overs[0])
            o_idx = np.where(labels == o)[0]
            # farthest points from o centroid are easiest to "give away"
            d_to_o = np.sum((X[o_idx] - centroids[o])**2, axis=1)
            give_order = o_idx[np.argsort(-d_to_o)]  # farthest first

            moved = False
            for p in give_order[:200]:  # cap search
                # choose closest undersized centroid
                d_to_u = np.sum((centroids[unders] - X[p])**2, axis=1)
                u = int(unders[int(np.argmin(d_to_u))])
                labels[p] = u
                moved = True
                break
            if moved:
                continue

        # If only overs remain (rare), or move failed: soften by moving from largest to smallest
        o = int(np.argmax(sizes))
        u = int(np.argmin(sizes))
        if o == u:
            return labels
        o_idx = np.where(labels == o)[0]
        if len(o_idx) == 0:
            return labels
        # move point closest to u centroid
        d = np.sum((X[o_idx] - centroids[u])**2, axis=1)
        labels[o_idx[int(np.argmin(d))]] = u

    return labels


# ------------------------- Routing heuristic with capacity -------------------------

@dataclass
class Stop:
    stop_id: str          # unique id (can be split pseudo-stops)
    stopnum: int
    kind: str             # Delivery or Pickup
    lat: float
    lon: float
    pkgs: float


def split_overcap_stops(df_cluster: pd.DataFrame, cap_pkgs: float) -> pd.DataFrame:
    """
    If a stop has NumOfPackages > cap, split into multiple pseudo-stops with same coords.
    Adds columns: StopID (string).
    """
    rows = []
    for _, r in df_cluster.iterrows():
        stopnum = int(r["StopNum"])
        pkgs = float(r["NumOfPackages"])
        if pkgs <= cap_pkgs:
            rr = r.copy()
            rr["StopID"] = f"{stopnum}"
            rows.append(rr)
            continue

        n_parts = int(math.ceil(pkgs / cap_pkgs))
        remaining = pkgs
        for part in range(1, n_parts + 1):
            take = min(cap_pkgs, remaining)
            remaining -= take
            rr = r.copy()
            rr["NumOfPackages"] = float(take)
            rr["StopID"] = f"{stopnum}_p{part}"
            rows.append(rr)

    out = pd.DataFrame(rows).reset_index(drop=True)
    return out


def greedy_route_with_depot_returns(
    cdf: pd.DataFrame,
    depot_lat: float,
    depot_lon: float,
    gmaps_client: googlemaps.Client,
    cap_pkgs: float = 350.0,
    mode: str = "driving",
    use_traffic: bool = False,
    candidate_limit: int = 25,
    chunk_size: int = 25,
    pause_s: float = 0.1,
    alpha_load: float = 0.0,
):
    """
    Build a route that visits all stops in cdf.

    We keep:
      delivery_stock: packages available to deliver
      pickup_load: picked up packages currently onboard
      total_load = delivery_stock + pickup_load <= cap

    At depot:
      pickup_load = 0
      delivery_stock = min(cap, remaining_delivery_total)
    """
    # Prepare stop list
    stops = []
    for _, r in cdf.iterrows():
        stops.append(
            Stop(
                stop_id=str(r["StopID"]),
                stopnum=int(r["StopNum"]),
                kind=str(r["Type"]).strip(),
                lat=float(r["Latitude"]),
                lon=float(r["Longitude"]),
                pkgs=float(r["NumOfPackages"]),
            )
        )

    remaining = {s.stop_id: s for s in stops}

    def remaining_delivery_total() -> float:
        return sum(s.pkgs for s in remaining.values() if s.kind.lower() == "delivery")

    # route log rows
    route_rows = []

    # state
    cur_lat, cur_lon = depot_lat, depot_lon
    at_depot = True
    pickup_load = 0.0
    delivery_stock = min(cap_pkgs, remaining_delivery_total())

    seq = 0
    total_dist_m = 0.0
    total_dur_s = 0.0
    total_dur_tr_s = 0.0

    def log_step(
        node_label: str,
        stop: Optional[Stop],
        leg_dist_m: float,
        leg_dur_s: float,
        leg_dur_tr_s: Optional[float],
        pickup_load_after: float,
        delivery_stock_after: float,
        note: str = "",
    ):
        nonlocal seq
        route_rows.append({
            "seq": seq,
            "node": node_label,
            "StopID": None if stop is None else stop.stop_id,
            "StopNum": None if stop is None else stop.stopnum,
            "Type": None if stop is None else stop.kind,
            "NumOfPackages": None if stop is None else stop.pkgs,
            "Latitude": cur_lat if stop is None else stop.lat,
            "Longitude": cur_lon if stop is None else stop.lon,
            "leg_distance_km": None if math.isinf(leg_dist_m) else leg_dist_m/1000.0,
            "leg_duration_min": None if math.isinf(leg_dur_s) else leg_dur_s/60.0,
            "leg_duration_in_traffic_min": None if (leg_dur_tr_s is None or math.isinf(leg_dur_tr_s)) else leg_dur_tr_s/60.0,
            "pickup_load_after": pickup_load_after,
            "delivery_stock_after": delivery_stock_after,
            "total_load_after": pickup_load_after + delivery_stock_after,
            "note": note,
        })
        seq += 1

    # initial depot record
    log_step("DEPOT_START", None, 0.0, 0.0, 0.0 if use_traffic else None, pickup_load, delivery_stock, note="start")

    # Precompute rough neighbor ordering by haversine to reduce DM calls
    rem_ids = list(remaining.keys())

    while remaining:
        # Candidate pool: nearest by haversine (cheap)
        rem_list = list(remaining.values())
        hv = np.array([haversine_m(cur_lat, cur_lon, s.lat, s.lon) for s in rem_list], dtype=float)
        order = np.argsort(hv)
        cand = [rem_list[i] for i in order[:max(5, min(candidate_limit, len(rem_list)))]]

        # Filter feasibility by capacity dynamics
        feasible = []
        for s in cand:
            if s.kind.lower() == "delivery":
                if s.pkgs <= delivery_stock:
                    feasible.append(s)
            else:  # pickup
                if (delivery_stock + pickup_load + s.pkgs) <= cap_pkgs:
                    feasible.append(s)

        if not feasible:
            # No feasible candidate: go depot to reset (unload pickups + refill deliveries)
            if not at_depot:
                # get travel from current to depot (1 destination)
                dist_m, dur_s, dur_tr = dm_one_to_many(
                    gmaps_client,
                    origin_lat=cur_lat, origin_lon=cur_lon,
                    dest_latlons=[(depot_lat, depot_lon)],
                    mode=mode, use_traffic=use_traffic, chunk_size=chunk_size, pause_s=pause_s
                )
                leg_dist_m = float(dist_m[0])
                leg_dur_s = float(dur_s[0])
                leg_dur_tr_s = float(dur_tr[0]) if (use_traffic and dur_tr is not None) else None

                total_dist_m += leg_dist_m
                total_dur_s += leg_dur_s
                if leg_dur_tr_s is not None:
                    total_dur_tr_s += leg_dur_tr_s

                # move to depot
                cur_lat, cur_lon = depot_lat, depot_lon
                at_depot = True
                pickup_load = 0.0
                delivery_stock = min(cap_pkgs, remaining_delivery_total())
                log_step("DEPOT", None, leg_dist_m, leg_dur_s, leg_dur_tr_s, pickup_load, delivery_stock, note="reset(load/unload)")
                continue

            # At depot and still no feasible candidates -> means some stop > cap (should have been split)
            # or delivery_stock is 0 but remaining deliveries exist (cap=0?) etc.
            bad = max(remaining.values(), key=lambda s: s.pkgs)
            raise RuntimeError(
                f"No feasible next stop even at depot. Example stop={bad.stop_id} kind={bad.kind} pkgs={bad.pkgs} cap={cap_pkgs}. "
                f"Use --split_overcap_stops or increase cap."
            )

        # Query Google DM from current -> feasible candidates (chunked)
        dest_latlons = [(s.lat, s.lon) for s in feasible]
        dist_m, dur_s, dur_tr = dm_one_to_many(
            gmaps_client,
            origin_lat=cur_lat, origin_lon=cur_lon,
            dest_latlons=dest_latlons,
            mode=mode, use_traffic=use_traffic, chunk_size=chunk_size, pause_s=pause_s
        )
        # Choose best by duration (traffic if enabled) + optional load penalty
        best_j = None
        best_cost = float("inf")
        for j, s in enumerate(feasible):
            base = float(dur_tr[j]) if (use_traffic and dur_tr is not None) else float(dur_s[j])
            if math.isinf(base):
                continue

            # simulate load after visiting s
            del_after = delivery_stock
            pick_after = pickup_load
            if s.kind.lower() == "delivery":
                del_after -= s.pkgs
            else:
                pick_after += s.pkgs

            # small extra: prefer to keep total load not too close to cap (optional)
            load_after = del_after + pick_after
            load_pen = alpha_load * abs(load_after - 0.6*cap_pkgs)  # soft target 60% cap
            cost = base + load_pen

            if cost < best_cost:
                best_cost = cost
                best_j = j

        if best_j is None:
            # all routes infeasible (inf) -> go depot and retry
            at_depot = False  # force depot return on next loop
            continue

        chosen = feasible[int(best_j)]
        leg_dist_m = float(dist_m[int(best_j)])
        leg_dur_s = float(dur_s[int(best_j)])
        leg_dur_tr_s = float(dur_tr[int(best_j)]) if (use_traffic and dur_tr is not None) else None

        # Move
        total_dist_m += leg_dist_m
        total_dur_s += leg_dur_s
        if leg_dur_tr_s is not None:
            total_dur_tr_s += leg_dur_tr_s

        cur_lat, cur_lon = chosen.lat, chosen.lon
        at_depot = False

        # Update load
        if chosen.kind.lower() == "delivery":
            delivery_stock -= chosen.pkgs
        else:
            pickup_load += chosen.pkgs

        log_step("STOP", chosen, leg_dist_m, leg_dur_s, leg_dur_tr_s, pickup_load, delivery_stock)

        # Remove stop
        remaining.pop(chosen.stop_id, None)

        # If delivery stock is near empty and deliveries remain, consider depot soon (optional)
        # We keep it simple; depot return happens automatically when no feasible deliveries remain.

    # return to depot (end)
    if not at_depot:
        dist_m, dur_s, dur_tr = dm_one_to_many(
            gmaps_client,
            origin_lat=cur_lat, origin_lon=cur_lon,
            dest_latlons=[(depot_lat, depot_lon)],
            mode=mode, use_traffic=use_traffic, chunk_size=chunk_size, pause_s=pause_s
        )
        leg_dist_m = float(dist_m[0])
        leg_dur_s = float(dur_s[0])
        leg_dur_tr_s = float(dur_tr[0]) if (use_traffic and dur_tr is not None) else None

        total_dist_m += leg_dist_m
        total_dur_s += leg_dur_s
        if leg_dur_tr_s is not None:
            total_dur_tr_s += leg_dur_tr_s

        cur_lat, cur_lon = depot_lat, depot_lon
        at_depot = True
        log_step("DEPOT_END", None, leg_dist_m, leg_dur_s, leg_dur_tr_s, pickup_load, delivery_stock, note="end")

    totals = {
        "total_distance_km": total_dist_m/1000.0,
        "total_duration_min": total_dur_s/60.0,
        "total_duration_in_traffic_min": total_dur_tr_s/60.0 if use_traffic else None,
    }
    return pd.DataFrame(route_rows), totals


# ------------------------- Plotting -------------------------

def plot_clusters(df_day: pd.DataFrame, depot_lat: float, depot_lon: float, out_png: Path):
    plt.figure()
    for cid, sub in df_day.groupby("cluster_id"):
        plt.scatter(sub["Longitude"], sub["Latitude"], s=10, label=f"c{cid}")
    plt.scatter([depot_lon], [depot_lat], marker="x", s=80, label="DEPOT")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("KMeans clusters (rebalanced sizes)")
    plt.legend(loc="best", ncol=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_route_samples(routes_df: pd.DataFrame, depot_lat: float, depot_lon: float, out_png: Path, max_clusters: int = 3):
    """
    Plot routes for first few clusters (so you don't generate 40+ images).
    """
    plt.figure()
    plt.scatter([depot_lon], [depot_lat], marker="x", s=80, label="DEPOT")

    shown = 0
    for cid in sorted(routes_df["cluster_id"].dropna().unique()):
        sub = routes_df[routes_df["cluster_id"] == cid].sort_values("seq")
        # take points in order
        lons = []
        lats = []
        for _, r in sub.iterrows():
            if r["node"] in ("DEPOT_START", "DEPOT", "DEPOT_END"):
                lons.append(depot_lon)
                lats.append(depot_lat)
            else:
                lons.append(float(r["Longitude"]))
                lats.append(float(r["Latitude"]))
        plt.plot(lons, lats, linewidth=1, label=f"route c{cid}")
        shown += 1
        if shown >= max_clusters:
            break

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Route samples (first clusters)")
    plt.legend(loc="best", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ------------------------- Optional: sync outputs to report/assets -------------------------

def sync_outputs_to_report(
    outdir: Path,
    report_dir: Path,
    excel_split_to_csv: bool = True,
    keep_excel: bool = True,
    patterns: tuple[str, ...] = ("*",),
) -> list[Path]:
    """
    Copy outputs into rapor/assets/figs|data|tables|misc and optionally split xlsx sheets to CSV.

    outdir:
      where this script writes outputs
    report_dir:
      e.g., grad/rapor

    Returns list of created/copied paths in report.
    """
    import shutil

    assets = report_dir / "assets"
    figs = assets / "figs"
    data = assets / "data"
    tables = assets / "tables"
    misc = assets / "misc"
    for p in (figs, data, tables, misc):
        p.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []

    def dest_for(file: Path) -> Path:
        suf = file.suffix.lower()
        if suf in [".png", ".jpg", ".jpeg", ".pdf", ".svg", ".webp"]:
            return figs / safe_filename(file.name)
        if suf in [".csv", ".tsv", ".txt"]:
            return data / safe_filename(file.name)
        if suf in [".tex"]:
            return tables / safe_filename(file.name)
        if suf in [".xlsx", ".xls", ".ods"]:
            return data / safe_filename(file.name)
        return misc / safe_filename(file.name)

    files = []
    for pat in patterns:
        files.extend(outdir.glob(pat))

    for f in files:
        if not f.is_file():
            continue
        if f.suffix.lower() in [".xlsx", ".xls", ".ods"] and excel_split_to_csv:
            # Split sheets to CSV into <stem>_sheets/
            try:
                xls = pd.ExcelFile(f)
                out_sheets = data / (safe_filename(f.stem) + "_sheets")
                out_sheets.mkdir(parents=True, exist_ok=True)

                for sheet in xls.sheet_names:
                    sdf = pd.read_excel(f, sheet_name=sheet)
                    out_file = out_sheets / (safe_filename(sheet) + ".csv")
                    sdf.to_csv(out_file, index=False, encoding="utf-8-sig")
                    created.append(out_file)

                if keep_excel:
                    dst = dest_for(f)
                    shutil.copy2(f, dst)
                    created.append(dst)
            except Exception as e:
                # fallback: just copy excel
                dst = dest_for(f)
                shutil.copy2(f, dst)
                created.append(dst)
            continue

        dst = dest_for(f)
        shutil.copy2(f, dst)
        created.append(dst)

    return created


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="CSV path (default: ./data_cleaned.csv if exists)")
    ap.add_argument("--outdir", default="vrp_trial_outputs", help="Output folder")
    ap.add_argument("--days", default="Mon", help='Day filter: "Mon" or "Mon,Tue" or "all"')
    ap.add_argument("--min_stops", type=int, default=80, help="Min stops per vehicle/cluster")
    ap.add_argument("--max_stops", type=int, default=110, help="Max stops per vehicle/cluster")
    ap.add_argument("--k", type=int, default=None, help="Manual cluster count (optional)")
    ap.add_argument("--cap_pkgs", type=float, default=350.0, help="Vehicle capacity (packages)")
    ap.add_argument("--split_overcap_stops", action="store_true", help="Split stops with pkgs>cap into pseudo-stops")
    ap.add_argument("--mode", default="driving", help="Google mode: driving/walking/bicycling/transit")
    ap.add_argument("--use_traffic", action="store_true", help="Use duration_in_traffic (driving only)")
    ap.add_argument("--candidate_limit", type=int, default=25, help="Greedy candidate pool size (<=25 recommended)")
    ap.add_argument("--chunk_size", type=int, default=25, help="DM chunk size (<=25 to avoid MAX_DIMENSIONS_EXCEEDED)")
    ap.add_argument("--pause_s", type=float, default=0.10, help="Pause between DM calls to be gentle on quota")
    ap.add_argument("--alpha_load", type=float, default=0.0, help="Soft penalty for load balancing (0 disables)")
    ap.add_argument("--sample_n", type=int, default=10, help="Sample size for from-to chart")
    ap.add_argument("--plot_route_samples", action="store_true", help="Also plot route samples for first clusters")
    ap.add_argument("--route_plot_clusters", type=int, default=3, help="How many clusters to draw in route sample plot")
    ap.add_argument("--ticker_s", type=int, default=10, help="Print running... every N seconds")

    # depot (Hacettepe/UPS Anadolu aktarma center provided earlier)
    ap.add_argument("--depot_lat", type=float, default=40.920153)
    ap.add_argument("--depot_lon", type=float, default=29.348245)

    # optional sync-to-report
    ap.add_argument("--sync_to_report", action="store_true", help="Copy outputs to rapor/assets/*")
    ap.add_argument("--report_dir", default=None, help="Rapor folder (default: ../rapor relative to script)")
    ap.add_argument("--xlsx_to_csv", action="store_true", help="When syncing, split excel sheets into CSV")
    ap.add_argument("--drop_excel", action="store_true", help="When syncing and xlsx_to_csv, don't copy the excel itself")

    args = ap.parse_args()

    # Resolve default input path
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1] if len(script_path.parents) >= 2 else Path.cwd()
    default_csv = project_root / "data_cleaned.csv"
    input_path = Path(args.input) if args.input else (default_csv if default_csv.exists() else Path("data_cleaned.csv"))
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}. Use --input to point to your CSV.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Google Maps client
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_MAPS_API_KEY not set in environment.")
    gmaps_client = googlemaps.Client(key=api_key)

    ticker = ProgressTicker(interval=args.ticker_s, msg="running...")
    ticker.start()

    try:
        df = read_csv_flexible(input_path)
        df = ensure_columns(df)

        # day selection
        if args.days.strip().lower() == "all":
            day_list = sorted(df["Day"].unique().tolist())
        else:
            day_list = [d.strip() for d in args.days.split(",") if d.strip()]

        for day in day_list:
            df_day = df[df["Day"].astype(str).str.strip().str.lower() == day.lower()].copy()
            if len(df_day) == 0:
                print(f"[WARN] day={day}: no rows found", flush=True)
                continue

            n_stops = len(df_day)
            if args.k is None:
                k_min, k_max = feasible_k_range(n_stops, args.min_stops, args.max_stops)
                k_used = choose_k_auto(n_stops, args.min_stops, args.max_stops)
                print(f"[INFO] day={day} | n_stops={n_stops} | feasible_k=[{k_min},{k_max}] | k_used={k_used} (auto)", flush=True)
            else:
                k_used = int(args.k)
                print(f"[INFO] day={day} | n_stops={n_stops} | k_used={k_used} (manual)", flush=True)

            # KMeans + rebalance
            labels0 = kmeans_labels(df_day, k_used)
            labels = rebalance_cluster_sizes(df_day, labels0, args.min_stops, args.max_stops)
            df_day["cluster_id"] = labels

            sizes = df_day["cluster_id"].value_counts().sort_index()
            size_str = ", ".join([f"{i}:{int(sizes[i])}" for i in sizes.index])
            print(f"[INFO] day={day} | cluster_sizes (cluster:count) -> {size_str}", flush=True)

            # export clustered
            clustered_csv = outdir / f"clustered_{safe_filename(day)}.csv"
            df_day.to_csv(clustered_csv, index=False, encoding="utf-8-sig")

            # from-to chart sample (DEPOT + first sample_n)
            sample_df = df_day.head(min(args.sample_n, len(df_day))).copy()
            ft = build_from_to_google_sample(
                gmaps_client,
                depot_lat=args.depot_lat,
                depot_lon=args.depot_lon,
                df_points=sample_df,
                mode=args.mode,
                use_traffic=args.use_traffic,
                chunk_size=args.chunk_size,
                pause_s=args.pause_s,
            )
            ft_xlsx = outdir / f"from_to_google_sample{len(sample_df)}_{safe_filename(day)}.xlsx"
            with pd.ExcelWriter(ft_xlsx, engine="openpyxl") as w:
                for sheet, sdf in ft.items():
                    sdf.to_excel(w, sheet_name=sheet)

            # plot clusters
            cluster_png = outdir / f"cluster_plot_{safe_filename(day)}.png"
            plot_clusters(df_day, args.depot_lat, args.depot_lon, cluster_png)

            # route per cluster
            all_routes = []
            summary_rows = []

            for cid in sorted(df_day["cluster_id"].unique()):
                cdf = df_day[df_day["cluster_id"] == cid].copy().reset_index(drop=True)
                if args.split_overcap_stops:
                    cdf = split_overcap_stops(cdf, args.cap_pkgs)

                # add StopID if not present
                if "StopID" not in cdf.columns:
                    cdf["StopID"] = cdf["StopNum"].astype(str)

                n_points = len(cdf) + 1
                approx_calls = len(cdf)  # roughly one call per step
                print(f"[INFO] day={day} | processing cluster={cid} | n_points(incl depot)={n_points} | approx_DM_calls~{approx_calls}", flush=True)

                routes_df, totals = greedy_route_with_depot_returns(
                    cdf=cdf,
                    depot_lat=args.depot_lat,
                    depot_lon=args.depot_lon,
                    gmaps_client=gmaps_client,
                    cap_pkgs=args.cap_pkgs,
                    mode=args.mode,
                    use_traffic=args.use_traffic,
                    candidate_limit=args.candidate_limit,
                    chunk_size=args.chunk_size,
                    pause_s=args.pause_s,
                    alpha_load=args.alpha_load,
                )
                routes_df.insert(0, "cluster_id", cid)
                routes_df.insert(0, "day", day)
                all_routes.append(routes_df)

                summary_rows.append({
                    "day": day,
                    "cluster_id": cid,
                    "n_stops": int(len(cdf)),
                    "total_distance_km": totals["total_distance_km"],
                    "total_duration_min": totals["total_duration_min"],
                    "total_duration_in_traffic_min": totals["total_duration_in_traffic_min"],
                })

            routes_all = pd.concat(all_routes, ignore_index=True) if all_routes else pd.DataFrame()
            summary_df = pd.DataFrame(summary_rows).sort_values(["day", "cluster_id"])

            routes_csv = outdir / f"routes_{safe_filename(day)}.csv"
            summary_csv = outdir / f"routes_summary_{safe_filename(day)}.csv"
            routes_all.to_csv(routes_csv, index=False, encoding="utf-8-sig")
            summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

            if args.plot_route_samples and not routes_all.empty:
                route_png = outdir / f"route_plot_sample_{safe_filename(day)}.png"
                plot_route_samples(routes_all, args.depot_lat, args.depot_lon, route_png, max_clusters=args.route_plot_clusters)

            print("[INFO] outputs:", flush=True)
            print(" -", clustered_csv, flush=True)
            print(" -", routes_csv, flush=True)
            print(" -", summary_csv, flush=True)
            print(" -", ft_xlsx, flush=True)
            print(" -", cluster_png, flush=True)
            if args.plot_route_samples:
                print(" -", outdir / f"route_plot_sample_{safe_filename(day)}.png", flush=True)

        # optional sync to report
        if args.sync_to_report:
            report_dir = Path(args.report_dir) if args.report_dir else (project_root / "rapor")
            created = sync_outputs_to_report(
                outdir=outdir,
                report_dir=report_dir,
                excel_split_to_csv=args.xlsx_to_csv,
                keep_excel=(not args.drop_excel),
                patterns=("*",),
            )
            print(f"[INFO] synced_to_report: {len(created)} item(s)", flush=True)

    finally:
        ticker.stop()


if __name__ == "__main__":
    main()
