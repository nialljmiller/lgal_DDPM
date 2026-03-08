#!/usr/bin/env python3
"""
build_window_catalogue.py

Reads per-bird CSVs from golden_eai_outputs/per_bird_csv/ and produces:
  - window_catalogue.csv   : one row per bird per 1-hour window, ML-ready features
  - bird_catalogue_enhanced.csv : bird-level summaries derived from those windows

Usage:
    python build_window_catalogue.py \
        --per_bird_dir golden_eai_outputs/per_bird_csv \
        --bird_catalogue golden_eai_outputs/bird_catalogue.csv \
        --outdir golden_eai_outputs \
        [--window_hours 1] \
        [--chunksize 200000]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Optional StochiStats import — used for variability indices on speed / altitude
# ---------------------------------------------------------------------------
try:
    import stochistats as ss
    from stochistats import (
        Stetson_K, Eta, MedianAbsDev, Meanvariance,
        RoMS, lagauto, AndersonDarling, IQR,
        skewness, kurtosis,
    )
    STOCHI = True
except ImportError:
    STOCHI = False
    print("[warn] stochistats not found — variability indices will be skipped.")
    print("       Install: pip install git+https://github.com/nialljmiller/StochiStats.git")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0088
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2.0 * np.arcsin(np.sqrt(a))


def circular_mean_deg(angles_deg):
    r = np.radians(angles_deg)
    return np.degrees(np.arctan2(np.nanmean(np.sin(r)), np.nanmean(np.cos(r)))) % 360


def circular_std_deg(angles_deg):
    """Circular standard deviation in degrees."""
    r = np.radians(angles_deg)
    R = np.sqrt(np.nanmean(np.sin(r)) ** 2 + np.nanmean(np.cos(r)) ** 2)
    R = np.clip(R, 0, 1)
    return np.degrees(np.sqrt(-2 * np.log(R)))


def circular_heading_diff(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    diff = (h2 - h1 + 180.0) % 360.0 - 180.0
    return np.abs(diff)


def radius_of_gyration(lons, lats):
    """RMS distance from centroid, in km."""
    if len(lons) < 2:
        return np.nan
    clat = np.nanmean(lats)
    clon = np.nanmean(lons)
    d = haversine_km(lons, lats, clon, clat)
    return np.sqrt(np.nanmean(d ** 2))


def path_efficiency(total_km, net_km):
    if total_km > 0:
        return net_km / total_km
    return np.nan


def safe_stat(func, arr, *args, fallback=np.nan, **kwargs):
    try:
        valid = arr[np.isfinite(arr)]
        if len(valid) < 2:
            return fallback
        return float(func(valid, *args, **kwargs))
    except Exception:
        return fallback


def quantile(arr, q):
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return np.nan
    return float(np.nanpercentile(valid, q))


# ---------------------------------------------------------------------------
# Per-window feature extraction
# ---------------------------------------------------------------------------

def extract_window_features(w: pd.DataFrame, bird_id: str, window_start: pd.Timestamp) -> dict:
    """
    Compute all features for a single 1-hour window.
    w: DataFrame subset already filtered to this window, sorted by timestamp.
    """
    feat = {
        "bird_id": bird_id,
        "window_start": window_start,
        "window_hour_utc": window_start.hour,
        "window_doy": window_start.dayofyear,
    }

    n = len(w)
    feat["n_fixes"] = n

    # ---- Time / cadence ----
    t = pd.to_datetime(w["timestamp"], errors="coerce", utc=True).sort_values()
    valid_t = t.dropna()
    feat["n_valid_timestamps"] = len(valid_t)

    if len(valid_t) >= 2:
        dt_s = valid_t.diff().dt.total_seconds().dropna().values
        feat["window_actual_duration_s"] = float((valid_t.max() - valid_t.min()).total_seconds())
        feat["timestep_median_s"] = float(np.nanmedian(dt_s))
        feat["timestep_max_s"] = float(np.nanmax(dt_s))
        feat["timestep_mean_s"] = float(np.nanmean(dt_s))
        feat["timestep_std_s"] = float(np.nanstd(dt_s))
        # Burstiness: (std - mean) / (std + mean)
        m, s = feat["timestep_mean_s"], feat["timestep_std_s"]
        feat["timestep_burstiness"] = (s - m) / (s + m) if (s + m) > 0 else np.nan
        # Fraction of large gaps (> 2× median)
        thresh = feat["timestep_median_s"] * 2
        feat["frac_large_gaps"] = float(np.mean(dt_s > thresh)) if len(dt_s) > 0 else np.nan
    else:
        for k in [
            "window_actual_duration_s", "timestep_median_s", "timestep_max_s",
            "timestep_mean_s", "timestep_std_s", "timestep_burstiness", "frac_large_gaps",
        ]:
            feat[k] = np.nan

    # ---- GPS validity ----
    lon = pd.to_numeric(w["location-long"], errors="coerce").values
    lat = pd.to_numeric(w["location-lat"], errors="coerce").values
    gps_ok = np.isfinite(lon) & np.isfinite(lat)
    feat["n_valid_gps"] = int(gps_ok.sum())
    feat["frac_valid_gps"] = float(gps_ok.mean())

    if gps_ok.sum() >= 2:
        vlon = lon[gps_ok]
        vlat = lat[gps_ok]

        # ---- Location / space ----
        feat["mean_lon"] = float(np.nanmean(vlon))
        feat["mean_lat"] = float(np.nanmean(vlat))
        feat["lon_std"] = float(np.nanstd(vlon))
        feat["lat_std"] = float(np.nanstd(vlat))
        feat["lon_range"] = float(np.nanmax(vlon) - np.nanmin(vlon))
        feat["lat_range"] = float(np.nanmax(vlat) - np.nanmin(vlat))
        feat["bbox_area_deg2"] = feat["lon_range"] * feat["lat_range"]
        feat["radius_of_gyration_km"] = radius_of_gyration(vlon, vlat)

        # Displacement: first → last GPS fix
        net_km = float(haversine_km(vlon[0], vlat[0], vlon[-1], vlat[-1]))
        feat["net_displacement_km"] = net_km

        # Step-by-step path
        step_km = haversine_km(vlon[:-1], vlat[:-1], vlon[1:], vlat[1:])
        feat["total_path_km"] = float(np.nansum(step_km))
        feat["path_efficiency"] = path_efficiency(feat["total_path_km"], net_km)
        feat["step_km_mean"] = float(np.nanmean(step_km))
        feat["step_km_median"] = float(np.nanmedian(step_km))
        feat["step_km_std"] = float(np.nanstd(step_km))
        feat["step_km_max"] = float(np.nanmax(step_km))
        feat["step_km_q75"] = quantile(step_km, 75)
        feat["step_km_q95"] = quantile(step_km, 95)
        feat["frac_zero_steps"] = float(np.mean(step_km < 0.01))

        # Step acceleration proxy: second diff of cumulative path
        if len(step_km) >= 2:
            step_accel = np.abs(np.diff(step_km))
            feat["step_accel_mean"] = float(np.nanmean(step_accel))
            feat["step_accel_max"] = float(np.nanmax(step_accel))
        else:
            feat["step_accel_mean"] = np.nan
            feat["step_accel_max"] = np.nan

    else:
        for k in [
            "mean_lon", "mean_lat", "lon_std", "lat_std", "lon_range", "lat_range",
            "bbox_area_deg2", "radius_of_gyration_km", "net_displacement_km",
            "total_path_km", "path_efficiency", "step_km_mean", "step_km_median",
            "step_km_std", "step_km_max", "step_km_q75", "step_km_q95", "frac_zero_steps",
            "step_accel_mean", "step_accel_max",
        ]:
            feat[k] = np.nan

    # ---- Ground speed ----
    if "ground-speed" in w.columns:
        gs = pd.to_numeric(w["ground-speed"], errors="coerce").values
        gs_v = gs[np.isfinite(gs)]
        if len(gs_v) >= 2:
            feat["speed_mean"] = float(np.nanmean(gs_v))
            feat["speed_median"] = float(np.nanmedian(gs_v))
            feat["speed_std"] = float(np.nanstd(gs_v))
            feat["speed_max"] = float(np.nanmax(gs_v))
            feat["speed_q25"] = quantile(gs_v, 25)
            feat["speed_q75"] = quantile(gs_v, 75)
            feat["speed_q95"] = quantile(gs_v, 95)
            feat["frac_low_speed"] = float(np.mean(gs_v < 1.0))   # < 1 m/s
            feat["frac_high_speed"] = float(np.mean(gs_v > 10.0)) # > 10 m/s
            if STOCHI and len(gs_v) >= 4:
                t_arr = np.arange(len(gs_v), dtype=float)
                e_arr = np.ones_like(gs_v) * 0.1
                feat["speed_stetson_k"] = safe_stat(Stetson_K, gs_v, e_arr)
                feat["speed_eta"] = safe_stat(Eta, gs_v, t_arr)
                feat["speed_mad"] = safe_stat(MedianAbsDev, gs_v, e_arr)
                feat["speed_iqr"] = safe_stat(IQR, gs_v)
                feat["speed_skew"] = safe_stat(skewness, gs_v)
                feat["speed_kurt"] = safe_stat(kurtosis, gs_v)
        else:
            for k in [
                "speed_mean", "speed_median", "speed_std", "speed_max",
                "speed_q25", "speed_q75", "speed_q95", "frac_low_speed", "frac_high_speed",
            ]:
                feat[k] = np.nan

    # ---- Heading / turning ----
    if "heading" in w.columns:
        hd = pd.to_numeric(w["heading"], errors="coerce").values
        hd_v = hd[np.isfinite(hd)]
        if len(hd_v) >= 2:
            feat["heading_circular_mean_deg"] = circular_mean_deg(hd_v)
            feat["heading_circular_std_deg"] = circular_std_deg(hd_v)
            # Turn angles between consecutive fixes
            turn = circular_heading_diff(hd_v[:-1], hd_v[1:])
            feat["turn_mean_deg"] = float(np.nanmean(turn))
            feat["turn_median_deg"] = float(np.nanmedian(turn))
            feat["turn_std_deg"] = float(np.nanstd(turn))
            feat["turn_max_deg"] = float(np.nanmax(turn))
            feat["frac_sharp_turns"] = float(np.mean(turn > 90))
            feat["frac_straight"] = float(np.mean(turn < 15))
            # Heading concentration (mean resultant length R)
            r_rad = np.radians(hd_v)
            R = np.sqrt(np.nanmean(np.sin(r_rad)) ** 2 + np.nanmean(np.cos(r_rad)) ** 2)
            feat["heading_concentration"] = float(R)  # 1=perfectly directed, 0=random
            # Tortuosity: total path / net displacement
            if feat.get("total_path_km", 0) > 0 and feat.get("net_displacement_km", 0) > 0:
                feat["tortuosity"] = feat["total_path_km"] / max(feat["net_displacement_km"], 1e-6)
            else:
                feat["tortuosity"] = np.nan
        else:
            for k in [
                "heading_circular_mean_deg", "heading_circular_std_deg",
                "turn_mean_deg", "turn_median_deg", "turn_std_deg", "turn_max_deg",
                "frac_sharp_turns", "frac_straight", "heading_concentration", "tortuosity",
            ]:
                feat[k] = np.nan

    # ---- Altitude ----
    if "height-above-ellipsoid" in w.columns:
        alt = pd.to_numeric(w["height-above-ellipsoid"], errors="coerce").values
        alt_v = alt[np.isfinite(alt)]
        if len(alt_v) >= 2:
            feat["alt_mean"] = float(np.nanmean(alt_v))
            feat["alt_median"] = float(np.nanmedian(alt_v))
            feat["alt_std"] = float(np.nanstd(alt_v))
            feat["alt_min"] = float(np.nanmin(alt_v))
            feat["alt_max"] = float(np.nanmax(alt_v))
            feat["alt_range"] = feat["alt_max"] - feat["alt_min"]
            # Vertical rates
            if len(alt_v) >= 3 and feat.get("timestep_median_s", np.nan) > 0:
                dalt = np.diff(alt_v)
                dt = feat["timestep_median_s"]  # approximate
                vrate = dalt / dt  # m/s vertical
                feat["vrate_mean"] = float(np.nanmean(vrate))
                feat["vrate_std"] = float(np.nanstd(vrate))
                feat["vrate_max_climb"] = float(np.nanmax(vrate))
                feat["vrate_max_descent"] = float(np.nanmin(vrate))
                feat["frac_climbing"] = float(np.mean(vrate > 0.1))
                feat["frac_descending"] = float(np.mean(vrate < -0.1))
                feat["frac_level"] = float(np.mean(np.abs(vrate) <= 0.1))
            else:
                for k in ["vrate_mean", "vrate_std", "vrate_max_climb", "vrate_max_descent",
                          "frac_climbing", "frac_descending", "frac_level"]:
                    feat[k] = np.nan
            if STOCHI and len(alt_v) >= 4:
                t_arr = np.arange(len(alt_v), dtype=float)
                e_arr = np.ones_like(alt_v) * 1.0
                feat["alt_eta"] = safe_stat(Eta, alt_v, t_arr)
                feat["alt_mad"] = safe_stat(MedianAbsDev, alt_v, e_arr)
        else:
            for k in [
                "alt_mean", "alt_median", "alt_std", "alt_min", "alt_max", "alt_range",
                "vrate_mean", "vrate_std", "vrate_max_climb", "vrate_max_descent",
                "frac_climbing", "frac_descending", "frac_level",
            ]:
                feat[k] = np.nan

    # ---- GPS quality ----
    if "gps:dop" in w.columns:
        dop = pd.to_numeric(w["gps:dop"], errors="coerce").values
        feat["dop_mean"] = float(np.nanmean(dop[np.isfinite(dop)])) if np.isfinite(dop).any() else np.nan
    if "gps:satellite-count" in w.columns:
        sats = pd.to_numeric(w["gps:satellite-count"], errors="coerce").values
        feat["sat_count_mean"] = float(np.nanmean(sats[np.isfinite(sats)])) if np.isfinite(sats).any() else np.nan
    if "eobs:horizontal-accuracy-estimate" in w.columns:
        hacc = pd.to_numeric(w["eobs:horizontal-accuracy-estimate"], errors="coerce").values
        feat["h_acc_mean"] = float(np.nanmean(hacc[np.isfinite(hacc)])) if np.isfinite(hacc).any() else np.nan

    # ---- Device / environment ----
    if "eobs:temperature" in w.columns:
        dt = pd.to_numeric(w["eobs:temperature"], errors="coerce").values
        feat["device_temp_mean"] = float(np.nanmean(dt[np.isfinite(dt)])) if np.isfinite(dt).any() else np.nan
    if "external-temperature" in w.columns:
        et = pd.to_numeric(w["external-temperature"], errors="coerce").values
        feat["ext_temp_mean"] = float(np.nanmean(et[np.isfinite(et)])) if np.isfinite(et).any() else np.nan
    if "eobs:battery-voltage" in w.columns:
        bv = pd.to_numeric(w["eobs:battery-voltage"], errors="coerce").values
        feat["battery_mean"] = float(np.nanmean(bv[np.isfinite(bv)])) if np.isfinite(bv).any() else np.nan

    return feat


# ---------------------------------------------------------------------------
# Per-bird windowing
# ---------------------------------------------------------------------------

def process_bird(bird_csv: Path, window_hours: int = 1, chunksize: int = 200_000) -> list[dict]:
    """
    Read a per-bird CSV in chunks, assign 1-hour window bins, and compute features.
    Returns a list of feature dicts (one per window).
    """
    rows = []

    # Accumulator for the current window across chunks
    pending = None

    # We must read the full file but can do it in chunks for large birds
    chunks = []
    for chunk in pd.read_csv(bird_csv, chunksize=chunksize, low_memory=False):
        if "timestamp" not in chunk.columns:
            continue
        chunk["_t"] = pd.to_datetime(chunk["timestamp"], errors="coerce", utc=True)
        chunk = chunk.dropna(subset=["_t"]).sort_values("_t")
        chunks.append(chunk)

    if not chunks:
        return []

    df = pd.concat(chunks, ignore_index=True).sort_values("_t")

    bird_id = str(df["individual-local-identifier"].dropna().iloc[0]) \
        if "individual-local-identifier" in df.columns and df["individual-local-identifier"].notna().any() \
        else bird_csv.stem

    # Floor each timestamp to the window boundary
    df["_window"] = df["_t"].dt.floor(f"{window_hours}h")

    for window_start, group in df.groupby("_window", sort=True):
        if len(group) < 2:
            continue
        feat = extract_window_features(group.sort_values("_t"), bird_id, window_start)
        rows.append(feat)

    return rows


# ---------------------------------------------------------------------------
# Bird-level summary from windows
# ---------------------------------------------------------------------------

def summarize_bird_from_windows(bird_windows: pd.DataFrame) -> dict:
    """One-row summary per bird from their window rows."""
    bw = bird_windows.copy()
    out = {"bird_id": bw["bird_id"].iloc[0]}

    out["n_windows"] = len(bw)
    out["n_windows_with_gps"] = int((bw["n_valid_gps"] >= 2).sum())

    numeric = bw.select_dtypes(include=[np.number])

    # Mean of per-window means
    for col in ["speed_mean", "speed_max", "turn_mean_deg", "alt_mean", "alt_range",
                "heading_concentration", "tortuosity", "path_efficiency",
                "net_displacement_km", "total_path_km", "radius_of_gyration_km",
                "frac_low_speed", "frac_high_speed", "frac_sharp_turns", "frac_straight",
                "frac_climbing", "frac_descending", "frac_level",
                "dop_mean", "sat_count_mean", "battery_mean", "ext_temp_mean"]:
        if col in bw.columns:
            out[f"bird_{col}_mean"] = float(bw[col].mean(skipna=True))
            out[f"bird_{col}_std"] = float(bw[col].std(skipna=True))

    # Daily pattern: mean speed per hour of day
    if "speed_mean" in bw.columns and "window_hour_utc" in bw.columns:
        hourly = bw.groupby("window_hour_utc")["speed_mean"].mean()
        if len(hourly) >= 6:
            # Amplitude of diel cycle (simple range)
            out["bird_diel_speed_amplitude"] = float(hourly.max() - hourly.min())
            # Hour of peak activity
            out["bird_peak_activity_hour"] = int(hourly.idxmax())

    # Fraction of windows in each rough behavioural regime
    if "speed_mean" in bw.columns:
        out["bird_frac_stationary_windows"] = float((bw["speed_mean"] < 0.5).mean())
        out["bird_frac_active_windows"] = float((bw["speed_mean"] >= 2.0).mean())
        out["bird_frac_fast_windows"] = float((bw["speed_mean"] >= 8.0).mean())

    # Seasonal coverage
    if "window_doy" in bw.columns:
        out["bird_doy_range"] = float(bw["window_doy"].max() - bw["window_doy"].min())
        out["bird_doy_first"] = float(bw["window_doy"].min())
        out["bird_doy_last"] = float(bw["window_doy"].max())

    # Total path
    if "total_path_km" in bw.columns:
        out["bird_total_path_km"] = float(bw["total_path_km"].sum(skipna=True))

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build windowed feature catalogue for golden eagle data.")
    parser.add_argument("--per_bird_dir", default="golden_eai_outputs/per_bird_csv",
                        help="Directory of per-bird CSVs")
    parser.add_argument("--bird_catalogue", default="golden_eai_outputs/bird_catalogue.csv",
                        help="Existing bird_catalogue.csv (merged in for metadata columns)")
    parser.add_argument("--outdir", default="golden_eai_outputs",
                        help="Output directory")
    parser.add_argument("--window_hours", type=int, default=1,
                        help="Window size in hours (default 1)")
    parser.add_argument("--chunksize", type=int, default=200_000,
                        help="Chunk size for reading large CSVs")
    args = parser.parse_args()

    per_bird_dir = Path(args.per_bird_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    window_csv = outdir / f"window_catalogue_{args.window_hours}h.csv"
    enhanced_csv = outdir / "bird_catalogue_enhanced.csv"

    bird_files = sorted(per_bird_dir.glob("*.csv"))
    print(f"Found {len(bird_files)} bird files")
    print(f"Window size: {args.window_hours} hour(s)")
    print(f"StochiStats available: {STOCHI}")

    # Before the bird loop
    all_columns = set()

    # First pass: gather rows in memory OR gather schemas
    all_bird_rows = []

    for bird_csv in tqdm(bird_files, desc="Processing birds"):
        try:
            bird_rows = process_bird(bird_csv, window_hours=args.window_hours, chunksize=args.chunksize)
        except Exception as e:
            print(f"[error] {bird_csv.name}: {e}")
            continue

        if not bird_rows:
            continue

        all_bird_rows.append(bird_rows)
        for row in bird_rows:
            all_columns.update(row.keys())

    all_columns = sorted(all_columns)

    # Then write with fixed schema
    first_write = True
    all_windows = []

    for bird_rows in all_bird_rows:
        chunk_df = pd.DataFrame(bird_rows)
        chunk_df = chunk_df.reindex(columns=all_columns)
        chunk_df.to_csv(
            window_csv,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )
        first_write = False
        all_windows.append(chunk_df)
        
    if not all_windows:
        print("No windows produced. Check input data.")
        return

    windows = pd.concat(all_windows, ignore_index=True)
    print(f"\nTotal windows: {len(windows)}")
    print(f"Wrote: {window_csv}")

    # ---- Bird-level enhanced catalogue ----
    print("Building bird-level enhanced catalogue...")
    bird_summaries = []
    for bird_id, bw in windows.groupby("bird_id"):
        bird_summaries.append(summarize_bird_from_windows(bw))

    enhanced = pd.DataFrame(bird_summaries)

    # Merge with existing bird_catalogue if available
    if Path(args.bird_catalogue).exists():
        base_cat = pd.read_csv(args.bird_catalogue, low_memory=False)
        if "individual-local-identifier" in base_cat.columns:
            base_cat["individual-local-identifier"] = base_cat["individual-local-identifier"].astype(str)
            enhanced["bird_id"] = enhanced["bird_id"].astype(str)
            enhanced = enhanced.merge(base_cat, how="left",
                                      left_on="bird_id",
                                      right_on="individual-local-identifier")

    enhanced.to_csv(enhanced_csv, index=False)
    print(f"Wrote: {enhanced_csv}")
    print(f"Enhanced catalogue rows: {len(enhanced)}")
    print("\nDone.")


if __name__ == "__main__":
    main()