#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def safe_filename(value: str) -> str:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        text = "UNKNOWN_BIRD"
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def datetime_to_jd(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    mask = dt.notna()
    if mask.any():
        unix_seconds = dt[mask].astype("int64") / 1e9
        out.loc[mask] = unix_seconds / 86400.0 + 2440587.5
    return out


def detect_datetime_columns_from_header(columns: list[str]) -> list[str]:
    """
    Use header names only, so schema is fixed for the whole run.
    """
    dt_cols = []
    for col in columns:
        name = col.lower()
        if "time" in name or "date" in name:
            dt_cols.append(col)
    return dt_cols


def process_metadata(meta_csv: Path, output_meta_csv: Path) -> pd.DataFrame:
    meta = pd.read_csv(meta_csv, low_memory=False)

    dt_cols = detect_datetime_columns_from_header(list(meta.columns))
    for col in dt_cols:
        meta[f"{col}_jd"] = datetime_to_jd(meta[col])

    meta.to_csv(output_meta_csv, index=False)
    return meta


def split_main_csv_by_bird_fixed_schema(
    input_csv: Path,
    out_dir: Path,
    chunksize: int = 200_000,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read header only once
    header_df = pd.read_csv(input_csv, nrows=0)
    base_columns = list(header_df.columns)

    if "individual-local-identifier" not in base_columns:
        raise KeyError("Expected column 'individual-local-identifier' not found in input CSV.")

    # Fixed datetime columns based only on header
    dt_cols = detect_datetime_columns_from_header(base_columns)

    # Fixed output schema for all chunks
    jd_columns = [f"{col}_jd" for col in dt_cols]
    output_columns = base_columns + jd_columns

    print(f"Detected {len(base_columns)} base columns")
    print(f"Adding {len(jd_columns)} JD columns")
    print(f"Total output columns: {len(output_columns)}")

    seen_birds = set()

    for i, chunk in enumerate(pd.read_csv(input_csv, chunksize=chunksize, low_memory=False), start=1):
        # Add fixed JD columns for this chunk
        for col in dt_cols:
            chunk[f"{col}_jd"] = datetime_to_jd(chunk[col]) if col in chunk.columns else np.nan

        # Ensure all expected columns exist
        for col in output_columns:
            if col not in chunk.columns:
                chunk[col] = np.nan

        # Enforce fixed column order
        chunk = chunk[output_columns]

        # Preserve important ID fields as strings
        for col in ("individual-local-identifier", "tag-local-identifier", "event-id"):
            if col in chunk.columns:
                chunk[col] = chunk[col].astype("string")

        for bird_id, group in chunk.groupby("individual-local-identifier", dropna=False, sort=False):
            bird_name = safe_filename(bird_id)
            out_file = out_dir / f"{bird_name}.csv"

            write_header = not out_file.exists()
            group.to_csv(
                out_file,
                mode="a",
                header=write_header,
                index=False,
            )

            seen_birds.add(str(bird_id))

        if i % 10 == 0:
            print(f"Processed {i} chunks...")

    return sorted(seen_birds), output_columns


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0088
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return r * c


def circular_heading_diff_deg(h1: pd.Series, h2: pd.Series) -> pd.Series:
    diff = (h2 - h1 + 180.0) % 360.0 - 180.0
    return diff.abs()


def summarize_bird_csv(bird_csv: Path) -> dict:
    df = pd.read_csv(bird_csv, low_memory=False)

    out = {"bird_file": bird_csv.name}

    for col in (
        "individual-local-identifier",
        "tag-local-identifier",
        "individual-taxon-canonical-name",
        "study-name",
        "sensor-type",
    ):
        out[col] = df[col].dropna().iloc[0] if col in df.columns and df[col].notna().any() else pd.NA

    time_col = "timestamp" if "timestamp" in df.columns else None
    if time_col is None and "eobs:start-timestamp" in df.columns:
        time_col = "eobs:start-timestamp"

    if time_col is not None:
        t = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        out["first_timestamp"] = t.min()
        out["last_timestamp"] = t.max()
        out["first_timestamp_jd"] = datetime_to_jd(pd.Series([t.min()])).iloc[0] if pd.notna(t.min()) else np.nan
        out["last_timestamp_jd"] = datetime_to_jd(pd.Series([t.max()])).iloc[0] if pd.notna(t.max()) else np.nan
        out["duration_days"] = (t.max() - t.min()).total_seconds() / 86400.0 if t.notna().any() else np.nan

        dt_sec = t.sort_values().diff().dt.total_seconds()
        out["n_rows"] = len(df)
        out["n_timestamps"] = int(t.notna().sum())
        out["mean_timestep_s"] = dt_sec.mean()
        out["median_timestep_s"] = dt_sec.median()
        out["min_timestep_s"] = dt_sec.min()
        out["max_timestep_s"] = dt_sec.max()
    else:
        out["n_rows"] = len(df)

    if "location-long" in df.columns and "location-lat" in df.columns:
        lon = pd.to_numeric(df["location-long"], errors="coerce")
        lat = pd.to_numeric(df["location-lat"], errors="coerce")
        gps_valid = lon.notna() & lat.notna()

        out["n_valid_gps"] = int(gps_valid.sum())
        out["fraction_valid_gps"] = float(gps_valid.mean())

        if gps_valid.any():
            out["mean_lon"] = lon[gps_valid].mean()
            out["mean_lat"] = lat[gps_valid].mean()
            out["min_lon"] = lon[gps_valid].min()
            out["max_lon"] = lon[gps_valid].max()
            out["min_lat"] = lat[gps_valid].min()
            out["max_lat"] = lat[gps_valid].max()

            work = pd.DataFrame({"lon": lon, "lat": lat, "gps_valid": gps_valid})
            if time_col is not None:
                work["time"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
                work = work.sort_values("time")

            work = work[work["gps_valid"]].copy()
            work["lon_prev"] = work["lon"].shift(1)
            work["lat_prev"] = work["lat"].shift(1)

            seg_mask = work["lon_prev"].notna() & work["lat_prev"].notna()
            if seg_mask.any():
                seg_km = haversine_km(
                    work.loc[seg_mask, "lon_prev"].to_numpy(),
                    work.loc[seg_mask, "lat_prev"].to_numpy(),
                    work.loc[seg_mask, "lon"].to_numpy(),
                    work.loc[seg_mask, "lat"].to_numpy(),
                )
                out["total_path_km"] = float(np.nansum(seg_km))
                out["mean_step_km"] = float(np.nanmean(seg_km))
                out["median_step_km"] = float(np.nanmedian(seg_km))
                out["max_step_km"] = float(np.nanmax(seg_km))

    if "ground-speed" in df.columns:
        gs = pd.to_numeric(df["ground-speed"], errors="coerce")
        out["mean_ground_speed"] = gs.mean()
        out["median_ground_speed"] = gs.median()
        out["max_ground_speed"] = gs.max()

    if "heading" in df.columns:
        hd = pd.to_numeric(df["heading"], errors="coerce")
        if time_col is not None:
            order = pd.to_datetime(df[time_col], errors="coerce", utc=True).argsort()
            hd = hd.iloc[order].reset_index(drop=True)
        dhead = circular_heading_diff_deg(hd.shift(1), hd)
        out["mean_abs_heading_change_deg"] = dhead.mean()
        out["median_abs_heading_change_deg"] = dhead.median()

    if "height-above-ellipsoid" in df.columns:
        h = pd.to_numeric(df["height-above-ellipsoid"], errors="coerce")
        out["mean_height_above_ellipsoid"] = h.mean()
        out["median_height_above_ellipsoid"] = h.median()
        out["max_height_above_ellipsoid"] = h.max()

    return out


def build_catalogue(per_bird_dir: Path, meta_df: pd.DataFrame, out_catalogue_csv: Path):
    rows = []
    for i, bird_csv in enumerate(sorted(per_bird_dir.glob("*.csv")), start=1):
        rows.append(summarize_bird_csv(bird_csv))
        if i % 50 == 0:
            print(f"Summarized {i} bird files...")

    cat = pd.DataFrame(rows)

    if "individual-local-identifier" in cat.columns:
        cat["individual-local-identifier"] = cat["individual-local-identifier"].astype("string")
    if "tag-local-identifier" in cat.columns:
        cat["tag-local-identifier"] = cat["tag-local-identifier"].astype("string")

    if "animal-id" in meta_df.columns:
        meta_df["animal-id"] = meta_df["animal-id"].astype("string")
    if "tag-id" in meta_df.columns:
        meta_df["tag-id"] = meta_df["tag-id"].astype("string")

    merged = cat.merge(
        meta_df,
        how="left",
        left_on="individual-local-identifier",
        right_on="animal-id",
        suffixes=("", "_meta"),
    )

    merged.to_csv(out_catalogue_csv, index=False)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("main_csv")
    parser.add_argument("meta_csv")
    parser.add_argument("--outdir", default="golden_eai_outputs")
    parser.add_argument("--chunksize", type=int, default=200000)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    per_bird_dir = outdir / "per_bird_csv"
    outdir.mkdir(parents=True, exist_ok=True)

    processed_meta_csv = outdir / "metadata_with_jd.csv"
    catalogue_csv = outdir / "bird_catalogue.csv"

    print("Processing metadata...")
    meta_df = process_metadata(Path(args.meta_csv), processed_meta_csv)

    print("Splitting main CSV by bird with fixed schema...")
    bird_ids, output_columns = split_main_csv_by_bird_fixed_schema(
        Path(args.main_csv),
        per_bird_dir,
        chunksize=args.chunksize,
    )

    print(f"Created {len(bird_ids)} bird files")
    print(f"Per-bird schema has {len(output_columns)} columns")

    print("Building catalogue...")
    cat = build_catalogue(per_bird_dir, meta_df, catalogue_csv)

    print(f"Wrote: {catalogue_csv}")
    print(f"Catalogue rows: {len(cat)}")
    print("Done.")


if __name__ == "__main__":
    main()