import csv
import os
from typing import List, Tuple

import numpy as np
import zarr


def pick_random_index_for_day(time: np.ndarray, day: np.datetime64) -> int:
    """Pick a random index from the time array that belongs to the given day."""
    day_mask = time.astype("datetime64[D]") == day
    indices = np.flatnonzero(day_mask)
    if indices.size == 0:
        raise ValueError("No timestamps available for day {}".format(day))
    choice = int(np.random.choice(indices))
    return choice


def compute_stats_for_index(root: zarr.group, variables: List[str], idx: int) -> List[Tuple[str, float, float]]:
    """Compute nanmean and nanstd for each variable at time index idx."""
    results: List[Tuple[str, float, float]] = []
    print(f"Fetching times for index {idx}")
    for var in variables:
        if var == "tp" or var =="aerot":
            data = np.log(root[var][idx, :, :] + 1e-8)
        else:
            data = root[var][idx, :, :]
        x_val = float(np.nansum(data))
        x2_val = float(np.nansum(data**2))
        results.append((var, x_val, x2_val))
    return results


def main() -> None:
    root = zarr.open_group(
        store="s3://hrrr-surface-sda/zarr-v1",
        mode="r",
        storage_options={"endpoint_url": "https://pdx.s8k.io"},
    )
    variables = [
        "u10m",
        "v10m",
        "u80m",
        "v80m",
        "t2m",
        "d2m",
        "q2m",
        "sp",
        "fg10m",
        "tcc",
        "sde",
        "snowc",
        "refc",
        "rsds",
        "tp",
        "aerot",
    ]
    time = root["time"][:]

    time_2023 = time[(time >= np.datetime64("2023-01-01T00:00:00")) & (time < np.datetime64("2024-01-01T00:00:00"))]
    days_2023 = np.unique(time_2023.astype("datetime64[D]"))

    output_path = "daily_random_moments.csv"
    write_header = not os.path.exists(output_path)
    with open(output_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["date", "timestamp", "variable", "x", "x2"])

    for day in days_2023:
        try:
            idx = pick_random_index_for_day(time, day)
        except ValueError:
            continue
        ts = time[idx]
        stats = compute_stats_for_index(root, variables, idx)
        with open(output_path, "a", newline="") as f:
            writer = csv.writer(f)
            for var, mean_val, std_val in stats:
                writer.writerow([str(day), np.datetime_as_string(ts, unit="s"), var, f"{mean_val:.16g}", f"{std_val:.16g}"])

if __name__ == "__main__":
    main()

