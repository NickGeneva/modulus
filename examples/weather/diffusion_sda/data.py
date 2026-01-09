import datetime
import csv
import os
import torch
import zarr
import numpy as np
from torch.utils.data import Dataset

from physicsnemo.utils.zenith_angle import cos_zenith_angle


class HRRRSurfaceDataset(Dataset):

    VARIABLES = [
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
        "aerot"
    ]
    LOG_VARIABLES = ("tp", "aerot") # Make sure is consistent with stats CSV
    EPSILON = 1e-8

    def __init__(self, zarr_root: zarr.group, time_indices: np.array, stats_csv: str = "stats/stats.csv"):

        self.root = zarr_root
        self.idx = np.asarray(time_indices, dtype=int).ravel()

        # Verify bounds against available time coordinate in zarr
        n_time = self.root["time"].size
        if np.any((self.idx < 0) | (self.idx >= n_time)):
            invalid_values = np.unique(self.idx[out_of_bounds_mask])
            raise IndexError(
                "time_indices contain out-of-bounds values for zarr_root['time']"
            )

        # Load normalization stats and log-scaling flags from summary_stats.csv
        stats_csv = os.path.join(os.path.dirname(__file__), stats_csv)
        means = []
        stds = []
        stats_map = {}
        with open(stats_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                var = row.get("variable")
                mu = float(row.get("mean", "nan"))
                sd = float(row.get("std", "nan"))
                stats_map[var] = (mu, sd)

        # Order based on VARIABLES
        for var in self.VARIABLES:
            mu, sd = stats_map[var]
            means.append(mu)
            stds.append(sd)

        # Instance-level overrides for normalization and log variables
        self.target_means = torch.tensor(means, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        self.target_stds = torch.tensor(stds, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)

        self.grid_lat = self.root["lat"][:]
        self.grid_lon = self.root["lon"][:]

    def __len__(self):
        return self.idx.shape[0]

    def __getitem__(self, idx):
        time_idx = self.idx[idx]
        time_stamp = self.root['time'][time_idx]

        # Target tensor
        # TODO: Make async
        data_arrays = np.empty((len(self.VARIABLES), self.grid_lat.shape[0], self.grid_lat.shape[1]))
        for i, variable in enumerate(self.VARIABLES):
            if variable in self.LOG_VARIABLES:
                data_arrays[i] = np.log(self.root[variable][time_idx] + self.EPSILON)
            else:
                data_arrays[i] = self.root[variable][time_idx]

        target = torch.Tensor(data_arrays)
        target = (target - self.target_means) / self.target_stds

        # Conditional encoding
        data_arrays = np.empty((4, self.grid_lat.shape[0], self.grid_lat.shape[1]))
        ts = ((time_stamp - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
        data_arrays[0] = cos_zenith_angle(datetime.datetime.utcfromtimestamp(ts), self.grid_lat, self.grid_lon)
        data_arrays[1][:] = int((time_stamp.astype('datetime64[D]') - time_stamp.astype('datetime64[Y]')).astype(int) + 1)
        data_arrays[2] = self.grid_lat
        data_arrays[3] = self.grid_lon

        condition = torch.Tensor(data_arrays)
        return condition, target


if __name__ == "__main__":
    root = zarr.open_group(store='s3://hrrr-surface-sda/zarr-v1', mode='r', storage_options={'endpoint_url': 'https://pdx.s8k.io'})
    time = root['time'][:]
    sidx = np.where(time == np.datetime64("2023-01-01T00:00:00"))[0][0]
    eidx = np.where(time == np.datetime64("2023-02-01T00:00:00"))[0][0]
    
    time_idx = np.arange(sidx, eidx)

    dataset = HRRRSurfaceDataset(root, time_idx)
    cond, target  = dataset[30]

    print(cond.shape)
    print(target.shape)