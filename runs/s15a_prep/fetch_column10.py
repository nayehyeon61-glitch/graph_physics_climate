"""stage15a prep: the 10 FourCastNet column channels missing from base11, 2008-2019 (same splits as
data/era5_wb2_64x32_2008_2019.npz), for use as t0 input conditions only."""
import sys, time
sys.path.insert(0, ".")
from gpclimate.data.era5_wb2 import fetch_wb2_subset
NEW = ["sp", "z1000", "z850", "t500", "z50", "r500", "r850", "tcwv", "u1000", "v1000"]
PERIODS = {"train": ("2008-01-01", "2017-12-31T18"), "calib": ("2018-01-01", "2018-12-31T18"),
           "test": ("2019-01-01", "2019-12-31T18")}
t = time.time()
fetch_wb2_subset("data/era5_wb2_64x32_column10_2008_2019.npz", PERIODS, names=NEW)
print(f"done in {time.time() - t:.0f} s", flush=True)
