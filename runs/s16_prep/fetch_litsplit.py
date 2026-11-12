"""stage16 prep: literature split of the 5.625 deg papers (ClimODE, ClimaX, CoDiCast, WeatherODE):
train 2006-2015, calib 2016 (validation), test 2017-2018.  Base 11 state channels + the 10 column channels of
stage15a/stage9b, same WB2 64x32 source and layout as data/era5_wb2_64x32_2008_2019.npz."""
import sys, time
sys.path.insert(0, ".")
from gpclimate.data.era5_wb2 import fetch_wb2_subset
PERIODS = {"train": ("2006-01-01", "2015-12-31T18"), "calib": ("2016-01-01", "2016-12-31T18"),
           "test": ("2017-01-01", "2018-12-31T18")}
COLUMN10 = ["sp", "z1000", "z850", "t500", "z50", "r500", "r850", "tcwv", "u1000", "v1000"]
t = time.time()
fetch_wb2_subset("data/era5_wb2_64x32_2006_2018.npz", PERIODS)
print(f"base11 done in {time.time() - t:.0f} s", flush=True)
t = time.time()
fetch_wb2_subset("data/era5_wb2_64x32_column10_2006_2018.npz", PERIODS, names=COLUMN10)
print(f"column10 done in {time.time() - t:.0f} s", flush=True)
import numpy as np
for f in ("data/era5_wb2_64x32_2006_2018.npz", "data/era5_wb2_64x32_column10_2006_2018.npz"):
    z = np.load(f)
    for s in PERIODS:
        tt = z[f"{s}/time"].astype("datetime64[ns]")
        print(f, s, len(tt), tt[0], tt[-1], flush=True)
print("FETCH_S16_DONE", flush=True)
