"""stage15a prep: column channels of the benchmark years (test splits aligned with data/era5_wb2_64x32_<year>test.npz)."""
import sys, time
sys.path.insert(0, ".")
from gpclimate.data.era5_wb2 import fetch_wb2_subset
NEW = ["sp", "z1000", "z850", "t500", "z50", "r500", "r850", "tcwv", "u1000", "v1000"]
for year, period in (("2020", ("2020-01-01", "2020-12-31T18")), ("2022", ("2022-01-02", "2022-12-31T18"))):
    t = time.time()
    fetch_wb2_subset(f"data/era5_wb2_64x32_column10_{year}test.npz", {"test": period}, names=NEW)
    print(f"{year} done in {time.time() - t:.0f} s", flush=True)
