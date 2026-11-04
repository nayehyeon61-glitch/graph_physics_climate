# Lag-1 autocorrelation of the 24 h model error — `runs/era5_64x32_physops_roll`

120 initial times of the calibration period; e_prev = X_t − M(X_{t−24h}), e_next = X_{t+24h} − M(X_t); ρ = corr(e_prev, e_next) (area-weighted, per channel), β = optimal linear carry coefficient, ρ² = fraction of next-step MSE a linear carry β·e_prev would remove (Evensen 2009 Ch. 12: red-noise model error q_k = ρ q_{k−1} + …, ρ = 1 − Δt/τ).

| channel | ρ global | ρ tropics | ρ midlat | ρ polar | β global | ρ² global |
|---|---|---|---|---|---|---|
| z500 | +0.178 | +0.298 | +0.158 | +0.191 | +0.178 | 0.032 |
| t850 | -0.021 | +0.079 | -0.050 | -0.030 | -0.021 | 0.000 |
| q700 | +0.091 | +0.146 | +0.013 | -0.027 | +0.091 | 0.008 |
| t2m | -0.034 | -0.074 | -0.053 | +0.007 | -0.034 | 0.001 |
| msl | +0.184 | +0.244 | +0.179 | +0.178 | +0.183 | 0.034 |
| u850/v850 | +0.066 | +0.142 | +0.033 | +0.063 | +0.066 | 0.004 |
| u500/v500 | +0.054 | +0.126 | +0.029 | +0.027 | +0.054 | 0.003 |
| u10/v10 | +0.056 | +0.142 | +0.040 | +0.006 | +0.056 | 0.003 |
| **mean** | **+0.072** | **+0.138** | **+0.043** | **+0.052** | | 0.011 |

MSE-weighted ρ (all channels pooled): global +0.067, tropics +0.140, midlat +0.040, polar +0.042
