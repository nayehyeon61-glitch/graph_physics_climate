# Physical Guide Feature Matrix — FourCastNet / SFNO 참고 설계 (stage10)

작성 2026-11-03 (KST 15:30). 감자 제안: "FourCastNet (NVIDIA, 2022) & SFNO (2023), https://github.com/NVlabs/FourCastNet 의 matrix 구성을 참고해 feature matrix의 형태와 ERA5의 특징을 전처리하는 작업 — physical guide feature matrix".

이 문서는 (1) 두 모델의 입력 행렬이 무엇인지, (2) 우리 모델(Lie physical guide × Dynamics V-Net × graph diffusion, 64×32 ERA5)의 현재 feature matrix와 어떻게 다른지, (3) WB2 64×32 버킷에서 실제로 받을 수 있는 변수, (4) 그에 따라 정한 확장 명세 `fcn21`과 전처리, (5) guide 쪽에서 무엇이 달라지는지, (6) 비용과 stage10a 실험 계획을 적는다. 구현은 커밋에 포함(`gpclimate/data/era5_wb2.py`의 `SPECS`, `scripts/train_eval.py --spec`, `tests/test_feature_matrix.py`).

## 0. 출처

| 출처 | 내용 |
|---|---|
| FourCastNet — Pathak et al. 2022, arXiv:2202.11214; 코드 https://github.com/NVlabs/FourCastNet | AFNO(Adaptive Fourier Neural Operator) 비전 트랜스포머, 0.25° (720×1440), 6 h 자기회귀. `config/AFNO.yaml`: `in_channels = out_channels = 0..19`, `normalization: zscore`(채널별 전역 평균·표준편차 파일 `global_means.npy / global_stds.npy`), `orography: False`(선택), `dt: 1`(= 6 h), `n_history: 0`, 2단계 학습(1-step 사전학습 → 2-step 미세조정). 강수는 별도 모델. |
| SFNO — Bonev et al. 2023, arXiv:2306.03838; 코드 NVIDIA `makani` | 구면 조화 변환 기반 FNO. 73채널 = 지표 8(u10, v10, u100, v100, t2m, sp, msl, tcwv) + 13개 기압면 × (z, u, v, t, r) 65. 정적 입력(지형, 해륙), 태양천정각·위경도 조건 추가. |
| WeatherBench2 ERA5 64×32 (`1959-2023_01_10-6h-64x32_equiangular_conservative.zarr`) | 13 기압면 [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] hPa. 지표: 10 m 바람, 2 m 기온·이슬점, msl, sp, tcwv, 총운량, 강수(6/12/24 h), SST, 해빙, 경계층 높이, 복사 플럭스 등. **100 m 바람은 없음**(SFNO 73채널 중 71채널까지 구성 가능). |

## 1. FourCastNet의 행렬 구성이 뜻하는 것

FourCastNet의 상태는 X_t ∈ R^{20×H×W}, 한 시각의 **20개 2-D 장을 채널로 쌓은 행렬**이다(채널 순서 0–19: u10, v10, t2m, sp, msl, t850, u1000, v1000, z1000, u850, v850, z850, u500, v500, z500, t500, z50, r500, r850, tcwv). 설계상 중요한 점은 세 가지다.

1. **입력 채널 = 출력 채널.** 자기회귀 예보에서 모델은 자기 예보를 다시 입력으로 받으므로, 예측하지 않는 채널은 두 번째 step부터 존재하지 않는다. 그래서 "입력에만 넣는 보조 변수"는 둘 수 없고, 추가 정보는 **상태로 승격해 함께 예측**하거나 **알려진 외부 조건**(지형, 태양 복사)이어야 한다. 우리 모델도 같은 제약을 받는다(Stage B rollout 손실, 평가 5-step rollout).
2. **채널별 z-score.** 단위가 전혀 다른 장(z 10^4 m²/s², q 10^-3 kg/kg)을 같은 스케일에 둔다. 우리는 이미 같은 방식이고, 바람쌍만 평균 이동 없는 공통 스케일(회전 등변성 유지, 설계서 9절)이다.
3. **기둥(column) 정보가 얇게나마 들어 있다.** z1000/z850/z500/z50, t850/t500, r850/r500, u/v 1000/850/500/10 m, sp — 대기의 연직 구조(두께, 안정도, 수분, 경계층 바람)를 20채널로 요약한 것이다. 우리 base11에는 지표기압·1000 hPa·500 hPa 기온·상대습도·tcwv·성층권이 없다.

SFNO는 같은 생각을 13개 기압면 전부로 확장한 것(73채널)이고, 정적·시간 조건을 명시적으로 추가한다.

## 2. 현재 feature matrix (base11, stage1–stage8)

| 역할 | 채널 | 개수 | 전처리 |
|---|---|---|---|
| 상태 스칼라 | z500, t850, q700, t2m, msl | 5 | 채널별 z-score(학습 기간 2016–2017, chunked float64 적률) |
| 상태 접벡터 u+iv | (u850,v850), (u500,v500), (u10,v10) | 3 | 쌍별 RMS 공통 스케일, 평균 이동 없음 |
| 정적 조건 | orog(m), lsm; ∇orog(벡터) | 2 + 1 | z-score / 공통 스케일 |
| 강제 조건 | toa_t, toa_t1(초기·검증 시각의 cos 천정각 근사), coriolis(sin φ) | 3 | z-score |
| guide 고정 물리 연산자(메시 0 입력) | div/curl(바람쌍), 이류 ∇s·u(모든 스칼라·조건 × 모든 바람쌍), 절대와도 이류, 신장, 라플라시안 | 스칼라 47 + 벡터 19 | 물리 단위 환산 후 고정 |
| Stage A 목표 | 재구성 11, 와도·발산 6, 6 h 경향 11 | | `delta_s/v`, `tend_s/v` 스케일 |

guide 출력 P = {P0, P1, P2, P1u, P0u}(폭 guide_cs 32 / guide_cv 8)은 feature matrix 크기와 무관하므로, 상태 채널을 늘려도 V-Net 융합부는 그대로다.

## 3. 확장 명세 `fcn21` (stage10a)

FourCastNet 20채널 + q700. q700을 남기는 이유: base11이 부분집합이어야 stage3과 **같은 11개 변수**로 비교할 수 있고, stage5c의 허용조건 사영(q700 ≥ 0, p_s 가중 기둥수분)이 그대로 쓰인다.

| # | 채널 | 역할 | WB2 변수 (level) | 단위 / 학습기간 통계(평균, 표준편차) | guide에 주는 물리 정보 |
|---|---|---|---|---|---|
| 1–5 | z500, t850, q700, t2m, msl | 스칼라 | 기존 | 기존 | 기존 |
| 6 | sp | 스칼라 | surface_pressure | Pa; 9.67e4, 9055 | 지표 질량 → 연속방정식(기둥 질량 플럭스 발산), 지형 위 기압 |
| 7 | z1000 | 스칼라 | geopotential (1000) | m²/s²; 721, 1092 | z500 − z1000 두께 ∝ 층 평균 기온(측고 방정식) |
| 8 | z850 | 스칼라 | geopotential (850) | 1.38e4, 1505 | 850 hPa 지균풍 vs 실제 u/v850 → 비지균 잔차 |
| 9 | t500 | 스칼라 | temperature (500) | K; 253.5, 12.96 | t850 − t500 감률 → 정적 안정도 |
| 10 | z50 | 스칼라 | geopotential (50) | 2.00e5, 5783 | 성층권 극소용돌이(느린 모드) |
| 11 | r500 | 스칼라 | relative_humidity (500) | 분율 0–1(WB2는 %가 아님); 0.41, 0.22 | 중층 포화도(응결 가열의 전제) |
| 12 | r850 | 스칼라 | relative_humidity (850) | 0.61, 0.23 (최대 1.01 과포화) | 하층 포화도 |
| 13 | tcwv | 스칼라 | total_column_water_vapour | kg/m²; 19.0, 16.5(양의 왜도) | 기둥 수분(허용조건의 직접 변수) |
| 14–16 | (u850,v850), (u500,v500), (u10,v10) | 벡터 | 기존 | 기존 | 기존 |
| 17 | (u1000,v1000) | 벡터 | u/v_component_of_wind (1000) | m/s; std 5.7 / 4.8 | 경계층 바람(Ekman 회전: u10 vs u1000) |

합계 13 스칼라 + 4 벡터쌍 = 21 상태 채널. 정적·강제 조건(5 스칼라 + 1 벡터)은 그대로. 확인용: 새 파일의 11개 기존 채널과 시각 축은 기존 파일과 **정확히 동일**(max|diff| 0, `data/fetch_fcn21_2016_2019.log`).

`fcn20`(q700 없는 FourCastNet 원본 구성)도 `SPECS`에 등록해 두었으나 실험하지 않는다(비교 변수가 깨짐).

## 4. 전처리 (ERA5의 특징을 어떻게 다루나)

stage10a는 **FourCastNet과 같은 전처리**를 유지한다 — 가설을 "채널 집합" 하나로 한정하기 위해서다.

- 채널별 z-score(스칼라), 쌍별 공통 스케일(벡터). 통계는 학습 기간에서만, chunked float64 적률(`fit_normalizer`).
- 경향 스케일 `tend_s/tend_v`(6 h 변화의 표준편차)와 lead 변화 스케일 `delta_s/delta_v`도 21채널로 자동 확장(zero-init head는 여전히 persistence에서 시작).
- 정적·강제 조건은 base11과 동일: FourCastNet의 선택적 orography, SFNO의 지형·해륙·천정각 조건과 같은 범주.
- 결측·극값: WB2 보존 재격자 자료라 결측 없음. r은 [0, 1.01], tcwv·q는 양의 왜도 — z-score 후 꼬리가 길다(아래 후보 1).

**stage10a에 넣지 않은 전처리 후보**(각각 별도 stage로):

| 후보 | 내용 | 기대 | 왜 지금 안 넣나 |
|---|---|---|---|
| (1) log 변환 | log(tcwv), log(q700) 후 z-score | 양의 왜도 완화, q ≥ 0 자동 | 역변환이 손실·허용조건 사영과 얽힘 |
| (2) 시간 부호화 | sin/cos(시각, 연중일)을 강제 조건으로 | 일주기·계절 조건 명시(SFNO 방식) | toa_t/toa_t1이 이미 대부분 담음 |
| (3) 기후값 편차 | 격자점·연중일 기후값을 빼고 학습 | 계절 신호 제거, z50 등 느린 모드에 유리 | 기후값 파일·평가 역변환 필요 |
| (4) 실제 TOA 복사 | `mean_top_downward_short_wave_radiation_flux`(WB2) | cos 천정각 근사 대체 | 강제 조건 변경은 별도 가설 |
| (5) 추가 외부 장 | SST, 해빙, 경계층 높이, 지형 통계(WB2에 있음) | 하층 경계 조건 | 상태/조건 역할 결정 필요(SST는 느려 정적 취급 가능) |
| (6) 기둥 행렬 | SFNO식 13면 × (z,u,v,t,r) + 지표 = 71채널(100 m 바람 없음) | 완전한 연직 구조 | 연산자 행렬 771+90으로 비대 → 연산자를 "같은 층끼리"로 재구성 후 GPU |

## 5. physical guide 쪽에서 달라지는 것

- 메시 0 입력 = 상태(13+4) + 조건(5+1) + 고정 연산자. 연산자 수는 `(2n_v + n_v·n_s + n_v·n_c + 2n_v + n_s, n_s + n_c + 3n_v)` → base11 **47+19**, fcn21 **101+30**. 늘어난 부분은 거의 **모든 바람층 × 모든 스칼라의 이류항**(4×13 = 52)이다 — 기둥 정보가 guide에 들어오는 통로.
- 헤드: 재구성 21, 와도·발산 보조 8(4쌍), 6 h 경향 21.
- guide 파라미터 275,817 → 278,738(+1.1 %), V-Net 554,665 → 555,489. P 수준의 폭이 그대로라 융합부는 변화 없음.
- stage5a의 닫힘 구조 guide(`gpclimate/nn/primitive2d.py`)는 base11 배치 전용(명시적 `ValueError`) — fcn21에서는 경향 head 경로(stage3 방식)를 쓴다. 확장하려면 sp를 쓰는 연속방정식, 두께 기반 온도 이류 등 **기둥 항**을 F에 추가해야 하며 이는 stage10b/stage9 후보다.
- 평가는 21개 변수 모두 RMSE를 내지만, stage 비교는 11개 공통 변수(+ ClimODE 5변수)로 한다. 새 10개 변수는 persistence·climatology 대비로만 본다.

## 6. 비용 (측정: 1 thread, nice 19, batch 4, 64×32, 메시 (4,3,2), stage3 모델 크기)

| | base11 | fcn21 |
|---|---|---|
| Stage A forward+backward | 1.84 s | 1.62 s(동일 수준, 측정 잡음) |
| Stage B forward+backward | 4.61 s | 4.30 s |
| 예보 1 step(no grad) | 1.93 s | 1.89 s |
| 데이터 파일 2016–2019 | 464 MB (11ch) | 878 MB (21ch), 받는 데 202 s |

폭(hidden 64, cs 48)이 비용을 지배하므로 채널 수는 거의 영향이 없다. stage10a 전체(guide 없음 V-Net → A → B → 평가 → 진단)는 stage7 체인과 같은 약 3시간.

## 7. 실험 계획

| stage | 내용 | 비교 | 상태 |
|---|---|---|---|
| stage10a | `--spec fcn21`, 나머지 stage3 레시피(2016–2017, A 1000 / B 1500, 섭동·2-step rollout) | stage3(11개 공통 변수), stage3 guide 없음, ClimODE(5변수); P 의존도(P=0, 섞은 P) | `runs/era5_64x32_s10a_fcn21/chain_s10.sh` — stage8b 종료 후 자동(≈ 11/4 02:00 KST 시작, ≈ 05:00 KST 종료) |
| stage10b 후보 | 4절 (1)–(5) 중 하나, 또는 5절의 기둥 항을 닫힘 guide에 추가 | stage10a | stage10a 결과 후 결정 |
| stage10c 후보 | fcn21 + 10년 데이터(stage7이 유효하면) | stage7 | 〃 |

채택 기준: 11개 공통 변수 평균 RMSE가 5개 리드 중 3개 이상에서 stage3보다 낮고, P 의존도(P=0 시 MSE 증가)가 stage3(+102 %) 아래로 떨어지지 않을 것. 특히 t2m·msl·q700(지표·수분)이 개선되는지가 가설의 직접 검증이다.

## 8. 위험

- Stage A 손실이 21채널 평균이 되어 기존 11채널의 비중이 상대적으로 줄어든다(채널 가중 없음). 결과가 나쁘면 11채널 가중을 stage10b로.
- z50은 계절 주기가 지배하는 느린 장이라 재구성·경향 head에 쉬운 목표를 더해 손실 수치가 좋아 보일 수 있다 — 반드시 변수별로 본다.
- sp와 msl은 해상에서 거의 같고 지형 위에서만 다르다(중복 정보). r은 상한이 있어 z-score가 어색하다.
- ClimODE 비교는 여전히 5변수(z500, t850, t2m, u10, v10)뿐이다.

## 9. 구현 지도

| 파일 | 변경 |
|---|---|
| `gpclimate/data/era5_wb2.py` | `WB2_SOURCES` 21개(+sp, u/v1000, z1000, z850, t500, z50, r500, r850, tcwv), `FCN20_SPEC`, `FCN21_SPEC`, `SPECS = {base11, fcn20, fcn21}`, `fetch_wb2_subset(..., names=)` |
| `scripts/train_eval.py` | `--spec {base11,fcn20,fcn21}`(run 생성 시 config.json에 저장; 이전 run은 base11), 평가 변수 = 명세의 모든 상태 채널 |
| `scripts/compare_runs.py` | 기준 열의 변수 전부를 행으로(기존 11개 먼저), 없는 변수는 "—" |
| `tests/test_feature_matrix.py` | 명세가 FourCastNet 20채널을 덮고 base11이 부분집합인지, WB2 소스·바람쌍 일관성, 21채널 npz → 쌍 → 정규화 → guide/V-Net forward, base11이 같은 파일을 읽어 동일 값인지, 연산자 행렬 47+19 / 101+30 |
| `data/era5_wb2_64x32_fcn21_2016_2019.npz` | 21채널 2016–2019(gitignore; `data/fetch_fcn21_2016_2019.log`에 통계) |
| `runs/era5_64x32_s10a_fcn21/chain_s10.sh` | stage10a 자동 체인 |


## 10. stage10a 결과와 stage10b (2026-11-04 추가)

stage10a(같은 폭·같은 step, 비가중 21채널 손실)는 공통 11변수에서 stage3보다 +5.4 %(24 h) … +2.9 %(120 h) 나빴고, guide 없는 baseline도 +4.1 … +1.5 % 나빠졌다(`reports/compare_stage10a.md`). 즉 악화의 대부분은 9절에서 예고한 **손실 희석**(11채널 비중이 절반으로, 같은 V-Net 폭이 21채널 출력을 분담)이고, fcn21에서 guide 이득은 −0.2 %(base11 −1.4 %), P=0 +58 %(+102 %)로 기둥 정보가 guide를 강화하지 못했다. 신규 10변수 자체는 skill이 있다(24 h, 1 − RMSE/기후: z50 0.88, t500 0.68, tcwv 0.67, z850 0.66, sp·z1000 0.64, u/v1000 0.50, r 0.32–0.35).

**stage10b** = 같은 fcn21 run에 채널 가중 손실만 추가: `--loss-focus base11 --loss-focus-weight 0.2`(config.json 키 `loss_focus`, `loss_focus_weight`; `field_mse(..., ch_w)`가 분자·분모를 함께 가중하므로 가중 1이면 기존 손실과 동일). 적용 범위는 Stage A 재구성·경향 head 손실, Stage B 1-step·rollout 손실, guide 없음 baseline; 검증 점수·평가는 비가중이라 stage10a와 직접 비교된다. 체인 `runs/era5_64x32_s10b_fcn21_w/chain_s10b.sh`(stage7b 뒤). 판정 기준은 stage10a와 같다(공통 11변수 평균이 stage3보다 5 lead 중 3 이상에서 낮고 P-의존도가 stage3 이상). 실패하면 fcn21은 '출력'이 아니라 **guide 입력 전용 조건**으로만 쓰는 변형(상태는 base11, 신규 채널은 t₀ 조건)을 stage9 후보로 남기고 stage10c(fcn21 + 10년)는 취소한다.
