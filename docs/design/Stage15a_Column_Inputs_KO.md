# stage15a — 기둥 입력(초기시각 전용 조건)

요청: 감자 2026-11-05 21:25 KST — "만약 추가 구조로 weather ode를 이길려면 어떤 단계를 추가하면 좋을것 같나요?" → 결정 카드(21:34 KST) "stage15a 기둥 입력부터"(stage7c 다음).

## 1. 왜 기둥 입력인가

WeatherODE(arXiv 2410.06560)는 5.625°에서 **48채널**(6변수 × 7층 + 지표 3 + 상수 3)을 입력으로 받고 5개 변수만 예측한다. stage9는 11채널(500·700·850 hPa 일부와 지표)만 본다. 500 hPa 지위 24 h 예보에는 층후(온도 구조), 안정도, 습도, 하층·성층권 흐름 같은 연직 정보가 중요하다.

우리 증거: stage10a/10b는 FourCastNet 기둥 채널을 **상태(출력)**로 넣었다가 21채널 손실 평균에 희석돼 기각됐다. 그래도 stage10b(가중 손실)에서 z500 +0.6…+1.4 %, msl 최대 +3.3 %가 좋아졌고 guide 없는 fcn21 짝은 stage3 짝보다 +0.8…+1.0 % 좋았다. 출력 희석 없이 입력으로만 쓰면 이 정보만 얻을 수 있다는 것이 가설이다.

## 2. 구조

- 채널: fcn21 중 base11에 없는 10개 — 스칼라 8개(sp, z1000, z850, t500, z50, r500, r850, tcwv) + 바람쌍 1개(u1000, v1000, 국소 (동, 북) 게이지의 접벡터 u + iv). 등변 분할은 상태와 같다.
- 정규화: 스칼라는 학습 분할(2008–2017) 평균·표준편차 z-score, 바람쌍은 공통 RMS 척도(평균 이동 없음, 회전과 교환). 통계는 run의 `column_norm.json`에 저장해 외부 연도 평가에도 같은 척도를 쓴다.
- 들어가는 곳: V-Net의 GridLift 입력에만 [상태, 조건, (stage14a 변화량), **기둥 10채널, 가용 플래그**]로 붙는다. guide·diffusion·PIGT 연산자 토큰은 그대로다.
- **초기시각 전용**: rollout k ≥ 2 단계에서는 미래 기둥값을 모른다(모델이 예측하지 않음). 진짜 X_{t+k·lead} 기둥값을 넣으면 정답 누설이므로, 2단계부터는 0과 가용 플래그 0을 넣는다. 학습의 2-step rollout 손실이 이 두 상태(있음/없음)를 모두 학습시킨다. 평가의 3–5단계도 2단계와 같은 '없음' 상태라 분포 이동이 없다.
- 예상: 이득은 24 h에 가장 크고, 더 나은 24 h 상태를 통해 뒤 lead로 전파된다.

## 3. 실험 설정 (한 가지 변화)

stage9와 같은 데이터(2008–2017 학습, 2018 calib, 2019 시험), 같은 normaliser, Stage B 4500 step(rollout 2, 1500부터), Stage C 800 step. guide는 기둥을 보지 않으므로 stage9의 guide와 Stage A 표지를 그대로 복사한다(재학습하면 섭동 모델이 바뀌어 두 번째 변화가 된다). guide 없는 짝은 같은 기둥 입력으로 재학습한다. 매개변수 증가는 lift의 입력 9 스칼라 + 1 벡터 채널뿐이다.

비교: stage9 대비 11변수 평균 RMSE 개선률(+ = 개선), guide 없는 짝끼리, guide 이득, ClimODE 열, P=0 진단, 2020·2022 벤치마크 연도. 채택 기준은 기존과 같다(5개 lead 중 3개 이상 개선).

## 4. 코드

- `gpclimate/data/era5_wb2.py`: `COLUMN_SETS["fcn10"]`, `ColumnInputs`(분할별 로드, 시간축 일치 검사, 정규화), `column_stats`.
- `gpclimate/models/dynamics_vnet.py`: `VNetConfig.col_s/col_v`, `DynamicsVNet.column_input`, `forward(..., col=)`, `LieGuidedForecaster.forward/rollout(..., col=)`(rollout은 첫 단계에만 전달).
- `scripts/train_eval.py`: `--column-inputs fcn10 --column-data <npz>`(config.json에 저장), 외부 연도는 `--eval-column-data`; 모든 첫 단계 호출(학습, 검증 1·2-step, Stage C, 평가 rollout·앙상블)에 `col=run.col(ds, idx)`. `scripts/diagnose_guide.py`, `scripts/diagnose_errors.py`도 같다.
- 데이터: `data/era5_wb2_64x32_column10_2008_2019.npz`(1.24 GB, 190 s), 벤치마크 연도 `..._column10_2020test.npz`, `..._column10_2022test.npz`(`runs/s15a_prep/`).
- 테스트: `tests/test_column_inputs.py` — 로더(정렬·정규화·시간 불일치 오류), rollout이 2단계부터 기둥을 넣지 않음(누설 없음), stage9+기둥 파이프라인 b_noguide → a → b → c → eval → 외부 연도(기둥 파일 필수) → diagnose.
- 실행: `runs/era5_64x32_s15a_column/chain_s15a.sh`(CHAIN_S7C_DONE 대기 → 약 7 h → 2020·2022 채점, 표지 CHAIN_S15A_DONE).

## 5. 다음 후보 (WeatherODE 추격)

- stage15b: 같은 forward에서 6/12/18 h 보조 출력 손실(WeatherODE의 중간 시각 손실, 감독 신호 4배).
- stage9b: stage7c 결과에 맞춘 step 예산.
- GPU가 있으면: 6 h 4-step 적분(stage8c), V-Net 용량 확대.
