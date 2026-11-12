# stage16 — 문헌 분할 재학습 (stage9b 학습식, 2006–2015 학습 / 2017–2018 시험)

작성 2026-11-07 17:25 KST. SOTA 비교 3단계(`reports/SOTA_COMPARISON_KO.md`)의 본선.

## 왜 필요한가

같은 급(5.625°) 논문들(ClimODE, ClimaX, FourCastNet 5.625°, CoDiCast, WeatherODE, SDECast, PINN-Cast)은
모두 2006–2015(또는 1979–2015)로 학습하고 2017–2018로 시험하며 체크포인트를 공개하지 않는다.
지금까지는 시험 연도가 달라 'ClimODE 기준 지수'(RMSE ÷ 같은 논문의 ClimODE)로만 간접 비교했다.
stage9b가 t850 72 h에서 WeatherODE 지수를 넘었으므로, 같은 분할로 한 번 재학습해 논문 표의 RMSE와
직접 비교한다.

## 무엇이 바뀌나 / 안 바뀌나

| 항목 | stage9b | stage16 |
|---|---|---|
| 구조 | stage9 + 기둥 입력(fcn10, t0만) | 같음 |
| 학습식 | b_noguide·b 9000 step, rollout 2(3000부터), Stage C 800 | 같음 |
| 학습 연도 | 2008–2017 | **2006–2015** |
| 검증 연도 | 2018 | **2016** |
| 시험 연도 | 2019 | **2017–2018** (+ 2019 보조) |
| 정규화·기둥 통계·guide | stage9 것 재사용 | **처음부터 다시** (2016–2017을 본 것은 모두 버림) |

stage9b의 정규화·guide·guide 없는 짝은 2016–2017을 학습에 썼으므로 그대로 쓰면 시험 연도가 새어 들어간다.
그래서 guide 없는 짝(= Stage A 섭동 모델) → Stage A(1000 step) → Stage B → Stage C를 모두 새로 돌린다.
차이 하나: stage9b의 Stage A 섭동 모델은 stage9의 4500 step 짝이었고, 여기서는 9000 step 짝이다.

## 비교

1. **본선: 2017–2018, stage16 vs ClimODE** — 공개 체크포인트(2006–2015 학습)를 같은 초기 시각(4일 간격,
   약 180개)에 저자 방식(train mode)으로 돌린다. ClimODE에게도 진짜 시험 연도다.
2. **논문 표와 직접 비교** — z500·t850·t2m·u10·v10의 24/72 h(가능하면 36 h 대신 48 h) RMSE를 논문 값과 나란히.
   우리 ClimODE 재현값을 논문의 ClimODE 값(z500 24 h 193.4)과 맞춰 보고 프로토콜 차이(WB2 보존 재격자 vs
   WB1, 초기 시각 간격)를 함께 적는다.
3. **직전 단계: 2019, stage16 vs stage9b** — 둘 다 2019를 보지 않았고 학습 연도만 다르다(같은 10년 길이).

부호 규약: + = 개선, − = 악화. 시각은 KST.

## 실행

- 데이터: `runs/s16_prep/fetch_litsplit.py` → `data/era5_wb2_64x32_2006_2018.npz`,
  `data/era5_wb2_64x32_column10_2006_2018.npz` (WB2 64×32, 기존과 같은 출처·형식).
- 체인: `runs/era5_64x32_s16_litsplit/chain_s16.sh` (끝 표시 `CHAIN_S16_DONE`).
  CPU 4스레드로 b_noguide 9000 → A 1000 → b 9000 → C 800 → 평가(앙상블 24) → 진단 → ClimODE 2017–2018 → 2019 평가.
  예상 약 15–16시간.
