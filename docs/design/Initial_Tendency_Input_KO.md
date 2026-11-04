# 초기 경향 입력 (stage14): "초기만 잡으면 된다"에 대한 가설과 설계

출처: 감자 2026-11-04 10:03 KST — "초기만 잡으면 괜찮을 것 같은데? 방향이나 가설 있나요?" (stage11a 결과·오차 진단 뒤).

## 1. 왜 초기(24–48 h)인가

`reports/error_diagnostics_era5_64x32_physops_roll.md`: stage3이 ClimODE에 지는 곳은 24–48 h의 중위도·극 종관규모뿐이다(z500 24 h: 열대 +29.6 %, 중위도 −22.2 %, 극 −21.8 %; 120 h는 전 위도 +). 편향은 없고(계통 비중 0.17–0.24), 24 h에 이미 파수 8–15 파워가 73 %로 깎여 있다. 즉 문제는 **첫 step의 종관규모 발달**이다.

## 2. 가설 네 갈래 (+ 담당 stage)

| # | 가설 | 근거 | stage |
|---|---|---|---|
| 1 | 용량·최적화 부족 | 검증 MSE가 마지막 500 step에 +5.5 %; 모델 크기 ClimODE의 1/3 | stage12a(EMA), stage12b(폭), stage7b(step) |
| 2 | 이산 연산자의 평활 | 메시 1-hop 핵·격자↔메시 보간이 첫 step부터 파수 8–15를 깎음 | stage13a(구면 neural operator), stage13b(에너지 연속성) |
| 3 | rollout 손실의 대가 | 2-step rollout이 1-step 검증 MSE를 7 % 희생(stage2 0.0787 → stage3 0.0842) | stage12d 변형(rollout 비중↓·늦게 시작; 비용 0) |
| **4** | **초기 경향 정보 부재** | 모델은 X_t 한 장만 본다. 초기 발달 속도(파동 위상속도, 비지균 성분)는 한 시각에서 추론해야 함. ClimODE는 과거 프레임으로 속도장을 초기화하고 GraphCast는 입력이 두 시각(t, t−6h). stage11a(곡률 연산자)가 24 h만 올린 것도 "초기 발달을 더 잘 읽으면 초기가 좋아진다"는 방증 | **stage14a**(이 문서) |

부수 후보(비용 0): 위도 가중 손실(극 >60°가 24 h에 −22 %인데 cos φ 가중으로 손실 기여가 작음) — stage14b 후보.

## 3. stage14a 설계: 이전 상태 입력

- **입력**: V-Net의 격자 입력에 관측된 직전 lead 변화 ΔX = X_t − X_{t−24h}(정규화 단위, 스칼라 n_s + 벡터 n_v 채널)를 추가한다(`VNetConfig.prev_state`, `GridLift` 입력 차원 증가). guide는 바꾸지 않는다(stage3의 `guide.pt` 재사용, Stage A 생략) — 물리 특징은 여전히 X_t의 함수이고, "어느 방향으로 움직이고 있었나"는 동역학 층이 받는다.
- **왜 6 h가 아니라 24 h인가**: 24 h 간격이면 rollout에서 직전 예보 상태 X̂_{t+(k−1)·24h}가 그대로 X_{t−24h} 역할을 하므로 학습(2-step rollout)과 추론(5-step)이 일관된다. 6 h 경향(ClimODE·GraphCast식)은 rollout 두 번째 step부터 X̂_{t+18h}가 없어 근사가 필요하다 → stage14a가 효과를 보이면 stage14b(6 h 경향 + 보간 근사)로 비교.
- **첫 시각**: 분할의 첫 lead 구간(2016-01-01 00–18 UTC 등)은 이전 상태가 없어 ΔX = 0으로 둔다(`ERA5Pairs.prev_state` clamp). 학습 표본은 idx ≥ lead에서 뽑고, 평가의 첫 초기시각(2019-01-01 00 UTC, 90개 중 1개)만 ΔX = 0으로 예보한다 — 비교표의 1/90 효과, 문서에 명시.
- **동등성**: `prev=None`이면 ΔX = 0이라 다른 호출자(Stage C, 진단)는 그대로 동작하고, `prev_state=0`인 run은 코드 변경 전과 비트 동일(ΔX 채널이 없음). 게이지: ΔX는 같은 프레임의 두 상태 차이라 공변 → 등변성 유지(`tests/test_prev_state.py`).
- **비용**: 입력 채널 +11, 매개변수 +0.3 % 미만, 속도 변화 없음.

## 4. 실행과 판정

체인 `runs/era5_64x32_s14a_prev/chain_s14a.sh`(stage13b 뒤, 11-05 ≈18:45 KST 예상, ≈2.6 h): stage3 config + `prev_state=1`, Stage B 1500 step(rollout 2, 500부터) + guide 없는 baseline(같은 입력) + 평가 + 진단. 비교 기준 stage3(+ClimODE). **가설이 맞으면** 24–48 h 중위도·극 z500/t850이 눈에 띄게(≥3 %) 좋아지고 72 h 이후는 중립, 그리고 `scripts/diagnose_errors.py`에서 24 h 파수 4–15 파워비가 오른다. 채택 기준은 다른 stage와 같다(11변수 평균 5 lead 중 3개 이상 +, P=0 열화 유지).
