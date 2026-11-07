# stage15b — 중간 시각 손실 (6/12/18 h 보조 출력)

작성 2026-11-07 03:40 KST. 요청 맥락: 감자 2026-11-05 21:25 KST "만약 추가 구조로 weather ode를 이길려면 어떤 단계를 추가하면 좋을것 같나요?" → 제안한 stage15a(기둥 입력)·stage15b(중간 시각 손실)·stage9b(step 예산 합치기) 중 15a와 9b를 마쳤다. stage9b 결과(11-07 03:15 KST)에서 ClimODE 기준 문헌 지수가 72 h는 WeatherODE와 같거나 앞서고(t850 +33.2 vs +29.8 %, z500 +30.9 vs +32.2 %) **24 h에만 차이가 남았다**(z500 +32.0 vs +40.8 %, t850 +24.9 vs +32.9 %). 다음 단계 결정 카드(03:25 KST)의 추천안으로, 답을 기다리는 동안 시작했다.

## 가설

WeatherODE(arXiv 2410.06560)는 1시간 간격 ERA5로 학습하면서 **모든 중간 시각의 상태에 손실**을 건다. 우리 모델은 24 h 뒤 상태 하나(+ rollout 48 h)만 감독받아서, 24 h 사이의 궤적 정보(6 h 간격 데이터에 이미 있는 X_{t+6h}, X_{t+12h}, X_{t+18h})를 버리고 있다. 같은 표현에서 중간 시각을 함께 맞히도록 하면 감독 신호가 4배가 되어 24 h 예보 자체가 좋아지고, 그 이득이 rollout으로 전파된다.

## 구조 변경 (stage9b 대비 하나)

- V-Net 디코더 임베딩(`head_dec` 출력, 24 h 출력 `head_out`과 공유)에 보조 선형 head `head_inter`(`EqLinear`, 스칼라 3×5·벡터 3×3 채널, 0으로 초기화 = 지속성에서 시작)를 붙여 X_{t+j·6h} − X_t, j = 1, 2, 3을 낸다(24 h 변화와 같은 delta 척도).
- 손실: `L = L_f + λ_roll L_roll + L_phys + w_int · (1/3) Σ_j MSE(X̂_{t+6j h}, X_{t+6j h})`, `w_int = 1.0`, 단위는 L_f와 같은 delta 척도(중간 시각 오차를 물리 공간에서 24 h 오차와 같은 무게로 봄). 첫 lead에서만 건다(rollout 단계는 이전과 같음).
- **예보 경로는 그대로**: 보조 head는 학습 손실에만 쓰이고, 예보·rollout·Stage C·평가는 24 h 출력만 쓴다. 따라서 추론 비용 변화 0, 학습 비용 + 몇 %(목표 상태 3개 읽기와 선형 head 하나).
- ODE 적분기(stage8)로 중간 상태를 직접 만드는 방식은 V-Net을 lead마다 4번 불러야 해 CPU에서 4배 비용이라 쓰지 않는다(stage8c는 GPU 후보로 남김).

## 실험 조건

stage9b와 모든 것이 같다: 10년(2008–2017) 데이터, 기둥 입력(fcn10, 초기시각만), PIGT 1 layer + 연산자 토큰, 이전 변화 입력, 곡률 연산자 guide(stage9 guide·normaliser 재사용), Stage B 9000 step·rollout 2단계 3000 step부터, Stage C 잡음 제약 800 step, 평가 2019 90 초기시각 + 2020·2022 각 88 초기시각. guide 없는 짝도 같은 손실로 재학습. 실행 `runs/era5_64x32_s15b_inter/chain_s15b.sh`(약 13 h), 끝에 `scripts/decide_adoption.py`로 stage9b 대비 채택 판정(5개 lead 중 3개 이상 개선, 평균 ≥ 0.5 %).

## 비교와 판정

- 기준: stage9b(직전 통합 모델) + 두 run의 guide 없는 짝 + ClimODE 열 + 24 h CRPS + P=0 진단.
- 검증 로그에 중간 시각 head의 MSE(6/12/18 h)도 남긴다(선택 기준에는 쓰지 않음): 중간 시각을 실제로 배우는지 확인용.
- 채택이면 stage15b가 곧 새 통합 모델(stage9b + 중간 시각 손실)이 된다(감자 2026-11-05 23:51 KST "좋아지면 9단계에 합쳐주세요").
- 예상 위험: 보조 목표가 공유 표현 용량을 나눠 24 h 출력이 조금 나빠질 수 있다. 그때는 `--inter-weight`를 낮춘 변형을 별도 stage로 본다.

## 코드

`gpclimate/models/dynamics_vnet.py`(`VNetConfig.inter_times`, `head_inter`, `LieGuidedForecaster.inter_forecasts`), `gpclimate/data/era5_wb2.py`(`ERA5Pairs.state_after`), `scripts/train_eval.py`(`--inter-times`, `--inter-weight`, config 키 `inter_times`·`inter_weight`, 학습 `L_int`, 검증 `val_inter_mse`), 테스트 `tests/test_intermediate_loss.py`(지속성 초기화·예보 불변·시각 정렬·파이프라인·lead 검사).
