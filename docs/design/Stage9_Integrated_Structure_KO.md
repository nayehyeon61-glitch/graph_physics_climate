# stage9 통합 구조 — 이점이 확인된 요소만 모은 모델

요청: 감자 2026-11-03 14:11 KST "각 결과 확인하고 마지막에 통합 구조 즉 이점들을 모은 통합구조 작성(우리꺼 안에서)", 2026-11-05 13:22 KST "통합 9단계가 가장 기대되는데 언제 시작되나요?".
run: `runs/era5_64x32_s9_integrated` (체인 `chain_s9.sh`, 2026-11-05 13:28 KST 시작). 비교 기준: **stage7b**(같은 10년 데이터·같은 4500 step) + ClimODE.

## 1. 들어가는 요소

| 층 | 요소 | 근거 stage | 근거 수치(+ = 개선) |
|---|---|---|---|
| 학습식 | 10년 데이터(2008–2017), Stage B 4500 step, 2-step rollout 1500 step부터 | stage7b 채택 | stage7 대비 +17.6/+15.6/+13.0/+10.0/+7.3 %, ClimODE를 모든 lead에서 앞섬 |
| 학습식 | Stage A 예보상태 섭동(p = 0.3), Stage B에서 guide 동결 | stage3, stage12c | stage12c(guide 미세조정) −4.3…−1.9 % → 동결 유지 |
| V-Net | 폭 cs/cv 48/12 → 64/16 | stage12b 채택 후보 | +3.77/+3.31/+2.37/+1.64/+0.96 %, 11/11 변수 |
| V-Net | 융합: 게이트 → PIGT(물리 통합 graph transformer, 1 layer, 연산자 토큰) | stage4 채택 후보 | +1.0/+1.0/+0.6/+0.3/0.0 % |
| V-Net | 직전 24 h 관측 변화 X_t − X_{t−24h} 입력 | stage14a 채택 후보 | +2.01/+1.22/+0.84/+0.39/−0.07 % |
| guide | 고정 물리 연산자 + 경향 head에 접속·곡률 연산자 추가 | stage11a 채택 후보 | +1.52/+0.91/+0.38/−0.24/−0.42 %, guide 이득 24–72 h 두 배 |
| Noise | Stage C 잔차 graph diffusion + 사영(q700 ≥ 0, 질량·수분 보존) + 허용·평균 제약 | stage5c 채택 | CRPS +3.0 %, spread/skill 1.00 |

## 2. 빠지는 요소

| 요소 | stage | 이유 |
|---|---|---|
| PDE 잔차 손실 | stage5b | 코드상 closure guide(stage5a, 보류)가 있어야 켜지고, stage5b 전체가 stage3보다 −0.78…+0.06 % |
| ODE 예측층 | stage8a/8b | 보류·기각, 비용 2.25× |
| fcn21 출력 행렬 | stage10a/10b | 기각(guide 이득 ≈ 0) |
| 심플리셜 블록 | stage11b | 기각 |
| EMA | stage12a | 이득 ≈ 0 |
| 3-step rollout 커리큘럼 | stage12d | 조건부: 72–120 h +, 24–48 h − (ClimODE에 지는 구간을 더 나쁘게 함) → stage9b 변형 후보 |
| 구면 neural operator 층 | stage13a | guide와 겹쳐 guide 모델에서 이득 없음 |
| 스펙트럼 에너지 손실 | stage13b | RMSE 악화, 스펙트럼은 개선 → Stage C 현실성 후보 |
| 초기 증분 nudging | stage14b | 이득 ≈ 0, 비용 +22 % |
| fcn21 신규 채널을 t0 조건으로만 | (stage10c 후보) | 아직 구현·검증 안 함 |

## 3. 위험과 확인 방법

- 요소 대부분은 2년·1500 step에서 선별됐다. 10년·4500 step에서 이득이 남는지가 stage9의 첫 질문이다.
- PIGT, 이전 변화 입력, 넓은 V-Net은 모두 guide가 주던 정보와 일부 겹친다(stage14a에서 guide 이득 +1.40 → +0.47 %). 그래서 guide 없는 쌍을 같은 구성으로 학습해 guide 이득을 따로 잰다.
- 조합 실행은 합성 데이터 스모크 테스트(`tests/test_stage9_integration.py`)로 확인했다.

## 4. 학습 순서와 시간 (CPU 4코어)

1. guide 없는 V-Net 4500 step (guide 없는 쌍이자 Stage A 섭동 모델)
2. Stage A 1000 step (접속·곡률 guide, 섭동 p = 0.3)
3. Stage B 4500 step
4. Stage C 800 step (잡음 제약)
5. 평가(90 초기시각 결정론 + 24 초기시각 8멤버 앙상블), guide 진단

예상 약 9–10 시간(PIGT의 step 비용을 깨끗한 CPU에서 아직 재지 못해 범위로 둔다) → 결과 2026-11-05 23:00–24:00 KST. 끝나면 2020·2022 외부 벤치마크가 stage9를 포함해 자동으로 돌고, 그 뒤 stage7c(9000 step)가 돈다.

## 5. 다음

- stage9가 stage7b를 앞서면: stage7c 결과(9000 step)로 step 예산을 정하고, stage9b 후보(3-step 커리큘럼 후반 적용, 스펙트럼 손실을 Stage C에)로 넘어간다.
- 문헌 직접 비교(3단계): 최종 학습식을 2006–2015로 다시 학습해 2017–2018에서 ClimaX·FourCastNet·CoDiCast·WeatherODE 표와 맞춘다.
