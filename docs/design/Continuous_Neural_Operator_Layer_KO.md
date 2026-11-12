# 연속 물리 층: Lie → 이산 → 연속 (neural operator) — stage13 설계

출처: 감자 제안 2026-11-04 09:28 KST — "GNN은 쉽게 겹칠 수 있지만 **이산적**이라는 한계가 있다. 이산 물리뿐 아니라 **연속 물리의 층을 neural operator로** 구성해 Lie → 이산 → 연속 → … 을 추가하자. 즉 **에너지 연속성**을 추가하는 것" + "https://github.com/neuraloperator/neuraloperator 이걸 써서".

## 0. 한 줄 판단

동의한다. 같은 날 아침의 오차 구조 진단(`reports/error_diagnostics_era5_64x32_physops_roll.md`)이 보여준 stage3의 약점 — 24 h부터 동서 파수 8–15의 파워가 관측의 73 %(ClimODE 82 %)로 깎이고 120 h에는 43 %(80 %), 반대로 격자규모(파수 16–32)는 1.6배로 자라는 것 — 은 **국소·이산 연산자(메시 1-hop 핵 + 격자↔메시 보간)만으로 전구 규모의 에너지 분포를 다루는** 모델의 전형적 증상이다. 함수 공간에서 정의된 전역 연산자(neural operator)는 바로 이 빈틈을 메우는 층이고, 우리 구조의 Lie 군(게이지) 층·이산(DEC/메시) 층과 충돌하지 않는다. 그래서 (i) 메시 V-Net 뒤에 **구면 neural operator 층**을 residual로 붙이는 stage13a, (ii) 그 위에 **스펙트럼 에너지 연속성 손실**을 더하는 stage13b로 나눠 실행한다(한 stage 한 변경). 라이브러리는 요청대로 `neuraloperator` 2.0.0(+ `torch-harmonics`, 구면 조화 변환)을 쓴다.

## 1. 제안의 해석

| 층 | 수학적 대상 | 우리 코드 | 성격 |
|---|---|---|---|
| Lie | SO(2) 게이지 등변 표현, 접속·곡률(stage11) | `LiePhysicalGNN`(guide), `MeshComplex` | 국소 프레임의 대칭성 |
| 이산 | 심플리셜 복합체 위의 코체인·DEC 연산자, 메시 메시지 전달 | `DynamicsVNet`(3-level 메시 U-Net), `SimplicialBlock`(stage11b) | 국소(1-hop), 해상도에 묶임 |
| **연속(신규)** | 함수 공간의 핵 적분 연산자 (K u)(x) = ∫ κ(x, y) u(y) dy — Fourier/구면조화 공간의 곱셈자 | `ContinuousOperatorBlock` (`gpclimate/nn/neural_operator.py`) | 전역, **이산화 불변**(같은 연산자를 다른 해상도에 적용 가능) |

"에너지 연속성"은 두 뜻으로 읽고 둘 다 넣는다.

1. **규모 간 에너지의 연속성**: 해상(메시가 표현하는) 규모와 그 아래 규모 사이에 인위적인 단절이 없어야 한다 — 예보의 에너지 스펙트럼이 모든 파수 대역에서 관측을 따라야 한다. → stage13b의 스펙트럼 에너지 손실(3절).
2. **연속체(함수) 수준의 연산자**: 격자점이 아니라 함수에 작용하는 층이므로 64×32에서 학습한 곱셈자 κ_l을 1.5°(KAI 67채널) 격자에도 그대로 적용할 수 있다(FNO/SFNO의 이산화 불변성). → stage13a의 층 자체; 해상도 전이는 KAI 전환 때 검증.

다른 해석(전체 에너지 수지 ∂E/∂t + ∇·F = S를 제약으로 거는 것)은 stage13c 후보로 남긴다(11변수에는 연직 적분 에너지가 없어 근사 수지만 가능; stage5c의 사영과 같은 방식으로 넣을 수 있음).

## 2. 구조

```
X_t, c ──► Lie guide P(X_t, c) ──► 메시 V-Net (이산) ──► δ_mesh (격자로 복원)
   │                                                      │
   └──────────── [X_t, c, δ_mesh, (sin φ, cos φ cos λ, cos φ sin λ)] ──► 연속 연산자 ──► δ_cont
                                                                               X̂ = X_t + δ_mesh + δ_cont
```

- 입력(격자점당): 정규화 상태 11 + 조건 7(스칼라 5, 벡터 1쌍) + V-Net δ 11 + 구면 좌표 3 = 32채널 → 1×1 lift → hidden 32.
- 연산자 블록 ×2(`neuralop.layers.fno_block.FNOBlocks`): h ← σ(W h + K h), K h = SHT⁻¹( κ_l · SHT(h) ), κ_l ∈ ℂ^{32×32}는 **차수 l(< 32)마다 하나**(`SphericalConv`, torch-harmonics `RealSHT`; 선형부는 SO(3) 등변), 채널 MLP(확장 0.5)와 soft-gating skip은 라이브러리 기본.
- 출력: 1×1 → δ 보정 11채널, **마지막 층 0-초기화** → 새로 만든 모델은 stage3과 정확히 같은 예보에서 출발한다.
- 게이지 처리: 격자 벡터는 캐노니컬 (동, 북) 프레임에서 연산하고, 격자 프레임이 회전된 경우(`GraphView.extras["grid_gauge"]`)에는 입력을 되돌리고 출력을 다시 회전한다 → 전체 예보기의 게이지 등변성 유지(`tests/test_neural_operator.py`, 오차 < 2e-5).
- 구면 조화 변환의 격자: torch-harmonics의 "equiangular" 노드는 극을 포함하고 우리 격자는 셀 중심(±87.19°)이라 정확히 같은 구적은 아니다. 층이 residual·학습형이므로 이 불일치는 κ_l이 흡수하며, 0-초기화 덕에 초기 모델은 영향을 받지 않는다. FFT 변형(`neural_op=fft`, AFNO/FourCastNet식 2-D FFT, 위도를 주기로 취급)도 구현해 두었다(비교용).

## 3. 에너지 연속성 손실 (stage13b)

대역 파워 P_b(f) = 평균_{c, φ(cos 가중), 배치} |FFT_λ f_c(φ, ·)|²_{k∈b}, b ∈ {1–3, 4–7, 8–15, 16–32} (벡터는 동·북 성분 → 운동에너지 스펙트럼).

L_spec = 평균_{c, b} [ log( P_b(X̂) / P_b(X) ) ]²

0이면 모든 대역에서 예보가 관측과 같은 에너지를 가진다; 감쇠(평활)도 과잉(격자 잡음)도 같은 크기로 벌한다. Stage B에서 1-step과 rollout 예보 모두에 적용, 가중 0.1(초기 L_spec ≈ 0.3–0.7, L_f ≈ 0.3). 구현 `gpclimate/training/losses.py::spectral_energy_loss`, CLI `--spec-energy-weight`.

## 4. 비용

CPU 1 thread, batch 4, 32×64, fwd+bwd (V-Net 전체 ≈ 3.9–4.6 s, 554,665 매개변수):

| 변형 | 모드 | hidden / 층 | 매개변수 | fwd+bwd |
|---|---|---|---|---|
| 구면(sht) | l < 32, m < 32 | 16 / 2 | 34,891 | 0.033 s |
| **구면(sht), stage13a** | l < 32, m < 32 | 32 / 2 | 137,867 | 0.057 s (+1.4 %) |
| 구면(sht) | l < 32 | 32 / 3 | 205,563 | 0.086 s |
| FFT | (16, 32) | 16 / 2 | 141,387 | 0.031 s |

구면 변형은 차수마다 하나의 곱셈자라 FFT 변형의 1/4 매개변수로 전 모드를 덮는다. 학습 시간 증가는 2 % 미만.

## 5. stage 계획

| stage | 변경(한 가지) | 기준 | run / 체인 |
|---|---|---|---|
| **stage13a** | 메시 V-Net 뒤 구면 neural operator 층(hidden 32, 2블록, l < 32), guide 없는 baseline도 같은 층 | stage3 (+ ClimODE) | `runs/era5_64x32_s13a_sfno/chain_s13a.sh` (stage12d 뒤, ≈3 h) |
| **stage13b** | stage13a + 스펙트럼 에너지 손실 0.1 | stage13a (+ stage3, ClimODE) | `runs/era5_64x32_s13b_sfno_spec/chain_s13b.sh` (stage13a 뒤, ≈3 h) |
| stage13c (후보) | 전체 에너지 수지 제약(사영 또는 벌칙) 또는 FFT 변형 대조 | stage13b | 결과 보고 결정 |

공통: 데이터 2016–2017, stage3 guide·정규화기 재사용(Stage A 생략), Stage B 1500 step, rollout 2(500부터), 평가 2019 90개 × 5 lead. 채택 기준: 11변수 평균 개선율 5 lead 중 3개 이상 +, P=0 열화가 stage3의 +102 % 아래로 무너지지 않음, 그리고 **진단 지표**(파수 8–15 파워비, 격자규모 파워비, 진폭비)가 stage3보다 1에 가까울 것(`scripts/diagnose_errors.py --run <run>`). 채택되면 stage9 통합 구조에 들어가고, KAI 1.5° 전환 때 κ_l 전이(해상도 불변성)를 별도 stage로 확인한다.

예상 시각(KST, CPU 직렬 대기열): stage12d ≈ 11-05 09:15 → stage13a ≈ 12:30 → stage13b ≈ 15:45.

## 6. 위험

- 스펙트럼 손실이 거짓 소규모 파워를 만들어 RMSE를 올릴 수 있다(스펙트럼 정합 손실의 알려진 부작용) → stage13b는 RMSE와 진단을 함께 보고, 가중 0.1에서 악화되면 0.03으로 한 번 더(stage13b′).
- 구면 조화 변환의 구적 불일치(2절) → 0-초기화 residual로 초기 영향 없음; 필요하면 셀 중심 격자용 르장드르 행렬을 직접 구성.
- 층이 메시 GNN의 역할을 대신 흡수해 guide 의존도(P=0)가 떨어질 수 있음 → P=0 진단을 채택 기준에 포함.
- 의존성: `neuraloperator` 2.0.0, `torch-harmonics`(pip, 설치 ≈ 10분 빌드). `requirements`에 선택 의존성으로 적는다.

## 7. 참고

Li et al. 2021 (Fourier Neural Operator); Bonev et al. 2023 (Spherical FNO); Kossaifi et al. 2024 (neuraloperator 라이브러리, github.com/neuraloperator/neuraloperator); Pathak et al. 2022 (FourCastNet/AFNO); 우리 문서 `reports/IMPROVEMENT_PLAN_KO.md`(진단), `docs/design/Five_Layer_Geometric_Stochastic_GNN_Plan_KO.md`(5층 계획 — 연속 층은 그 1층(flux 핵)·5층(적분기)과 독립이며 stage9 통합 시 함께 둘 수 있다).
