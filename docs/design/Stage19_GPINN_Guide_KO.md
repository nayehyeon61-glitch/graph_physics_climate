# stage19 GPINN guide 설계

작성 2026-11-07 21:25 KST. 요청:

- 감자 2026-11-07 20:20 KST (arXiv 2306.09792 GPINN 첨부): "이 논문과 같은 pinn의 구조와 예측 dynamics guide를 위한 physics를 더해보면 어떨까요? gpinn은 원본 가이드 그리고 dynamic guide는 예측구조의 보조로서 추가해보면 어떨까요?"
- 감자 2026-11-07 21:18 KST: "https://github.com/MTMLab/GNN-PINN graph pinn은 이걸 참고하면 괜찮을까요?" (답: 개념 참고만, 아래 §4)

stage18 계열(18 → 18b → 18c, `Stage18_Graph_Tendency_Core_KO.md`)이 끝난 뒤, 체인이 채택한 쪽 위에서 시작합니다.

## 1. GPINN에서 가져오는 것

| GPINN 요소 | 우리 구조에서 | 구현 |
| --- | --- | --- |
| 물리 잔차 손실(PDE residual, soft) | ERA5에서 실제로 성립하는 관계식의 잔차, 영역 평균 제곱을 ERA5 var(L)로 나눔 | `gpclimate/training/pinn.py` `PINNResiduals` |
| collocation 점 | Stage A: guide의 복원 상태와 그 6 h core 한 걸음. Stage B(19c): 예보 경로의 모든 6 h 상태 | `train_eval.py` Stage A / Stage B |
| 그래프 Laplacian 고유벡터 입력(Fiedler 벡터 등) | 지형·해안 가중 격자 Laplacian의 고유벡터 8개를 guide 입력에 추가(0 시작 어댑터) | `gpclimate/geometry/graph_embedding.py`, `GuideConfig.graph_embed` |

## 2. 물리 잔차: 성립하는 식만

`scripts/pinn_equation_check.py`(학습 160시점, mesh 0, 중위도 |lat| > 20°, var(L − R)/var(L), 작을수록 잘 성립):

| 관계식 | 종류 | L | R | 중위도 비율 | 영역 |
| --- | --- | --- | --- | --- | --- |
| geo500 | 상태 | u500 | i ∇z500 / f | 0.03 | 중위도 |
| geo850 | 상태 | u850 | i ∇Φ850 / f (Φ850 = R T̄ ln(msl/850 hPa)) | 0.09 | 중위도 |
| thermal | 상태 | u500 − u850 | α i R ln(850/500) ∇T850 / f, α 0.77 | 0.31 | 중위도 |
| geo500_tend | 6 h 변화 | Δu500 | i ∇Δz500 / f | 0.30 | 중위도 |
| hyps_tend | 6 h 변화 | Δz500 | α (ΔΦ850 + R ln(850/500) ΔT850), α 0.76 | 0.35 | **전구** |
| bve500 | 6 h 변화 | Δζ500 | −Δt V_mid · ∇(ζ + f)_mid (중간점) | 0.39 | 중위도 |

제외: 500 hPa 운동량 방정식 그대로(1.97, stage5a 실패 원인), t850 이류(0.83). 열대에서는 균형식과 와도식이 성립하지 않아 마스크합니다(f는 |lat| ≥ 5°로 하한).

손실 = Σ_관계식 mean_영역 |L − R|² / var_ERA5(L). 분석장처럼 균형 잡힌 예보는 위 비율 정도, 균형이 깨진 예보는 O(1)이 됩니다. var(L)은 학습 32쌍으로 한 번 맞춰 `pinn_scales.json`에 저장합니다. 모든 계산은 mesh 0에서 물리 단위, `physics_ops`의 등변 stencil을 쓰므로 gauge 증강과 무관합니다.

로그: Stage A 검증에서 guide 출력의 비율과 같은 시점 ERA5의 비율을 나란히(`PINN guide/ERA5 geo500 0.xx/0.03 …`), Stage B는 `L_pinn`.

## 3. 지형 가중 그래프 임베딩

균일 구면 Laplacian의 고유벡터는 구면조화함수라서 guide가 이미 가진 좌표 정보와 같습니다(테스트: 평평한 바다 행성에서 첫 3개 모드가 l = 1 조화함수를 0.95 이상 재현). 그래서 face 가중치를

w_f = (ℓ_f / d_f) · exp(−|Δh| / 1000 m) · (해륙 경계를 넘으면 0.25)

로 두어 산맥과 해안선을 넘는 연결을 약하게 합니다. 일반화 고유문제 L v = λ A v(A = cell 면적)의 상수 다음 8개 모드를 부호 고정, 면적 RMS 1로 정규화해 guide 격자 입력에 더합니다. 어댑터 `gemb_in`은 0에서 시작하므로 stage18 계열 guide의 기능은 그대로 유지되고(테스트로 확인), Stage A에서 head와 함께 학습됩니다(stage18c 체인이면 Stage B에서도 head로 취급).

## 4. MTMLab/GNN-PINN에서 참고한 점

분자 GCN(TensorFlow, RDKit)으로 물성을 예측하는 코드이고 PDE는 없습니다. "물리" 항은 포화(평형) 표본에서만 Gibbs 등식 H_v − T S_v = H_l − T S_l을 맞추는 대수 제약입니다. 코드나 모델은 가져오지 않고 두 가지 원칙만 참고합니다.

- **관계식이 성립하는 곳에만 벌점**: 포화 표본에만 거는 것처럼, 우리는 성립도 검사를 통과한 식만, 성립하는 위도대(중위도 / 전구)에만 겁니다.
- **단계 학습**: 사전학습 → 보조 모델 → PINN 미세조정 순서처럼, stage9b guide 몸통은 고정하고 head만 PINN과 함께 학습한 뒤 예보 경로(19c)로 넓힙니다.

## 5. 단계와 비교

| 단계 | 변화 하나 | 비교 대상 |
| --- | --- | --- |
| stage19a | guide Stage A(stage17c 방식: stage9b guide에서 head만, 1000 step, 교란 0.3)에 PINN 잔차 손실, 가중치 0.05 | stage18 계열 채택본(18c > 18b > 18) |
| stage19b | + 지형 가중 그래프 임베딩 8개 | stage19a |
| stage19c | + 예보 경로(6 h마다)에 같은 잔차, 가중치 0.02 | stage19b |

- 누적 구성(19b는 19a의 손실을 유지). 각 단계는 바로 앞 단계와 `decide_adoption.py`(5개 선행 중 3개 이상 + 평균 ≥ 0.5 %)로 판정하고, 기준(18 계열)과의 누적 비교도 함께 기록합니다.
- 각 단계: 짝 미세조정(V-Net은 stage9b에서 1000 step, lr 3e-4, rollout 2), guide 없는 짝은 기준의 것 재사용. 2019 평가 + 2020/2022, ClimODE 열 포함.
- 의미 지표: guide 출력의 잔차 비율(ERA5 대비), 예보 경로의 잔차 비율, physics_only(guide core만의 예보), guide on/off 이득, β와 항별 크기.
- 실행 `runs/era5_64x32_s19_gpinn/chain_s19.sh`(CHAIN_S18C_DONE 대기, 표지 CHAIN_S19_DONE). 단계별 run `runs/era5_64x32_s19{a,b,c}_gpinn`.

## 6. 위험

- 균형 손실이 너무 세면 guide가 분석장보다 더 균형 잡힌, 즉 과도하게 매끄러운 상태를 내 예보 기술을 잃을 수 있습니다. 잔차를 ERA5 var(L)로 나누고 가중치를 작게(0.05 / 0.02) 두었고, 로그에서 guide 비율이 ERA5 비율보다 크게 낮아지면 과적합 신호로 봅니다.
- 64×32(5.6°)에서는 균형식의 기울기가 거칠어 열대·극 근처 오차가 큽니다. 마스크와 f 하한으로 막지만 경계(20°) 근처는 민감할 수 있습니다.
- 비용: Stage A 한 step에 mesh 0 stencil 계산 3회(작음), 19c는 경로 상태마다 1회.
