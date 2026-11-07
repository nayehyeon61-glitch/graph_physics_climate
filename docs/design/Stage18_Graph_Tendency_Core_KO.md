# stage18 graph tendency core 설계 (stage19 GPINN guide 포함)

작성 2026-11-07 21:20 KST. 요청:

- 감자 2026-11-07 20:20 KST (arXiv 2306.09792 GPINN 첨부): "gpinn은 원본 가이드 그리고 dynamic guide는 예측구조의 보조로서 추가해보면 어떨까요?"
- 감자 2026-11-07 20:31 KST: "우리의 dynamic guide가 실제로는 정적인 부분이 있는 것 같습니다. predict에 반영하는 guide 구조를 적용하는 것이 필요합니다. 특히 pg flowmatching의 tendency core를 graph version 및 물리가이드에 대한 graph transformer가 이해하도록 하는 게 필요할 것 같습니다."
- 결정 카드 2026-11-07 20:38 KST: "17b 취소, 17c 뒤 core".

## 1. stage17a dynamic guide에 남은 정적인 부분 (코드 확인)

| # | 정적인 부분 | 위치 | 해결 stage |
| --- | --- | --- | --- |
| 1 | guide 가중치는 한 시점 상태(복원 + 6 h tendency)로만 학습되고 예측 학습(Stage B) 중 고정 | `train_eval.py` Stage B `model.freeze_guide()` | stage18c (경로 위 guide 미세조정) |
| 2 | guide는 sub-step마다 다시 계산되지만 예측에는 상태 설명 token P로만 들어가고, guide tendency는 변수별 상수 γ(0 시작)로만 섞임 | `DynamicsVNet.forward`, `physics_drift` | **stage18** |
| 3 | 직전 변화량 prev(X_t − X_{t−24h})가 4개 sub-step 내내 t0 값 | `LieGuidedForecaster.integrate` | stage18b |
| 4 | V-Net 출력이 자유 증분이라 물리 구조(flux, source, 수송)를 거치지 않음 | `rate()` | **stage18** |

## 2. PG-Flow tendency core

PG-Flow(`nayehyeon61-glitch/physical_guide_flowmatching`, `docs/era5_15day_ko.md`, `src/physical_guide_flowmatching/spherical*.py`)의 결정론 core:

dq/dt = −∇·(V q) + K Δq + R_θ,  R_θ = −∇·J_θ(q, c) + S_θ(q, c)

- 수송 속도 V는 매 RHS 호출마다 `velocity(t, state)`로 다시 계산
- 학습 closure를 보존 flux 발산(면적 적분 불변)과 비보존 source로 분리, 둘 다 0으로 시작
- 구면 격자의 cell 면적과 face 길이를 쓰는 유한체적, `torchdiffeq` RK4

## 3. graph version (stage18)

매 6 h sub-step k에서

X_{k+1} = SL_{β⊙V_k}(X_k) − div_G J_θ + S_θ

| 항 | 정의 | 구현 |
| --- | --- | --- |
| 그래프 | 64×32 격자 cell = node, cell 경계 face = edge(동서 주기, 극은 닫힘). 면적 A_i, face 길이 ℓ_f는 구면 값 | `gpclimate/nn/graph_core.py` `GridFaceGraph` |
| div_G | div_i(J) = (Σ_{i에서 나가는 f} J_f ℓ_f − Σ_{i로 들어오는 f} J_f ℓ_f)/A_i → Σ_i A_i div_i = 0 정확 | `GridFaceGraph.divergence` |
| V_k | guide flow(층 바람 + 등변 보정, stage17c flow head)를 **현재 sub-step 상태에서 다시 계산** | `LiePhysicalGNN.core_parts` |
| 수송 | V를 β배 한 반-라그랑주 수송(벡터 평행 수송). 6 h sub-step에서 Euler flux형 수송은 극 근처 CFL이 10배 이상 초과(PG-Flow는 분 단위 sub-cycle)하므로 stage17c의 무조건 안정 SL을 씀 | `guide.transport(..., flow, beta)` |
| β | cell × 층 그룹(850 hPa, 500 hPa, 지상) 별 guide flow 사용 비율. V-Net embedding의 head, 0에서 시작, [−0.5, 2] | `DynamicsVNet.core_beta` |
| J_θ | face flux = ψ(f_i(n), f_j(n)) − ψ(f_j(−n), f_i(−n)), f(n) = [스칼라 특징, 벡터 특징의 법선·접선 성분]. 반대칭이라 **message passing 자체가 보존 유한체적 flux**, gauge 불변. 스칼라 5채널, 0 시작 | `FaceFluxHead`, `DynamicsVNet.core_flux` |
| S_θ | 기존 V-Net 출력(head_out)이 source | `core_rate` |

**PIGT가 core를 "이해"하는 방식.** guide의 core 항 [guide 수송 변화 SL_V(X)−X, guide source·Δt]를 매 sub-step

1. 격자 embedding에 더하고(0 시작 EqLinear, `GridLift(extra=...)`),
2. 모든 메시 수준으로 내려 **PIGT의 네 번째 token 원천 C**로 넣습니다(H/P/O 옆, 같은 softmax). C의 value는 0, logit 편향은 −4에서 시작해 학습된 stage9b V-Net의 기능을 거의 그대로 유지합니다(시작 시 C 주의 질량 약 0.5 %).

V-Net은 guide가 말하는 물리(어디로 얼마나 수송되는지, 무엇이 생기는지)를 매 단계 읽고, 그 위에서 β(얼마나 믿을지), J_θ(보존 재분배), S_θ(나머지)를 냅니다. guide 없는 짝(stage17a 것 재사용)에는 core가 없습니다.

**정적 γ 제거.** stage17c의 변수별 상수 이득은 없애고(ode_physics 0), guide의 기여는 상태에 따라 변하는 β와 C token 주의로만 들어갑니다.

## 4. 비교와 측정

- 같은 paired 미세조정: V-Net은 stage9b 가중치에서 1000 step, lr 3e-4, rollout 2, 경로 감독(6/12/18 h). guide는 stage17c의 것(몸통 stage9b, head 재학습).
- 직전 stage17c와 비교(`decide_adoption.py`, 5개 선행 중 3개 이상 + 평균 ≥ 0.5 %), stage17a·stage17-ctl·ClimODE·2020/2022도 함께.
- guide 의미 지표: β 평균·분포(어디서 guide flow를 믿는지), 항별 크기(수송/flux/source rms), PIGT C 주의 질량, guide만의 core 예보(physics_only), guide on/off 이득, flux 보존(면적 적분 0).
- 실행 `runs/era5_64x32_s18_core/chain_s18.sh`(CHAIN_S17C_DONE 대기, 표지 CHAIN_S18_DONE).

## 5. 다음 단계

- **stage18b 경로 기억 갱신** (구현, `VNetConfig.prev_path`, `--prev-path 1`, 체인 `runs/era5_64x32_s18b_prevpath/chain_s18b.sh`): sub-step k의 prev = X_{t+6k h−24 h}(첫 lead는 관측 과거 상태 `ERA5Pairs.prev_path`, 이후 lead는 앞 lead의 경로 `LieGuidedForecaster.next_prev`), 24 h 변화량의 의미 유지. guide 없는 짝도 같은 입력이라 다시 미세조정.
- **stage18c 경로 위 guide head** (구현, `--guide-lr 3e-5 --guide-heads-only 1`, 체인 `runs/era5_64x32_s18c_guidepath/chain_s18c.sh`): stage18/18b 중 채택된 쪽 위에서 guide의 물리 head(격자 어댑터, flow, source)만 core 예보 경로에서 함께 학습(정적인 부분 1). 몸통과 특징 P는 고정합니다. stage12c에서 guide 전체를 lr 1e-4로 미세조정하자 guide가 두 번째 예보기로 바뀌어 24 h −4.3 %가 됐기 때문입니다.
- **stage18d core의 Laplace block** (구현, `VNetConfig.core_laplace`, `--core-laplace 1`, 체인 `runs/era5_64x32_s18d_laplace/chain_s18d.sh`; 감자 2026-11-07 21:32 KST, PhyMPGN ICLR 2025 `github.com/intell-sci-comput/PhyMPGN` "주위를 covering 하는 게 core 만들때 괜찮을것 같은데요"): stage18/18b/18c 중 채택된 쪽 core에 매 6 h sub-step마다 스칼라 채널의 확산 한 걸음을 더함. PhyMPGN의 Laplace block(식 4–5)은 노드 주위를 덮는 다각형(circumcenter를 이은 영역) 면적 d_i와 cotangent 가중치 w_ij = 쌍대 길이/원 길이를 씁니다. 직교 격자인 우리 face graph에서는 그 다각형이 cell(면적 A_i), 가중치가 face 길이/중심 거리 ℓ/d이므로 `GridFaceGraph.laplacian`이 바로 그 연산자입니다(구면조화함수 Y_1–Y_3에서 중위도 오차 3 % 이내, 테스트). 학습되는 부분은 cell·채널별 확산계수 K ≥ 0(V-Net embedding head, 거의 0인 K ≈ 700 m²/s에서 시작, 1000 step 안에 자랄 수 있게 이득 8). PhyMPGN의 연산자 보정 z_i 역할은 이미 반대칭 flux J_θ가 맡습니다. 극 근처 cell은 동서 face 간격이 30 km라 6 h 명시 확산이 불안정하므로 face 수 ν = K(ℓ/d)Δt/min(A_i, A_j)를 0.12에서 부드럽게 상한 → 새 극값이 생기지 않고 면적 적분은 정확히 보존. PG-Flow core의 KΔq 항을 되살리는 것이기도 합니다. BC padding은 구면에 경계가 없고 경도 주기가 이미 정확히 연결되어 있으며 극은 길이 0인 점이라 쓰지 않습니다. 비교 대상은 그 base, stage19는 18d 뒤로 이동.
- **stage19 GPINN guide**: PINN 잔차를 core 항과 guide에 soft loss로. 성립도(`scripts/pinn_equation_check.py`, 학습 160시점, mesh 0, 중위도 var(L−R)/var(L), 작을수록 성립):

| 식 | 중위도 | 판단 |
| --- | --- | --- |
| 500 hPa 지균 균형 | 0.03 | 사용 |
| 850 hPa 지균 균형(Φ850 추정) | 0.09 | 사용 |
| 500 hPa 지균 경향 균형 | 0.30 | 사용 |
| 온도풍(t850) | 0.37 (α 0.77 → 0.31) | 보정 계수와 함께 사용 |
| 층후 경향(z500 ← msl, t2m, t850) | 0.41 (α → 0.35) | 보정 계수와 함께 사용(z500 경향의 첫 물리 제약) |
| 500 hPa 순압 와도 방정식, 중간점 | 0.39 (전진 0.49) | 사용 |
| q700 이류, 중간점 | 0.53 | 약하게 |
| t850 이류, 중간점 | 0.83 | 제외 |
| 500 hPa 운동량 방정식 그대로 | 1.97 | 제외(stage5a 실패 원인) |

  모든 식에서 중간점 계산이 전진 계산보다 낫고, 열대에서는 균형식이 성립하지 않습니다(마스크). 그래프 임베딩은 균일 구면 Laplacian 고유벡터(=구면조화함수) 대신 지형·해륙 가중 Laplacian 고유벡터를 씁니다. 예측 경로 상태를 PINN의 collocation 점으로 씁니다.

## 6. 위험

- β·J_θ·C가 0에서 시작하므로 1000 step 미세조정 안에 충분히 자라지 않으면 stage17a와 차이가 작게 나올 수 있습니다. 로그의 β와 항별 크기로 바로 확인합니다.
- SL 수송은 질량 보존이 아닙니다. 보존은 학습 flux 부분에서만 정확하고, 전구 평균 drift를 따로 보고합니다.
- 비용: sub-step마다 SL 2회(guide 입력용, β 적용)와 face flux가 더해집니다(작음).
