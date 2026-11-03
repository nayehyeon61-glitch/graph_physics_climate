# stage8 — Graph Neural ODE 예측층: 학습된 물리 위에 ODE 구조 겹치기

감자 제안(2026-11-03 04:39 UTC): "예측항 쪽을 건드려 보자. GNN neural ODE(Poli et al. 2019, arXiv 1911.07532; jaketae/graph-neural-ode)를 통해 ODE의 구조를 학습된 physics 위에 겹치도록 구현하면 어떨까."

## 0. 출처에서 가져오는 것

**논문 (Graph Neural Ordinary Differential Equations, Poli·Massaroli·Park·Yamashita·Asama·Park 2019).**
- 연속 깊이 GNN: Ḣ(s) = F_G(s, H(s), Θ), H(0) = X_e, 출력 Y = X_e + ∫ F_G dτ. GNN 층의 잔차 연결을 ODE의 벡터장으로 읽고 forward에 수치 solver(고정 step RK2/RK4, 적응 step dopri5)를 내장한다.
- 시공간 설정에서는 깊이 s ≡ 물리 시간 t: H(t+Δt) = H(t) + ∫_t^{t+Δt} F(τ, H(τ), Θ) dτ. 자기회귀 GDE는 "관측 사이의 연속 흐름 + 관측 시점의 이산 점프"(hybrid automata)로, 관측 간격이 불규칙해도 적분 구간만 바꾸면 된다.
- 실험: 다체 궤적 외삽에서 GDE가 Neural ODE·정적 MLP보다 외삽 step이 늘어날 때 오차 증가가 훨씬 느렸고(그래프 구조 + 연속 동역학의 결합 효과), 교통 예측(PeMS, 불규칙 샘플링)에서 GCGRU보다 좋았다. 정적 노드 분류에서는 고정 step RK가 적응 step보다 일반화가 좋았다(NFE 과다 → 과적합).

**레포 (jaketae/graph-neural-ode, "Graph Neural ODE++").** 수업 과제(CPSC 483) 구현: torchdiffeq solver 선택(euler … dopri8, adjoint 옵션)과 두 가지 학습 휴리스틱(linear simplex refinement, consistency modeling)을 Cora/Citeseer/Pubmed 노드 분류에 적용. 예보 코드는 아니므로 코드를 가져오지는 않고, 아래 세 가지 구조만 가져온다.

1. **예측층 = 벡터장 + 적분기**: one-shot 24 h 변화량 대신 sub-step 적분.
2. **고정 step solver를 forward에 내장**(Euler/Heun; 등변 Field 자료형 유지, torchdiffeq 불필요).
3. **hybrid arc**: 적분 경로의 관측 시점(6 h)마다 감독할 수 있다(stage8c).

## 1. 현재 예측층과 stage8 예측층

현재(stage1–stage5, 설계 §4.4): X̄_{t+24h} = X_t + δ · Δ_φ(X_t, P(X_t), c_t) — V-Net 한 번 평가, δ = 24 h 변화의 정규화 scale.

stage8 (`VNetConfig.integrator="ode"`): K개의 고정 sub-step(길이 L/K data step, L = 4 × 6 h)으로

X_0 = X_t,  X_{k+1} = X_k + **T_guide(X_k) · τ · (L/K)** + δ · f_φ(X_k, P(X_k), c_t) / K,  X̄ = X_K.

- T_guide: 동결된 guide의 6 h 경향(S2의 경향 head, 또는 stage5a guide의 F(X)+S_θ를 메시 0 → 격자로 전송), τ = 6 h 경향의 정규화 scale. `ode_physics=1`일 때만 포함(stage8b). **물리 drift가 알려진 항으로 먼저 들어가고 V-Net은 보정 f_φ만 학습**한다. head는 0으로 초기화돼 있으므로 학습 시작점은 S8a에서 persistence, S8b에서 **순수 물리 적분**이다.
- P(X_k): 물리 특징을 sub-step마다 현재 상태에서 재계산 → Lie층(guide)과 예측층이 적분 경로 전체에서 결합된다(GDE의 "구조 의존 벡터장"). gradient는 guide의 입력을 통해 흐르고 guide 가중치는 동결(설계 §4.1 유지).
- solver: 명시적 Euler(기본) 또는 Heun(sub-step당 2회 평가). 조건 c_t(TOA t, t+24h)는 24 h 내에서 고정 — 근사.
- 출력 계약은 그대로 (X̄, d = (X̄ − X_t)/δ) → rollout 손실, PDE 잔차 손실(stage5b), Stage C, 평가 코드가 모두 그대로 동작한다.
- `physics_only`: V-Net을 끄고 drift만 적분한 예보. stage8b 평가에 `physics_only` 행으로 추가(guide 경향의 24–120 h 품질 진단, 비용 0).

## 2. 가설

- H1 (stage8a): 같은 파라미터·같은 손실에서 K-step 적분(물리 특징 재계산)이 one-shot delta보다 72–120 h 오차 증가를 줄인다. 근거: GDE 외삽 실험, 그리고 학습 구조(적분)가 평가 구조(자기회귀)와 같아진다.
- H2 (stage8b): 물리 drift를 알려진 항으로 두면 V-Net이 보정만 학습 → 24 h 단기 개선 + guide 의존도 상승(P=0·drift=0 진단). stage3 guide의 6 h 경향 검증 손실은 0.356(persistence 1.0)이라 drift가 변동의 약 64 %를 설명한다.
- H3 (stage8c, 후속): 6 h 중간 관측 감독(hybrid arc)으로 적분 경로가 물리적으로 맞춰지면 rollout 안정성이 더 좋아진다. K = 4가 필요하다.

## 3. Stage 분할 (규칙: 한 stage에 한 변경, 바로 앞 stage와 같은 데이터·step)

| stage | 변경 | 비교 대상 | run |
|---|---|---|---|
| stage8a | 예측층 delta → ODE(K = 2 × 12 h, Euler, drift 없음). stage3 guide 재사용(Stage A skip), 게이트 융합, 2-step rollout from 500 등 S3와 동일 | stage3 | `runs/era5_64x32_s8a_ode` |
| stage8b | + `ode_physics=1` (drift = stage3 guide의 6 h 경향 head) + `physics_only` 진단 | stage8a | `runs/era5_64x32_s8b_ode_phys` |
| stage8c (후속) | + 6 h 중간 관측 감독 L_mid, K = 4 | stage8b | GPU 또는 야간 |
| stage8d (후속) | stage5a 닫힘 guide의 F+S_θ drift / stage4 PIGT 융합 + ODE | stage8b | — |

guide 없는 ODE baseline(`TRAIN_NOGUIDE=1`)은 S8이 S3를 이기면 추가한다(stage8 run에는 S3의 delta형 guide-free V-Net을 복사하지 않음: `COPY_NOGUIDE=0`).

## 4. 비용 (CPU 4코어)

1 thread·nice에서 측정한 fwd+bwd 비율: delta 1.0, ODE K=2+drift 2.4×, K=4 5.4×, K=4 drift 없음 4.7× (sub-step 2부터 guide 입력으로 gradient가 흐르므로 K배보다 크다). Stage B 1500 step(500 이후 2-step rollout) 예상: K=2 ≈ 3 h, K=4 ≈ 6.7 h. 그래서 CPU stage는 K=2, K=4(6 h sub-step, stage8c)는 GPU/야간용. 평가·진단 시간도 같은 배수.

## 5. 위험과 완화

- 12 h Euler step의 안정성(drift = 2·T_6h): 학습 초기에 발산하면 Heun(`--ode-solver heun`) 또는 drift 계수 β_T 학습으로 완화.
- 조건 고정 근사(TOA): K=4 이상에서 중간 시각 TOA를 넣는 확장 가능(`ERA5Pairs`에 6 h 시각이 모두 있음).
- 경향 head 품질: `physics_only` 행이 24–120 h에서 persistence보다 나쁘면 drift가 오히려 해가 될 수 있음 → stage8a/stage8b 비교로 판단.
- 비용 증가 2.4×: 같은 step 수를 유지하므로 결과 비교는 공정하지만 벽시계 시간이 길다.

## 6. 다른 stage와의 관계

- stage6(5층 설계)의 Layer 5 "분할 적분기"를 현재 모델에서 먼저 실현한 것이다. S6에서는 sub-step의 수송을 semi-Lagrangian·DEC flux로 바꾸고 Layer 4의 수송형 잡음을 같은 적분기에 넣는다.
- "물리 drift + 학습 보정" 계열: ClimODE(이류 ODE + NN, 2차), NeuralGCM(역학 코어 + 학습 물리). 우리 drift는 학습된(동결) 물리 guide의 경향이라는 점이 다르며, stage5a guide를 쓰면 해석 가능한 F(X)가 drift의 일부가 된다.

## 7. 구현

- `gpclimate/models/dynamics_vnet.py`: `VNetConfig.integrator/ode_steps/ode_solver/ode_physics`; `LieGuidedForecaster.integrate/rate/physics_drift/physics_only`, `tend_scale_*`, `lead_steps`.
- `gpclimate/models/lie_physical_gnn.py`: `physics_tendency` (격자 6 h 경향: 경향 head 또는 F+S_θ 전송).
- `scripts/train_eval.py`: `--integrator ode --ode-steps K --ode-solver {euler,heun} --ode-physics {0,1}` (config.json 저장), Stage B 로그, 평가의 `physics_only` 행.
- `scripts/run_guide_ablation.sh`: `INTEGRATOR/ODE_STEPS/ODE_SOLVER/ODE_PHYSICS`, `COPY_NOGUIDE`, GUIDE_FROM 복사 config에 적분기 키 반영.
- `tests/test_ode_dynamics.py`: K=1·drift 없음 = delta 모델, head 0 = 순수 물리 적분 = `physics_only`, Heun+drift 게이지 등변성·gradient 경로, stage5a 닫힘 guide drift 형태.
- 체인: `runs/era5_64x32_s8a_ode/chain_s8.sh` (stage7 종료 `CHAIN_S7_DONE` 후 stage8a → stage8b).
