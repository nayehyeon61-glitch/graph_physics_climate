# 5층 기하·확률 동역학 GNN — 현재 구조와의 대응, S6 단계 계획

출처: 감자 첨부 「Five-Layer Geometric Stochastic Dynamics GNN Research Design」(2026.11) + 메시지(2026-11-03 02:52 UTC) — PIGT가 잘되면 physical guide를 **1 discrete → 2 Lie group → 3 Lie algebra → 4 noise form → 5 differential solver** 의 5층 통합 GNN 학습 구조로 보강하자는 제안. 이 문서는 (i) 다섯 층을 지금 코드에 대응시키고, (ii) 무엇이 이미 있고 무엇이 새로 필요한지, (iii) stage 규칙(한 run에 한 가지 변경)에 맞춘 S6a–S6f 계획과 비용·위험을 정리한다.

## 0. 한 줄 판단

가능하고, 지금 구조의 자연스러운 다음 단계다. 다섯 층 중 **2층(Lie 군 표현)은 이미 있고**(S1–S2의 SO(2) 게이지 등변 guide), **1층은 S5a의 고정 역학 F가 절반**(이류형이지 보존형 flux가 아님), **3·4·5층은 새로** 만들어야 한다. 가장 큰 구조 변화는 5층이다: 지금은 24 h를 한 번에 맞히는 사상(X_{t+24h} = X_t + V-Net 출력)인데, 5층은 6 h(또는 더 짧은) 부분단계 적분기 안에서 1·3·4층의 연산자를 **분할 적분**한다. 모든 층이 한꺼번에 바뀌므로 S6a–S6f로 나누고, 설계서의 실험 A–F가 그대로 stage가 된다. 적분기 부분단계 때문에 CPU 비용이 4–6배라 S6c부터는 GPU(KAI 1.5° 전환과 함께)를 전제로 한다.

## 1. 층별 대응표

| 층 | 설계서의 역할 | 지금 코드 | 상태 | S6에서 할 일 |
|---|---|---|---|---|
| 1 Discrete physical core, F_disc(q, Γ) | 보존형 flux + 외력 + 필수 동역학, F_ij = −F_ji | `gpclimate/nn/primitive2d.py`의 고정 역학 F: 이류·코리올리·기압경도(보정계수), 1-ring 최소제곱 스텐실의 **이류형** | 부분 | **flux 형**으로 다시 쓰기: 이중(Voronoi) 셀 면적 A_i, 간선 법선·길이, 간선 flux F_ij = −F_ji(중심 또는 upwind), 질량·수분이 구조적으로 보존. 코리올리·기압경도는 노드 항으로 유지 |
| 2 Lie group representation, h = E_H(q, Γ) | 군 작용 아래 일관되게 변환하는 특징 | `LiePhysicalGNN`(SO(2) 게이지 등변, P0·P1·P2·P1u·P0u), 연산자 입력, 등변성 오차 < 2e-5(테스트 22개) | **있음** | 그대로 사용. SO(3) 전역 회전은 메시 재표본화 없이는 근사 — 설계서의 경계조건과 같은 입장. 새 진단: 회전+재표본화 오차 |
| 3 Lie algebra generator, ξ_θ = Σ α_k X_k | 허용된 변화 방향을 만드는 drift, 예: streamfunction ψ_θ → 면적 보존 흐름 | 없음. V-Net의 Δ-head는 **무제약 잔차**, S5a의 S_θ는 closure(소스항) | 신규 | 생성원 head: 메시 0에서 ψ_θ(선택적으로 속도퍼텐셜 χ_θ)를 읽고 u_ψ = k×∇ψ(비발산)로 모든 변수를 수송하는 drift −u_ψ·∇q. PIGT/V-Net 특징에서 읽음. 게이지 회전(`so2.py`)과 물리 수송(`physics_ops.py`)이 이미 분리돼 있어 설계서의 "가장 중요한 주의점"(프레임 변경 ≠ 대기 이동)을 만족 |
| 4 Geometric noise form, B_{θ,r}(q) | 공간 상관 확률 벡터장, 생성원과 같은 기저 | Stage C residual graph diffusion = **상태 공간의 일반 잡음**(설계서 실험 D). S5c는 사영으로 허용조건 | 대체물 있음 | **수송형 잡음**: 확률 streamfunction ψ_r (R개 모드, 조건부 진폭 σ_r(h)) → dq = −(u_ψ dt + Σ_r u_{ψ_r} ∘ dW_r)·∇q. 수송형이므로 q ≥ 0, 극값, 질량, 수분이 **구조적으로** 보존 → S5c의 사영이 필요 없어짐. 비보존 물리(가열·마찰·확산)의 불확실성은 closure 잡음으로 따로 |
| 5 Differential solver | ODE/SDE 적분, rollout, 앙상블 | 없음(24 h 한 번 사상 + 2-step rollout 손실) | 신규 | 6 h 부분단계(×4) 분할 적분: A F_disc(보존형, Heun/SSP + flux limiter) → B Lie 흐름(u_ψ 수송) → C 확률 흐름(Stratonovich, Heun형 확률 적분 또는 Itô drift 보정) → D closure S_θ. Δt 민감도·분할 차수·10일 안정성 검증 |

설계서의 "계층 간 계약"에 대한 우리 답: 2층은 특징만 내고 변화량을 더하지 않는다(지금도 guide 출력은 V-Net 입력일 뿐). 3층만 물리 생성원을 내고, 4층은 확률 생성 방향만 내며, 5층이 둘을 중복 없이 합친다. 1층과 3층의 "중복 dynamics" 위험은 5절.

## 2. 우리 11변수(64×32, 6 h)에서의 식

**1층 flux 형.** 이중 셀 i(면적 A_i), 간선 ij(길이 ℓ_ij, 법선 n_ij)에서 스칼라 q의 수송은

  dq_i/dt = −(1/A_i) Σ_j F_ij,  F_ij = ℓ_ij (u·n)_ij q_ij,  F_ij = −F_ji

(q_ij는 중심 평균 또는 upwind). 수송 바람은 **상태 변수인 바람**(u850, u500, u10; q700은 S5a처럼 log-p 보간 u700). 코리올리·기압경도는 노드 항(S5a의 보정계수에서 출발해 flux 형에 맞게 1회 재적합). 닫힌 영역의 내부 flux가 상쇄되므로 전구 질량·수분 예산이 수치적으로 0이 된다(S5c에서 사영으로 강제하던 것).

**3층 생성원.** h(2층 특징, PIGT이면 융합 후 특징)에서 메시 0의 게이지 불변 스칼라 ψ_θ를 읽고

  u_ψ = k × ∇ψ_θ,  drift_ψ(q) = −u_ψ·∇q (스칼라), 벡터 변수는 기존 벡터 이류 연산자(계량항 포함)

로 모든 채널을 같은 흐름으로 수송한다. 이것이 미분동형군의 무한소 생성원이 상태에 유도하는 drift이며, ∇·u_ψ = 0이라 면적(질량) 보존이 구조적이다. χ_θ(발산 성분)를 넣으면 보존이 깨지므로 넣더라도 질량 가중 벌칙과 함께 둔다.

**4층 수송형 잡음.** ψ_r = σ_r(h)·φ_r (φ_r: 메시 2의 저주파 고정 기저 또는 학습 기저, r = 1..R), Stratonovich SDE

  dq = −[u_phys + u_ψ]·∇q dt − Σ_r (u_{ψ_r}·∇q) ∘ dW_r + S_θ dt

멤버마다 독립 W_r. Itô로 구현하면 drift 보정 ½ Σ_r (u_{ψ_r}·∇)(u_{ψ_r}·∇) q 를 더한다(설계서 5층 주의). σ_r은 CRPS·spread-skill(L_prob)로 학습.

**5층 적분.** 부분단계 Δt = 6 h, Strang형 분할 A(Δt/2) B(Δt/2) C(Δt) B(Δt/2) A(Δt/2) + D(Δt). CFL 주의: 5.6° 셀 ≈ 625 km, 30 m/s × 6 h = 648 km → CFL ≈ 1. 그래서 (a) 1–2 h 부분단계(ClimODE는 1 h Euler), 또는 (b) **반라그랑주 수송**(출발점 보간; 기존 `transport_interpolate`/쌍선형 보간 기반) 중 하나가 필요하다. (b)를 권장한다 — 무조건 안정이고, "흐름 사상을 따라 당겨오기"가 곧 군 작용이라 3층의 의미와 일치한다.

## 3. 손실과 진단 (설계서 표 + 기존 진단 재사용)

| 손실 | 설계서 | 우리 |
|---|---|---|
| L_pred | lead 가중 RMSE/MAE | S3의 2-step rollout 손실 그대로(부분단계 적분 후 24 h·48 h) |
| L_cons | flux/예산 오류 | 1·3·4층 수송은 구조적으로 보존 → 0. closure S_θ의 예산 잔차만 보고 |
| L_eq | 변환 등변 오차 | 게이지 등변성은 구조적(테스트). SO(3) 회전+재표본화 오차를 진단으로 추가 |
| L_prob | CRPS·spread-skill·calibration | 기존 `wb2_scores`(CRPS, spread) + rank histogram 추가 |
| 기존 | — | P-zeroing·shuffled-P, PIGT 어텐션 질량, resolved fraction, PDE 잔차(S5b), 허용조건 위반량(S5c), S5c의 비편향 L_mean |

## 4. Stage 계획 (설계서 실험 A–F ↔ S6a–S6f; 각 stage는 바로 앞 stage와 같은 데이터·step으로 비교)

| stage | 구성 | 가설 / 질문 | 비교 대상 | 비용(CPU 추정) |
|---|---|---|---|---|
| S6a | 1층 flux 형 + 5층 적분기, 학습 없음(보정계수만) | 설계서 A "필수 물리만의 기준 성능". S5a 데이터 점검으로 미리 아는 것: F는 6 h 경향 분산의 9–29 %만 설명 → persistence보다 조금 나은 정도일 것. 핵심은 **보존·안정성 검증**(10일 rollout, Δt 민감도) | persistence, S5a의 F-only | 학습 없음, 평가만(분 단위) |
| S6b | S6a + 2층 guide 특징으로 closure S_θ(S5a/S5b guide)를 적분기 안에서 사용, V-Net 없음 | 적분기 안의 closure가 24 h 한 번 사상(S5b)보다 나은가 | S5b | Stage A 재학습 ≈ 20 min |
| S6c | S6b + 3층 ψ_θ 생성원 head(PIGT 특징에서) | **H1** Lie drift가 같은 파라미터의 무제약 잔차 head보다 안정·정확한가. 동일 용량 residual 변형(S6c′)을 반드시 함께 | S6b, S6c′ | Stage B 재학습: 부분단계 4개 × 2-step rollout ≈ 15–20 s/step → 1500 step ≈ 7 h CPU → **GPU 권장**(또는 cs 32·1000 step 축소판) |
| S6d | S6c + 상태 공간 잡음(Stage C diffusion = S5c-0/S5c) 적분기 안에서 | 설계서 D | S6c | ≈ 1 h |
| S6e | S6c + 4층 수송형 잡음 ψ_r | **H2** 기하 잡음이 가우시안/상태 잡음보다 CRPS·spread-skill·스펙트럼에서 나은가, 사영 없이 q ≥ 0·질량·수분이 지켜지는가 | S6d, S5c | ≈ 1 h |
| S6f | closure 제거 ablation(S6e − S_θ) | 설계서 F를 뒤집은 형태: 우리는 S_θ가 이미 있으므로 "closure가 필요함"을 제거로 증명. 6 h 경향의 70–90 %가 미해상이라 크게 나빠질 것으로 예상 | S6e | ≈ 1 h |

**H3**(통합 적분이 장기 rollout과 확률 예측을 동시에 개선)는 S6e vs S5c(같은 guide·같은 잡음 평가)로 판단한다.

## 5. 위험과 대응 (설계서 8쪽 표 + 우리 데이터에서 본 것)

- **중복 dynamics**(1층 flux와 3층 drift가 같은 이류를 이중 계산): 1층은 **상태 바람**으로 수송하고 3층 u_ψ는 보정 흐름만 만든다. ‖u_ψ‖ 크기 벌칙과 var(u_ψ)/var(u) 보고로 범위를 제한한다.
- **제약 과잉**(Lie 생성원만으로 가열·확산·습윤 표현 불가): S5a 점검에서 고정 역학이 설명하는 6 h 경향 분산은 9–29 %뿐이다. closure 없는 S6a–S6c는 약할 것이고, 설계서도 closure를 선택(F)으로 두었지만 우리 데이터에서는 필수다 → S6b부터 S_θ를 넣고 S6f로 필요성을 보인다. 설계서의 권고("논문 핵심을 보존형 동역학 + Lie drift + 구조 잡음으로 좁히고 군 표현은 공통 기반으로")에는 동의한다.
- **잡음 불일치**(CRPS는 좋아지는데 평균장·스펙트럼 훼손): S5c의 비편향 L_mean과 에너지 스펙트럼 진단을 재사용한다.
- **수치 안정성**: ClimODE 평가에서 본 것처럼(eval 모드에서 72 h 이후 발산) 연속 적분은 정규화·CFL·limiter에 민감하다. S6a에서 학습 없이 10일 rollout 안정성을 먼저 확인한다(설계서 Phase 0–1).
- **비용**: 적분기 부분단계가 guide·V-Net 호출을 4배로 늘린다. CPU에서는 S6a–S6b까지, S6c부터는 GPU.

## 6. 데이터

설계서는 16×32 z850/z500/z250/u850/v850/지형으로 시작할 것을 제안한다. 우리는 이미 64×32 11변수(지형·해륙·TOA 포함)로 S1–S5를 쌓았으므로 **장부 연속성을 위해 같은 데이터로 S6a–S6c**를 하고, 다층 예산(z250 등 연직 구조)은 KAI 67채널(1.5°)로 옮길 때 확장한다.

## 7. 착수 조건과 순서

1. PIGT(S4)와 S5a–S5c 결과가 나온 뒤 결정(오늘 08:00 UTC 예상). PIGT가 S3보다 나으면 3층 head는 PIGT 융합 특징에서, 아니면 게이트 융합 특징에서 읽는다. (중간 신호: step 1000에서 PIGT 검증 MSE 0.0872 vs S3 0.0891, 2-step 0.159 vs 0.164.)
2. S6a — 새 모듈 `gpclimate/nn/flux_discrete.py`(이중 셀·간선 flux·반라그랑주 수송)와 `gpclimate/models/integrator.py`(분할 적분기). 학습 없이 보존·안정성·CFL 검증.
3. S6b → S6c(GPU) → S6d/S6e → S6f. 각 stage마다 장부(`docs/EXPERIMENT_STAGES_KO.md`)에 가설·결과를 적는다.
