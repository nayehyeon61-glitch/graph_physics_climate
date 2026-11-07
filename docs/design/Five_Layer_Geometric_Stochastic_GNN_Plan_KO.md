# 5층 기하·확률 동역학 GNN — 현재 구조와의 대응, stage6 단계 계획

출처: 감자 첨부 「Five-Layer Geometric Stochastic Dynamics GNN Research Design」(2026.11) + 메시지(2026-11-03 02:52 UTC) — PIGT가 잘되면 physical guide를 **1 discrete → 2 Lie group → 3 Lie algebra → 4 noise form → 5 differential solver** 의 5층 통합 GNN 학습 구조로 보강하자는 제안. 이 문서는 (i) 다섯 층을 지금 코드에 대응시키고, (ii) 무엇이 이미 있고 무엇이 새로 필요한지, (iii) stage 규칙(한 run에 한 가지 변경)에 맞춘 stage6a–stage6f 계획과 비용·위험을 정리한다.

## 0. 한 줄 판단

가능하고, 지금 구조의 자연스러운 다음 단계다. 다섯 층 중 **2층(Lie 군 표현)은 이미 있고**(stage1–S2의 SO(2) 게이지 등변 guide), **1층은 S5a의 고정 역학 F가 절반**(이류형이지 보존형 flux가 아님), **3·4·5층은 새로** 만들어야 한다. 가장 큰 구조 변화는 5층이다: 지금은 24 h를 한 번에 맞히는 사상(X_{t+24h} = X_t + V-Net 출력)인데, 5층은 6 h(또는 더 짧은) 부분단계 적분기 안에서 1·3·4층의 연산자를 **분할 적분**한다. 모든 층이 한꺼번에 바뀌므로 stage6a–S6f로 나누고, 설계서의 실험 A–F가 그대로 stage가 된다. 적분기 부분단계 때문에 CPU 비용이 4–6배라 S6c부터는 GPU(KAI 1.5° 전환과 함께)를 전제로 한다.

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
| 기존 | — | P-zeroing·shuffled-P, PIGT 어텐션 질량, resolved fraction, PDE 잔차(stage5b), 허용조건 위반량(stage5c), S5c의 비편향 L_mean |

## 4. Stage 계획 (설계서 실험 A–F ↔ stage6a–stage6f; 각 stage는 바로 앞 stage와 같은 데이터·step으로 비교)

| stage | 구성 | 가설 / 질문 | 비교 대상 | 비용(CPU 추정) |
|---|---|---|---|---|
| stage6a | 1층 flux 형 + 5층 적분기, 학습 없음(보정계수만) | 설계서 A "필수 물리만의 기준 성능". stage5a 데이터 점검으로 미리 아는 것: F는 6 h 경향 분산의 9–29 %만 설명 → persistence보다 조금 나은 정도일 것. 핵심은 **보존·안정성 검증**(10일 rollout, Δt 민감도) | persistence, S5a의 F-only | 학습 없음, 평가만(분 단위) |
| stage6b | stage6a + 2층 guide 특징으로 closure S_θ(stage5a/stage5b guide)를 적분기 안에서 사용, V-Net 없음 | 적분기 안의 closure가 24 h 한 번 사상(stage5b)보다 나은가 | stage5b | Stage A 재학습 ≈ 20 min |
| stage6c | stage6b + 3층 ψ_θ 생성원 head(PIGT 특징에서) | **H1** Lie drift가 같은 파라미터의 무제약 잔차 head보다 안정·정확한가. 동일 용량 residual 변형(stage6c′)을 반드시 함께 | stage6b, stage6c′ | Stage B 재학습: 부분단계 4개 × 2-step rollout ≈ 15–20 s/step → 1500 step ≈ 7 h CPU → **GPU 권장**(또는 cs 32·1000 step 축소판) |
| stage6d | stage6c + 상태 공간 잡음(Stage C diffusion = stage5c-0/stage5c) 적분기 안에서 | 설계서 D | stage6c | ≈ 1 h |
| stage6e | stage6c + 4층 수송형 잡음 ψ_r | **H2** 기하 잡음이 가우시안/상태 잡음보다 CRPS·spread-skill·스펙트럼에서 나은가, 사영 없이 q ≥ 0·질량·수분이 지켜지는가 | stage6d, stage5c | ≈ 1 h |
| stage6f | closure 제거 ablation(stage6e − S_θ) | 설계서 F를 뒤집은 형태: 우리는 S_θ가 이미 있으므로 "closure가 필요함"을 제거로 증명. 6 h 경향의 70–90 %가 미해상이라 크게 나빠질 것으로 예상 | stage6e | ≈ 1 h |

**H3**(통합 적분이 장기 rollout과 확률 예측을 동시에 개선)는 stage6e vs stage5c(같은 guide·같은 잡음 평가)로 판단한다.

## 5. 위험과 대응 (설계서 8쪽 표 + 우리 데이터에서 본 것)

- **중복 dynamics**(1층 flux와 3층 drift가 같은 이류를 이중 계산): 1층은 **상태 바람**으로 수송하고 3층 u_ψ는 보정 흐름만 만든다. ‖u_ψ‖ 크기 벌칙과 var(u_ψ)/var(u) 보고로 범위를 제한한다.
- **제약 과잉**(Lie 생성원만으로 가열·확산·습윤 표현 불가): stage5a 점검에서 고정 역학이 설명하는 6 h 경향 분산은 9–29 %뿐이다. closure 없는 stage6a–S6c는 약할 것이고, 설계서도 closure를 선택(F)으로 두었지만 우리 데이터에서는 필수다 → S6b부터 S_θ를 넣고 S6f로 필요성을 보인다. 설계서의 권고("논문 핵심을 보존형 동역학 + Lie drift + 구조 잡음으로 좁히고 군 표현은 공통 기반으로")에는 동의한다.
- **잡음 불일치**(CRPS는 좋아지는데 평균장·스펙트럼 훼손): S5c의 비편향 L_mean과 에너지 스펙트럼 진단을 재사용한다.
- **수치 안정성**: ClimODE 평가에서 본 것처럼(eval 모드에서 72 h 이후 발산) 연속 적분은 정규화·CFL·limiter에 민감하다. S6a에서 학습 없이 10일 rollout 안정성을 먼저 확인한다(설계서 Phase 0–1).
- **비용**: 적분기 부분단계가 guide·V-Net 호출을 4배로 늘린다. CPU에서는 stage6a–S6b까지, S6c부터는 GPU.

## 6. 데이터

설계서는 16×32 z850/z500/z250/u850/v850/지형으로 시작할 것을 제안한다. 우리는 이미 64×32 11변수(지형·해륙·TOA 포함)로 stage1–S5를 쌓았으므로 **장부 연속성을 위해 같은 데이터로 stage6a–stage6c**를 하고, 다층 예산(z250 등 연직 구조)은 KAI 67채널(1.5°)로 옮길 때 확장한다.

## 7. 착수 조건과 순서

1. PIGT(stage4)와 stage5a–stage5c 결과가 나온 뒤 결정(오늘 08:00 UTC 예상). PIGT가 S3보다 나으면 3층 head는 PIGT 융합 특징에서, 아니면 게이트 융합 특징에서 읽는다. (중간 신호: step 1000에서 PIGT 검증 MSE 0.0872 vs stage3 0.0891, 2-step 0.159 vs 0.164.)
2. stage6a — 새 모듈 `gpclimate/nn/flux_discrete.py`(이중 셀·간선 flux·반라그랑주 수송)와 `gpclimate/models/integrator.py`(분할 적분기). 학습 없이 보존·안정성·CFL 검증.
3. stage6b → stage6c(GPU) → stage6d/stage6e → stage6f. 각 stage마다 장부(`docs/EXPERIMENT_STAGES_KO.md`)에 가설·결과를 적는다.

## 8. 호몰로지 / 이산 외미분학(DEC) 보강 (감자 첨부 「homology_lie」·「homology_code」, 2026-11-03 03:42 UTC)

두 자료의 제안: (i) 그래프에 면을 채운 심플리셜 복합체 위에서 호지 라플라시안으로 각 층의 물리량을 위상적으로 사영·보정하고, (ii) 2층의 연결(리 군 변환)의 홀로노미(이산 곡률)로 연속성을 점검하며, (iii) 4층 잡음을 호지 고유기저로 제한하고, (iv) 5층은 Stratonovich SDE + 지수 사상 retraction으로 두라는 것. 도구로 TopoX(TopoNetX·TopoModelX), MPSN/CWN/EMPSN/TOGL을 들었다. 자료 자체가 지적한 반박 세 가지(층마다 면을 새로 채우면 미분 불가, 호몰로지는 계량을 못 고침, 잡음이 다양체 밖으로 밀어냄)에 모두 동의하며, 우리 구조에서는 다음과 같이 된다.

### 8.1 사실 확인 (`gpclimate/geometry/dec.py`, `tests/test_dec.py`)

- **리프팅이 필요 없다.** 우리 메시는 정이십면체 삼각분할 = 꼭짓점·간선·삼각형 면이 이미 있는 **심플리셜 2-복합체**다(`SphereGraphs.mesh_faces`). 자료의 수정 1("구조 생성은 1층에서 끝내고 중간층은 호지 사영만")은 자동으로 충족된다. 복합체는 고정이므로 미분 가능성 문제가 없다.
- **구면의 베티 수는 (1, 0, 1)이다.** 코바운더리 d₀(간선×꼭짓점), d₁(면×간선)을 만들어 d₁d₀ = 0을 정확히 확인했고, 계수(rank)로 b₀ = 1, b₁ = 0, b₂ = 1 (오일러 수 V − E + F = 2). **b₁ = 0이므로 접벡터장(간선 1-형식)의 호지 분해에 조화(harmonic) 성분이 없다** — 자료가 말한 "구멍 주위를 도는 호몰로지적 보존량"은 전구 구면에는 존재하지 않는다. 그 항은 구멍이 있는 영역(제한 영역, 태풍 주변 절단 영역)이나 장(場)의 부분수준집합 위상(persistent homology)에서만 의미가 있다. 따라서 전구에서는 호지 분해 = **Helmholtz 분해(속도퍼텐셜 χ + 유선함수 ψ)** 이고, 이것은 기상역학의 표준 도구다.
- **2층 연결의 홀로노미 = 면적(가우스–보네).** 기존 SO(2) 게이지 연결(간선의 평행이동 각 α)을 각 삼각형 면을 따라 한 바퀴 합하면 모든 층에서 그 삼각형의 구면 면적과 2e-5 이내로 같고(부호 일정), 전구 합이 4π다(테스트). 즉 자료의 "2층 홀로노미 보정"은 **이미 정확히 만족**되며 보정할 것이 없다. 우리 연결은 Levi-Civita 연결이고, 세포 층(cellular sheaf) 용어로는 SO(2) 줄기(stalk)와 기하에서 정해진 제한 사상을 가진 층이다. 제한 사상을 학습시키는 sheaf NN은 이 정확성을 깨므로 쓰지 않고, sheaf 라플라시안은 진단으로만 쓴다.
- **DEC 연산자는 d d = 0이 정확하다.** 원-쌍대(외심) 호지 스타 ⋆₀(꼭짓점 면적)·⋆₁(쌍대길이/원길이)·⋆₂(1/면적)와 호지 라플라시안 L₀·L₁·L₂, Helmholtz 분해(χ: L₀χ = δu, ψ: L₂ψ = d₁u)를 구현했다. 기존 1-ring 최소제곱 스텐실(`physics_ops.py`)은 curl∘grad = 0, div∘curl = 0이 근사로만 성립하지만 DEC에서는 정확하다.

### 8.2 다섯 층에 넣는 방법 (stage6 계획의 수정)

| 층 | 자료의 제안 | 우리 구현 |
|---|---|---|
| 1 Discrete | 복합체 리프팅, 코체인 할당 | 메시가 이미 복합체. **flux 형 핵을 DEC로 쓴다**: 질량·수분은 쌍대 셀의 2-형식, flux는 간선 1-형식, 발산 = δ(= ⋆⁻¹dᵀ⋆) → 보존이 d d = 0으로 정확(2절의 F_ij = −F_ji가 곧 d₁ᵀ) |
| 2 Lie group | 간선에 군 변환 + 홀로노미 보정 | 이미 있음. 홀로노미 = 면적 테스트를 CI에 추가(8.1). 새 진단: 전역 SO(3) 회전+재표본화 오차 |
| 3 Lie algebra | 속도장에 L₁ 필터로 비압축 조건 강제 | 생성원 head가 **ψ_θ(꼭짓점 0-형식)** 를 내고 u_ψ = ⋆d₀ψ_θ (curl 성분) → ∇·u_ψ = 0 정확. V-Net의 바람 증분에는 Helmholtz 사영을 걸어 회전/발산 성분 비율을 진단·제어(종관규모 바람은 회전 성분이 대부분) |
| 4 Noise | 호지 고유기저 위의 "위상 보존 잡음" | 수송형 잡음의 기저 φ_r = **L₀의 저주파 고유벡터**(메시 위 구면조화함수) → ψ_r = σ_r φ_r, u_{ψ_r} = ⋆d₀ψ_r 는 비발산·질량 보존. 조화 1-형식이 없으므로 "구멍 모드"는 없다(b₁ = 0). 스칼라 잡음이 필요하면 L₂(쌍대 셀) 고유기저에서 상수 모드를 제외해 질량 보존 |
| 5 Solver | 기하 Stratonovich SDE + exp retraction | Stratonovich 수송(5절 계획대로). SO(2) 게이지는 정확해서 retraction이 필요 없고, 상태 제약(q ≥ 0, 질량)의 retraction이 곧 S5c의 사영 P_C |

### 8.3 위상을 "평가"에 쓰는 법 — persistent homology (TOGL의 아이디어)

전구 구면에는 조화 성분이 없지만, **장(場)의 부분수준집합 위상**은 있다: z500·msl의 저기압(국소 최솟값, H₀ 성분의 탄생·소멸)과 고기압의 개수·지속도(persistence)는 cyclone 구조가 예보에서 보존되는지를 재는 지표가 된다. 격자 64×32의 cubical persistent homology(GUDHI, PyPI 3.13.0)는 초기시각당 수 ms라 평가에 바로 넣을 수 있다. 제안 지표: lead별 "저기압 개수 오차"와 "persistence 분포의 Wasserstein 거리"(예보 vs 관측). 손실로 쓰는 것(미분 가능 PH)은 비용이 커서 진단부터 한다. PIGT/stage5 결과에 이 지표를 추가하면 RMSE가 비슷한 모델들의 구조 보존 차이를 볼 수 있다.

### 8.4 도구

TopoNetX(0.4.0)·TopoModelX(0.0.1)·GUDHI는 PyPI에서 설치 가능하다. 우리 메시는 면 정보를 이미 가지고 있어 리프팅 함수는 필요 없고, DEC 연산자는 `dec.py`의 희소행렬 수십 줄로 충분하다. TopoNetX는 호지 라플라시안 교차 검증용, GUDHI는 8.3의 지표용으로 쓴다. MPSN/CWN의 고차 메시지 패싱(면→간선→꼭짓점)은 우리 등변 층과 결합하려면 면·간선 특징의 게이지 변환 규칙을 새로 정의해야 하므로 stage6 이후 과제로 둔다.

### 8.5 stage 반영

- stage6a(1층+5층)에 DEC flux 핵을 쓰고, 테스트로 d d = 0·베티 수·홀로노미를 CI에 둔다(이미 추가).
- S6c의 생성원 head는 ψ_θ → ⋆d₀ψ(8.2), S6e의 잡음 기저는 L₀ 고유벡터(8.2).
- 평가 지표에 persistent-homology 저기압 통계(8.3)를 stage4·stage5 결과에도 소급 적용(별도 작업, 학습 불필요).
