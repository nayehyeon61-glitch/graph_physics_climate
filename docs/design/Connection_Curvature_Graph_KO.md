# 접속(connection)·곡률(curvature)로 확장한 그래프 — Lie 구조들의 어긋남을 그래프에 넣기 (stage11)

작성 2026-11-03 18:40 KST. 감자 제안: "미분구조 Lie → … 에서 호몰로지 연결을 말했는데, 거기에 추가로 Lie들의 어긋남, 즉 connection과 curvature에 대한 graph의 확장적 연결도 필요하다."

5층 설계서 8절(호몰로지/DEC 보강)에서 "면→간선→꼭짓점 고차 메시지 전달은 면·간선 특징의 게이지 변환 규칙을 새로 정의해야 하므로 뒤로 미룬다"고 했던 과제다. 이 문서는 (1) 우리 메시 위의 **두 Lie 구조**와 그 "어긋남"이 수학적으로 무엇인지, (2) 이미 검증한 사실, (3) 그래프를 어떻게 확장하는지(stage11a 고정 연산자, stage11b 심플리셜 메시지 전달), (4) 구현·테스트·비용·실험 계획을 적는다.

## 1. 메시 위의 두 Lie 구조와 "어긋남"

| | 기하 접속 (SO(2) 게이지) | 흐름 접속 (바람장) |
|---|---|---|
| Lie 군 / 대수 | 각 꼭짓점의 접평면 틀을 돌리는 SO(2); 대수 so(2) ≅ ℝ(각) | 구면 미분동형사상군의 생성원 = 접벡터장 u; 대수의 괄호 = Lie bracket [u, w] |
| 접속(connection) | 간선 평행이동 각 α_ij (Levi-Civita; `SphereGraphs.mesh_edges.alpha`) | 간선 1-형식 u♭_ij = ∫_edge u·dl (`MeshComplex.flat`) |
| 곡률(curvature) | 면 둘레 홀로노미 Σα = 면적 × K (가우스–보네, K = 1/R²) — **상수** | 면 둘레 순환 d₁u♭ = ζ·A (상대 와도 2-형식) — **상태에 따라 변함** |
| 어긋남(non-commutativity) | 두 경로의 평행이동이 다름 = 홀로노미 = 곡률 (Ambrose–Singer) | (a) 순환 ζ·A; (b) 물질 틀의 회전율 ζ/2 vs 기하 수송 → Okubo–Weiss W = \|S\|² − ζ²; (c) 두 바람층 생성원의 비가환성 [u_k, u_l] |
| 우리 코드에서 | 정확(홀로노미 = 면적, 2e-5; `tests/test_dec.py`) → 등변성이 정확한 이유 | 지금까지는 1-ring 최소제곱 스텐실의 curl/div 근사(`physics_ops.py`): curl∘grad ≠ 0 |

핵심 판단: **기하 접속의 곡률은 구면에서 상수라 특징으로서 정보가 없다**(이미 정확히 만족). 정보가 있는 곡률은 **흐름 접속의 곡률**(순환/와도)과, 두 접속의 어긋남(W, bracket), 그리고 그 곡률의 비국소 적분(Helmholtz ψ, χ)이다. "그래프의 확장적 연결"은 따라서 (i) 간선에 접속(α와 u♭), 면에 곡률(K·A, ζ·A, f·A)을 올리는 **심플리셜 복합체 그래프**, (ii) 꼭짓점의 복소 벡터 표현(Lie 군 층)과 간선 1-형식(호몰로지 층)을 잇는 **flat/sharp 사상**이다.

## 2. 구현한 것 (`gpclimate/geometry/connection.py`, `gpclimate/nn/connection_ops.py`)

### 2.1 `MeshComplex` — 게이지를 아는 심플리셜 복합체 (메시 모든 층, `SphereGraphs.complexes`, `GraphView.extras["complexes"]`)

- 방향 있는 간선(i<j)·면, 코바운더리 d₀(0→1형식), d₁(1→2형식), 전치, 여미분 δ₁ = ⋆₀⁻¹d₀ᵀ⋆₁ (양의 라플라시안 L₀ = δ₁d₀), 원·쌍대 호지 스타(`dec.py`). 길이는 평균 간선길이 h 단위(`physics_ops.Stencil`과 동일) → 모든 출력이 정규화 단위에서 O(1).
- **간선 방향을 양 끝 꼭짓점의 게이지로** 저장(dir_i: i의 틀에서 본 j의 방향, dir_j: j의 틀에서 본 i의 방향). 그래서
  - `flat(v)`: 꼭짓점 복소 벡터장 → 간선 1-형식 u♭ = ½ℓ[Re(d̄ᵢvᵢ) − Re(d̄ⱼvⱼ)] — **게이지 불변**(각 항이 자기 틀의 내적),
  - `sharp(w)`: 간선 1-형식 → 꼭짓점 벡터(2×2 최소제곱) — **등변**(결과가 꼭짓점 게이지로 나옴),
  - `gauge_transformed(θ)`: 틀 회전 시 dir을 e^{−iθ}로 돌려 등변성 테스트에 쓴다.
- `curvature(v)` = d₁ flat(v) (면 순환, 흐름 접속의 곡률 2-형식), `vorticity(v)`(인접 면 순환 합 / 면적 합 = Stokes), `divergence(v)` = −δ₁ flat(v), `potentials(v)`: Δψ = ζ, Δχ = div 를 캐시된 의사역행렬로 풀어 **비국소** 유선함수·속도퍼텐셜. `holonomy`: 기하 접속의 홀로노미(= ±면적).

### 2.2 `ConnectionCurvatureOperators` — 고정 연산자(매개변수 0, 정확히 등변)

바람쌍 k마다 스칼라 7개: `zeta_dec`(정확 와도), `div_dec`, `zeta_abs`(= ζ/2Ω + sin φ: 흐름 곡률 + 행성 곡률), `strain` \|S\|, `okubo_weiss` W, `psi`, `chi`; 벡터: `u_psi` = i∇ψ(회전풍), `u_chi` = ∇χ(발산풍), 바람층 쌍마다 `bracket_kl` = (u_k·∇)u_l − (u_l·∇)u_k. base11(3 바람쌍): 21 스칼라 + 9 벡터; fcn21(4쌍): 28 + 14. guide 메시 0 입력에 `EqNorm → EqLinear`로 더한다(`GuideConfig.curvature_ops`, `--curvature-ops 1`).

### 2.3 검증한 사실 (`tests/test_connection_curvature.py`, 6개 통과; 전체 38개)

| 사실 | 결과 |
|---|---|
| d₁d₀ = 0, ⟨d₀s, w⟩ = ⟨s, d₀ᵀw⟩, ⟨d₁w, c⟩ = ⟨w, d₁ᵀc⟩ | 모든 층에서 정확(1e-5) |
| 기하 접속 홀로노미 = 면적, 합 4π | 모든 층 |
| 강체회전 u = ẑ×r: 곡률(와도) = 2 sin φ, 발산 0, ψ = −sin φ, flat→sharp 복원, i∇ψ = u | 상대 RMS 5 % 이내(메시 2562 노드) |
| 기울기장 u = ∇sin φ: 곡률 ≈ 0, 발산 = −2 sin φ; d₁d₀f = 0 정확 | 통과(발산 부호 검증으로 δ₁ 부호 오류 1건 발견·수정) |
| Lie bracket [ξ_z, ξ_x] = −ξ_y (회전 생성원의 교환자) | 상대오차 10 % 이내 |
| Killing 장: W = −ζ² ≤ 0(회전 지배), 변형률 ≈ 0 | 99 % 이상 꼭짓점 |
| 게이지 등변성(불변량 불변, 벡터 회전; 복합체도 함께 변환) | 2e-5 |
| curvature_ops 켠 guide 전체의 등변성 P0…P0u | 5e-5 |

### 2.4 비용 (1 thread, nice, batch 4, stage3 크기)

Stage A forward+backward 1.88 s → 2.00 s(+6 %), 연산자 자체 0.07 s, guide 매개변수 +0.3 %(275,817 → 276,716), 메시 0 Poisson 의사역행렬 8 s(프로세스당 1회, 26 MB). V-Net·평가 비용은 변화 없음(guide 동결).

## 3. 그래프 확장의 두 단계

| stage | 그래프에 무엇을 더하나 | 학습되는 것 | 비교 |
|---|---|---|---|
| **stage11a** (체인 등록, stage10a 뒤 자동 ≈ 11/4 09:00 KST) | 흐름 접속의 곡률·어긋남을 **고정 연산자**로 꼭짓점에 내려 guide 입력에 추가 | 기존 guide(임베딩만 21+9 채널 늘어남) | stage3(+ClimODE); P 의존도 |
| **stage11b** (후속) | **심플리셜 메시지 전달**: 간선 특징 h_e(1-형식: u♭, d₀s, α, ℓ), 면 특징 h_f(2-형식: K·A, ζ·A = d₁h_e, f·A)와 꼭짓점 특징을 d₀/d₁/δ₁/sharp로 주고받는 블록(MPSN/CWN 구조, 제한 사상은 고정 = 기하에서 옴) | 간선·면 MLP, 꼭짓점 결합 | stage11a |
| stage11c (후보) | 기하 접속의 **학습 비틀림** α′ = α + a_θ(불변량) — 흐름에 맞춘 수송(물질 틀) | a_θ | stage11b; 곡률 정칙화 \|Hol(α′) − Hol(α)\| |

stage11b가 "확장적 연결"의 본체다. 등변성은 간선·면 특징이 모두 불변량(1-형식, 2-형식)이고 꼭짓점으로 돌아올 때만 sharp(등변)를 거치므로 그대로 유지되며, d₁d₀ = 0 덕분에 **기울기 성분(발산풍)은 면 특징에 전혀 기여하지 않고 회전 성분만 곡률로 보인다** — 종관 규모 바람의 회전 성분 우세를 구조로 반영한다. 5층 설계서의 3층(생성원 ψ_θ → u = ⋆d₀ψ)과 4층(L₀ 고유기저 잡음)이 같은 복합체 위에서 바로 이어진다.

## 4. stage11a 실험 계획

- run `runs/era5_64x32_s11a_curv`, 체인 `chain_s11.sh`: stage3 config.json을 복사해 `curvature_ops=1`만 바꾸고(실행 중인 `run_guide_ablation.sh`는 수정하지 않음), stage3 정규화기·guide 없음 V-Net 재사용, Stage A 1000 → Stage B 1500(2-step rollout) → 평가 → 진단. ≈ 3시간.
- 채택 기준: 11변수 평균 RMSE가 5개 리드 중 3개 이상에서 stage3보다 낮고, P=0 열화가 stage3(+102 %) 이상으로 **유지되거나 커질 것**(곡률 정보가 guide를 통해 쓰이는지의 증거). 진단에 `curvature_terms`의 선형 프로브(24 h 경향에 대한 R²)를 추가해 어느 항(ζ_dec, W, ψ, bracket)이 설명력을 갖는지 본다(후속 작업).
- stage10a(fcn21)와 독립적으로 stage3 기준 비교. 둘 다 유효하면 stage9 통합에서 fcn21 + curvature_ops 조합.

## 5. 위험

- ψ/χ는 전구 Poisson 풀이라 극 근처·거친 메시에서 오차가 크고, 정규화 단위 O(1)이지만 분산이 커서 EqNorm이 필요(적용됨).
- 곡률 연산자 21+9 채널이 기존 47+19 연산자와 중복(ζ, div)된다 — 중복은 정확/근사 두 버전이 함께 들어가는 것으로, 선형 프로브로 기여를 분리한다.
- 의사역행렬은 메시 0(2562) 기준 26 MB; 더 고운 메시(5·6 split)에서는 희소 풀이(CG, 사전 Cholesky)로 바꿔야 한다.

## 6. 구현 지도

| 파일 | 내용 |
|---|---|
| `gpclimate/geometry/connection.py` | `MeshComplex` (from_level, d0/d1/δ₁, flat/sharp, curvature/vorticity/divergence, poisson/potentials, holonomy, gauge_transformed, to) |
| `gpclimate/geometry/graphs.py` | `SphereGraphs.complexes`, `GraphView.extras["complexes"]`, `gauge_transformed` 전파 |
| `gpclimate/nn/connection_ops.py` | `ConnectionCurvatureOperators` (out_dims, scalar/vector_names) |
| `gpclimate/models/lie_physical_gnn.py` | `GuideConfig.curvature_ops`, `curvature_terms`, 메시 0 입력 합산 |
| `scripts/train_eval.py` | `--curvature-ops`(config.json), v_scale 전달 |
| `tests/test_connection_curvature.py` | 2.3절의 사실 |
| `runs/era5_64x32_s11a_curv/chain_s11.sh` | stage11a 자동 체인 |
