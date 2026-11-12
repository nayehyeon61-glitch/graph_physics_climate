# MPAS(Model for Prediction Across Scales)의 Voronoi–Delaunay 구조와 우리 core의 대응

작성 2026-11-09 16:20 KST. 요청: 감자 2026-11-09 16:00 KST "https://github.com/MPAS-Dev/MPAS-Model 이것처럼 기후에 대한 보르노이와 들로네도 참고해 보세요".

읽은 것: MPAS-Model 저장소(v8.4.2; `src/operators` = 메시 연산자, `src/core_*` = atmosphere/ocean/land-ice/seaice), MPAS 메시 사양서(mpas_meshspec.pdf), Skamarock 외 2012(MWR, "A Multiscale Nonhydrostatic Atmospheric Model Using Centroidal Voronoi Tesselations and C-Grid Staggering"), Ringler·Thuburn·Klemp·Skamarock 2010(JCP, "A unified approach to energy conservation and potential vorticity dynamics for arbitrarily-structured C-grids"; 이른바 TRiSK).

## 1. MPAS가 dual 공간을 쓰는 방식

- **메시**: 구면 중심 Voronoi 분할(SCVT). 생성점 x_i = 밀도함수 ρ에 대한 cell의 질량중심(Lloyd 반복). 가변 해상도는 ρ를 매끄럽게 바꿔서(coarse/fine 8배까지, 162→21 km) 만들고, 전이대에서 특별한 처리 없이 돈다. 준균일 메시 예: 2,562 cell ≈ 480 km, 10,242 ≈ 240 km, 40,962 ≈ 120 km.
- **세 종류의 위치(C-grid)**: cell(Voronoi, `areaCell`) — 질량·θ·수분 등 스칼라; edge(x_e, 두 cell 중심을 잇는 Delaunay 간선과 두 vertex를 잇는 Voronoi 변의 교점, 직교 가정) — **법선 속도 u_e(예측변수)**, `dcEdge` d_e(cell 중심 거리), `dvEdge` l_e(Voronoi 변 길이), `angleEdge`; vertex(Delaunay 삼각형, `areaTriangle`) — 와도·PV, `kiteAreasOnVertex`(cell∩삼각형 연 면적).
- **연산자(Ringler 2010)**: 발산 (div F)_i = (1/A_i) Σ_e n_{e,i} F_e l_e; 기울기 (∇h)_e = (h_j − h_i)/d_e; 회전 (curl F)_v = (1/A_v) Σ_e t_{e,v} F_e d_e. 모두 미분 형식(DEC)의 이산화이고 primal/dual 변의 직교가 전제.
- **접선 재구성(Thuburn 2009)**: F⊥_e = (1/d_e) Σ_{e'} w_{ee'} l_{e'} F_{e'}, 반대칭 가중치 `weightsOnEdge`. 이것으로 비선형 Coriolis 항 q h u⊥가 일을 하지 않게(에너지 보존) 하고, 가짜 비정상 지균 모드를 없앤다.
- **운동량은 벡터 불변형**: ∂u_e/∂t = −(q h u⊥)_e − ∇_e(K + gh), K_i = (1/A_i) Σ_e (A_e/4) u_e². 천수 모형에서 질량·PV·potential enstrophy 보존, 에너지는 시간 절단 오차까지 보존. 3D 비정수압 모형은 질량·스칼라 질량·Θm만 정확 보존.
- **스칼라 수송**: 간선 flux 형태(최소제곱 다항식 간선값, 3/4차, 단조 옵션), 질량 flux를 음향 sub-step 평균으로 써서 연속식과 일치. 시간적분은 split-explicit RK3(60 km에서 Δt 450 s).

## 2. 우리 core와의 대응표

| MPAS | 우리 (stage18~21 core / guide) | 상태 |
|---|---|---|
| Voronoi cell, `areaCell`, 생성점 = 질량중심(SCVT) | `GridFaceGraph` cell(위경도 cell, A_i); 18g power diagram은 면만 옮기고 생성점은 고정 → 변형 후 생성점 ≠ 질량중심 | 부분 (중심 일치 없음) |
| edge의 법선 속도 u_e (예측변수) | 없음 — 바람은 cell 중심 복소 벡터(A-grid), guide 흐름도 cell 중심, 학습 flux J_θ는 면당 스칼라 | ✗ |
| `dcEdge` d_e, `dvEdge` l_e, `angleEdge` | 면의 d, ℓ, 법선; 가중치 ℓ/d (18d Laplace, 18f κ) | ✓ |
| vertex(Delaunay 삼각형)의 와도·PV, `kiteAreasOnVertex` | core에는 없음. guide의 메시 0(정20면체, 2,562 노드 ≈ MPAS 480 km 메시)에는 stage11a DEC 연산자(u♭ 간선 1-형식, 삼각형 순환 = 와도, 노드 발산, ψ/χ)가 있어 **guide는 이미 C-grid식 배치**를 쓴다 | guide ✓ / core ✗ |
| cell 발산 ← 간선 flux (정확 보존) | −div_G J_θ, Laplace block | ✓ |
| 간선 기울기 (h_j − h_i)/d_e | Laplace block·κ의 Δq/d | ✓ |
| Thuburn 접선 재구성, 에너지 보존 Coriolis, 벡터 불변형 운동량 | 없음(운동량은 학습). PINN 잔차로는 부적합(ERA5 6 h mom500 var(L−R)/var(L) = 1.97) → 특징으로는 적합 | ✗ |
| 간선 flux 수송(3/4차, 단조) | SL 수송 — 위경도 극 근처 cell 폭 31 km, 6 h·30 m/s = 648 km → CFL 21(적도 1.0)이라 flux 형태 불가. 준균일 2,562-cell 메시에서는 CFL 1.3, 1 h sub-cycling이면 0.22 | ✗ (메시 바꾸면 가능) |
| 밀도함수 ρ로 가변 해상도(사용자 지정) | 18g의 a_i = **학습된** 밀도함수(정적, 이동 ≤ 0.25 d) | 유사 (재중심화 없음) |
| split-explicit, 음향 sub-step, 비정수압 | 6 h 자료 간격의 학습 core에는 해당 없음 | 해당 없음 |

## 3. 반영 후보 (한 stage에 하나)

- **A. stage22 TRiSK 특징 → guide 입력(dual feature matrix)**: 상태에서 C-grid 양을 고정 연산자로 만들어 guide 입력에 더한다 — 면 법선 속도 u_e = ū·n_e, vertex 와도 ζ_v(curl, dual 면적), q_v = (ζ_v + f)/h(층후 z500−z850로 h), Thuburn 가중치로 u⊥_e, 비선형 Coriolis 항 (q h u⊥)_e, K_i = Σ A_e u_e²/4A_i, ∇_e(K + Φ), 두께 flux 발산. edge→cell로 모아 노드 특징(19b `gemb_in`처럼 0 초기화 adapter + head 학습, trunk 고정). 매개변수 ≈ 0. 2026-11-09 15:20 KST의 "들로네→보르노이 전처리 feature matrix" 제안과 같은 것이고 MPAS가 정확한 정의를 준다. 기대: stage11a 기준 24–48 h +0.5~1.5 %, guide 이득·P=0 손실 증가. 비용: Stage A 10분 + Stage B 9.5 h + 평가 1.2 h.
- **B. stage23 준균일 Voronoi 메시 core(MPAS식 간선 flux 수송)**: core의 tendency를 메시 0(2,562 cell, ≈ MPAS 480 km)에서 계산 — guide 흐름의 면 법선 성분 u_e, 1 h sub-cycling 간선 flux 수송(보존·단조), vertex 와도, 격자↔메시는 기존 `transport_interpolate`. 극 CFL 문제가 없어져 SL 대신 정확 보존 수송이 가능. 위험: 매 6 h 격자↔메시 보간의 평활화(선명도 ↓), 중위도 격자 cell 625 km vs 메시 480 km. 기대 RMSE ±1 %, 물리 충실도 최대. 구현 ≈ 2일 + 11 h.
- **C. 18g의 중심 일치(SCVT 성질)**: 변형된 cell의 질량중심과 생성점의 거리 |c_i − x_i|/d를 진단으로 남기고, 필요하면 Lloyd식 재중심화 또는 벌점. 새 stage 없이 18g 보고에 넣을 수 있다.
- 채택하지 않을 것: 음향 split-explicit, RBF cell 중심 속도 재구성(우리는 이미 cell 중심 바람), 벡터 불변형 운동량의 PINN 잔차(6 h에서 성립 안 함).

추천: A를 stage22로 stage19 뒤(stage20 앞)에 넣고, C는 18g 보고에 진단으로 넣는다. B는 GPU가 있을 때.

출처: [MPAS-Model](https://github.com/MPAS-Dev/MPAS-Model), [MPAS 메시 사양서](https://www2.mmm.ucar.edu/projects/mpas/mpas_website_linked_files/mpas_meshspec.pdf), [Skamarock et al. 2012](https://twister.caps.ou.edu/CFD2023/SkamarockEtal2012.pdf), [Ringler et al. 2010](https://core.ac.uk/download/12827548.pdf).
