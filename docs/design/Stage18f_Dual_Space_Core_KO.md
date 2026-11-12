# stage18f·stage18g: dual 공간 분해 core 설계 (θ_p 간선 물리 / θ_g 기하)

작성 2026-11-09 14:45 KST. 요청:

- 감자 2026-11-09 14:09 KST, Delaunay–Voronoi 논의와 arXiv 2406.05225 첨부: "이것도 core 반영해 보죠, 우리는 연속된 공간에 대한 물리를 통해 tendency core인데 위와 같은 dual 공간을 통한 분해가 필요할 것 같기도 합니다. 특히 위와 같은 graph manifold를 인식해야 한다는 거죠."
- 결정 카드 2026-11-09 14:14 KST: "18f 뒤 항상 실행" (θ_p와 θ_g를 각각 판정).

핵심은 두 종류의 parameter를 분리하는 것입니다.

- θ_g (기하): 공간을 어떻게 나눌 것인가 — cell 면적, 면 길이, 중심 거리.
- θ_p (물리): 나뉜 공간 사이에 무엇이 전달되는가 — 간선의 전달계수.

## 1. 지금 core가 이미 가진 dual 구조

- cell = 격자점의 Voronoi cell(경계 = 이웃 중점 = 수직이등분선, 면적 A_i), 면 = Delaunay 간선(4-이웃), |F_ij|/d_ij = ℓ/d(cotangent 가중치). `gpclimate/nn/graph_core.py::GridFaceGraph`.
- 간선 위 물리: SL 수송(guide flow), 반대칭 학습 flux J_θ(방향성, 보존), Laplace block의 확산 K(cell별, 면 평균 → κ_ij = κ_ji ≥ 0)[stage18d], 경계 보존 인자 ε/√(Δq²+ε²)[stage18e].
- 따라서 감자의 상자식 du_i/dt = (1/|V_i|) Σ_j (|F_ij|/d_ij) κ_θ(u_i, u_j, p_ij) (u_j − u_i)에서 기하 부분은 이미 있고, κ가 물리 특징 p_ij에 명시적으로 의존하지 않는 것(18d는 κ = (K_i + K_j)/2)이 빠진 부분입니다. 이류·회전은 설계대로 방향성 flux J_θ와 SL 수송이 맡습니다.

## 2. 첨부 논문과의 연결 (Wang, Cerviño, Ribeiro, "A Manifold Perspective on the Statistical Generalization of GNNs", ICML 2025)

- 그래프를 manifold에서 뽑은 표본으로 보고, GNN을 가중 Laplace 연산자 L_ρ f = −(1/2ρ) div(ρ² ∇f)(식 3)의 manifold 신경망의 이산화로 봅니다. 일반화 격차는 노드 수 N에 대해 log 축에서 선형으로 줄고, 필터의 스펙트럼 연속성 상수 C_L에 비례해 커집니다(정리 1, 2). 변별력과 일반화의 trade-off이고, 스펙트럼 정칙화로 C_L을 조절합니다.
- 우리 격자(2048 cell)는 S²의 표본이고, Laplace block의 면 전도도 κ_ij는 L_ρ의 ρ²에, cell 면적 A_i는 표본 측도 μ에 해당합니다. 즉 θ_p(κ)는 manifold 위의 전도도를 배우는 것이고 θ_g(power diagram 가중치)는 측도 μ를 바꾸는 것입니다. 두 parameter를 분리해 두면 core가 "graph manifold"를 두 층(측도·전도도)으로 인식하게 됩니다.
- 논문의 교훈 중 쓰는 것: κ_θ는 상태·특징의 매끄러운 함수(작은 MLP, log 배수를 ±4로 제한)로 두고, 기하 변형은 유계(면 이동 ≤ 0.25 d)로 둬서 연산자의 스펙트럼이 급변하지 않게 합니다(C_L을 키우지 않음).

## 3. stage18f θ_p: 간선 전달계수 (`VNetConfig.core_edge`, `--core-edge 1`, `graph_core.py::EdgeConductance`)

- κ_f = ½(K_i + K_j) · exp(φ_θ(f)), φ ∈ [−4, 4], 마지막 층 0 초기화 → 18d/18e와 같은 출발(테스트: 예보 동일).
- 입력: 두 cell embedding의 대칭 결합(스칼라 합·|차|, 면 법선 방향 벡터 투영의 |합|·|차|) + 물리 간선 특징 p_f = [|Δorog|, |Δlsm|, guide flow의 면 가로/따라 성분 |·|(3 level group, 10 m/s 단위), |Δq|(5 스칼라, 정규화 단위), cos φ_f, 면 종류(동=1/북=0)] — 15차원.
- 모두 (i ↔ j, n → −n)에 불변이라 κ_ij = κ_ji ≥ 0(테스트: 모든 면을 뒤집어도 같은 값). 보존·단조·안정 상한은 Laplace block 그대로입니다.
- 로그: `edge x`(배수 평균), `sd`(log 배수의 표준편차). `last_core["edge_mult"]`에 변수별 평균.

## 4. stage18g θ_g: power diagram 기하 (`VNetConfig.core_geometry`, `--core-geometry 1`, `GridFaceGraph.deformed`)

- cell 가중치 a_i = MLP(orog, lsm, sin φ, cos φ)(정적, 16 hidden, 마지막 층 0 초기화 → 원래 격자). power 거리 |x − x_i|² − w_i의 이등분선은 중점에서 (w_i − w_j)/2d_ij만큼 가벼운 쪽으로 이동하므로, 유계형으로 δ_f/d_f = 0.25·tanh(a_i − a_j)(법선 방향, 무거운 cell이 커짐).
- 갱신(`deformed`): 동서 늘임 s_x = (δ_E − δ_W)/Δx, 남북 늘임 s_y = (δ_N − δ_S)/Δy, Ã_i = A_i(1 + s_x)(1 + s_y)(총면적 보정), 면 길이 = 원래 길이 × (양쪽 cell 늘임의 평균), 가중치 ℓ̃/d(중심은 고정), face mass = min(Ã_i, Ã_j). Delaunay 연결은 유지(이동 ≤ 0.25 d라 생길 수 있는 작은 대각 면은 무시; 면적비 ≥ 0.25).
- 이 기하를 core의 모든 연산자(flux 발산 div_G J_θ, Laplace block)가 씁니다 → 이산 보존은 새 면적 Ã에 대해 정확합니다(테스트: Σ Ã_i Δu_i = 0, 단조). 손실·평가의 면적 가중은 실제 면적 그대로입니다(감자: "기하를 움직이면 Cell 면적과 이산 보존 법칙도 함께 갱신").
- 로그: `geom |d|` 평균/최대 이동(d의 비율), `A` 면적비 범위.

## 5. 체인·판정

- 순서: 18e → **18f** (`runs/era5_64x32_s18f_edge/chain_s18f.sh`, CHAIN_S18F_DONE) → **18g** (`runs/era5_64x32_s18g_geom/chain_s18g.sh`, CHAIN_S18G_DONE) → 19 (`chain_s19.sh`는 18g의 base.txt/bflags.txt를 읽음).
- 부모(config의 출발): 직전 stage가 채택이면 그것, 아니면 그 부모. 18f의 부모 = 18e 채택 시 18e, 아니면 18d(둘 다 Laplace block 보유). 18g의 부모 = 18f 채택 시 18f, 아니면 18f의 부모. config = 부모 + 새 flag 하나(변경 하나).
- 판정: 부모 대비 `decide_adoption`(3/5 lead, 평균 ≥ 0.5 %). 부모가 체인 base(18/18b/18c 중 채택본)와 다르면 base 대비도 통과해야 채택 → 채택본이 stage19의 base. 짝(guide 없는 모델)과 guide는 base 것.
- 보고(`reports/compare_stage18f.md`, `compare_stage18g.md`): 개선율(+ = 개선), guide 이득, β·T F S D K, edge 배수 분포(변수별 평균·sd), 기하 이동·면적비 지도, 경사 선명도, ClimODE 열, 2020/2022.
- 비용: 18f는 면당 작은 MLL(≈ +3 %), 18g는 정적 변형(무시) → 각 ≈ 11 h(guide 모델 Stage B 1000 step ≈ 9.5 h + 평가 1.2 h). ETA 18f ≈ 11-12 오전, 18g ≈ 11-12 저녁, stage19c ≈ 11-14 새벽, stage20 ≈ 11-15 밤 KST.

## 6. 위험

- 18f: κ가 p_f로 전선 근처 확산을 키우면 RMSE는 좋아지고 선명도는 나빠질 수 있음 → 선명도 비율을 함께 봅니다. 반대로 18e의 ε 인자와 역할이 겹치면 배수가 1 근처에 머무를 수 있습니다(sd로 확인).
- 18g: 기하가 바뀌면 core의 보존 측도가 실제 면적과 달라집니다(설계상 허용, 논문의 측도 μ 해석). 극 근처에서 면적비 하한 0.25. 정적 가중치라 바람·기압 같은 동적 요인은 θ_p 쪽에만 둡니다.
- 0 초기화 head가 1000 step 미세조정 안에 충분히 자라는지 β처럼 로그로 확인합니다.
