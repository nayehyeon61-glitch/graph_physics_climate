# stage22: TRiSK(MPAS C-grid) 양을 guide 입력으로 — dual feature matrix

작성 2026-11-09 16:30 KST. 요청: 감자 2026-11-09 15:17 KST "feature matrix→물리적 보르노이 … 물리적 전처리를 통해서 들로네→보르노이로 분해해서 feature matrix를 잡는 것은 어느 정도 효과가 있을까요?", 16:00 KST "MPAS-Model 이것처럼 기후에 대한 보르노이와 들로네도 참고해 보세요" → 결정 카드 16:07 KST "A: stage22 TRiSK→guide". 참고 대응표: `MPAS_Dual_Mesh_Reference_KO.md`.

## 1. 무엇을 넣나 (`gpclimate/nn/trisk_ops.py::TriskOperators`, `GuideConfig.trisk_ops`, `--trisk-ops 1`)

MPAS가 예측하는 양을 우리 guide의 가장 촘촘한 메시(정20면체 노드 = Voronoi cell 2,562개 ≈ MPAS 480 km 메시, 삼각형 = dual vertex, `MeshComplex`)에서 상태로부터 고정 연산자로 계산해 입력에 더합니다. 매개변수 0, SO(2) 등변(테스트). 격자→메시는 쌍선형 수송 보간(`grid2mesh_lin`).

- 간선(Delaunay 간선 = cell 중심 연결, MPAS의 u_e 위치): 법선 성분 u_n = ½(u_i + u_j)·n, 접선 성분 u_t(t = k × n, 반시계). MPAS는 u_n만 저장해 Thuburn(2009) 가중치로 u_t를 재구성하지만 우리 상태는 cell 중심 벡터 전체를 가지므로 직접 읽습니다(가중치의 에너지 보존 성질은 예측 core용이고 특징에는 불필요).
- vertex(삼각형): 와도 ζ_v = 순환/면적(DEC `d1`), 절대 와도 η_v = ζ_v + f_v, PV q_v = η_v / h_v, h = Φ₅₀₀/g(등가 깊이; 500 hPa 등가순압 1층 모형).
- 스칼라(7): `ke{k}` TRiSK 운동에너지 K_i = (1/A_i) Σ_e (l_e d_e/4) u_e²(바람쌍 3개), `pv500`(삼각형 면적 가중 평균), `div_hu500` = div(h u)(간선 질량 flux의 cell 발산), `adv_t850` = −div(u₈₅₀ t850), `adv_q700` = −div(ū₈₅₀,₅₀₀ q700)(간선 flux 형태의 이류 경향 — 15:17 KST 제안의 "들로네 flux → 보르노이").
- 벡터(5, 간선 법선값 → 최소제곱 `sharp`로 노드 벡터): `pvflux500` = +q̃_e ĥ_e u_t(−η k×u의 법선 성분, 비선형 Coriolis/PV flux 항), `gradB500` = ∇_e(K + Φ)(Bernoulli 기울기), `vi_tend500` = pvflux − gradB(벡터 불변형 운동량 경향; 정상상태면 0), `gradK{k}`(850·10 m 쌍, Φ가 상태에 없음).
- 척도: 2Ω, U_REF 10 m/s, H_REF 5500 m, 1e-6/1e-5 s⁻¹로 나눠 O(1); 물리 단위는 `set_scales(s_mean, s_std, v_scale)`(train_eval `forecaster`가 정규화기에서 설정, guide와 함께 저장).
- 들어가는 길: `trisk_norm`(EqNorm) → `trisk_embed`(EqLinear, **0 초기화**) → 메시 0 입력 m0에 더함. 0 초기화라 학습된 guide의 특징 P는 처음엔 그대로이고(테스트), `GUIDE_HEADS`에 포함되어 head-only Stage A에서 adapter만 학습됩니다(19b `gemb_in`과 같은 방식).

검증(`tests/test_trisk_ops.py`, 6): Williamson TC2 정상 지균류에서 pvflux와 gradB가 서로 상쇄 — 잔차/|gradB| 0.1 미만, 메시 642→2,562 노드에서 0.7배 이하로 수렴(부호·단위 검증); K = u²/2(5 % 이내), q = η/h(8 %); u = cos φ 북풍의 div(h u) = −2h sin φ/a(8 %), sin φ 이류 경향 (2 sin²φ − cos²φ)/a(15 %); 게이지 등변; guide adapter 0 초기화·등변; 파이프라인 A→B→eval.

## 2. 왜 효과를 기대하나, 어느 정도

- stage11a(DEC 곡률·발산·ψ/χ·bracket을 guide 입력에): +1.5/+0.9/+0.4/−0.2/−0.4 %(24→120 h), guide 이득 2배, P=0 손실 +90 %. stage22는 같은 종류(고정 간선→노드 물리 연산자)이고, 11a에 없는 **비선형 항**(PV flux q h u⊥, Bernoulli 기울기, 그 잔차 = 운동량 불균형, 질량·열·수분의 간선 flux 발산)을 더합니다. 선형 연산자(기울기·Laplacian)는 메시지 전달이 스스로 배울 수 있지만 곱 형태의 항은 그렇지 않습니다.
- 기대: 24–48 h +0.5~1.5 %, 96 h 이후 ≈ 0; guide 이득·P=0 손실 증가가 더 뚜렷할 것. 비용 ≈ 0(Stage A 10분, Stage B 9.5 h, 평가 1.2 h).
- 쓰지 않는 것: 벡터 불변형 운동량의 PINN 잔차(ERA5 6 h에서 mom500 var(L−R)/var(L) = 1.97 — 성립하지 않음), split-explicit 시간적분, RBF 재구성.

## 3. 체인·판정 (`runs/era5_64x32_s22_trisk/chain_s22.sh`)

- CHAIN_S19_DONE 뒤. 부모 = stage19a/b/c 중 stage18 base(stage21의 base.txt) 대비 채택 규칙을 넘는 것 중 평균 이득 최대(`scripts/pick_parent.py`), 없으면 base. config = 부모 + trisk_ops 1; 부모의 전처리 산출물(graph_embed.npy, pinn_scales.json, tv_eps.json)과 pair를 복사.
- Stage A: 부모 guide에서 head + adapter만, 17c 섭동, 부모의 `--pinn-weight 0.05`(19 계열이면), 1000 step. Stage B: stage9b V-Net에서 1000 step 짝 미세조정, 부모의 Stage B 플래그(bflags + 19c면 `--pinn-weight-b 0.02`; `extra_b.txt`로 stage20 compose에 전달).
- 판정: 부모 대비(+ base 대비, 다를 때) 채택 규칙; 물리 지표(`core_physics_summary.py`: 물리 항 비중·guide 단독 기술·guide 이득)와 `diagnose_guide`의 P=0 손실·probe를 보고. `reports/compare_stage22.md`.
- stage20 체인은 CHAIN_S22_DONE을 기다리도록 재지정(2026-11-09 16:22 KST); `stage20_compose.RUNS`에 stage22 포함.
- ETA(KST): stage19 종료 ≈ 11-15 새벽 → stage22 ≈ 11-15 저녁 → stage20 ≈ 11-17 오전.

## 4. 함께 넣은 것 — stage18g의 SCVT 진단 (결정 카드의 C)

MPAS의 SCVT는 생성점 = 질량중심입니다. 18g의 power diagram은 면만 옮기고 생성점은 고정이라 변형 뒤 생성점이 질량중심에서 벗어납니다(1차 오차). `GridFaceGraph.deformed`가 `deformation["centroid"]` = |질량중심 이동|/cell 크기(= 마주보는 두 면 이동의 평균)를 계산하고, 학습 로그 `geom |d| … A … |c| mean/max`, `last_core["geom_centroid"]`로 남깁니다(테스트: 격자 0, 한 cell만 무거우면 그 cell 0·이웃 > 0, 상한 0.25√2). 18g 보고에서 |c|가 크면 Lloyd식 재중심화 또는 벌점을 후속 후보로 둡니다.
