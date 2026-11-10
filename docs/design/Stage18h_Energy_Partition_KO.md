# stage18h: 에너지 기반 들로네–보로노이 분할 (2026-11-10)

## 1. 요청과 출발점

감자 2026-11-10 14:40 KST: "들로네, 보르노이가 단순히 격자가 아닌 에너지 base로 나눌 수 있을까요? 물리적 에너지(이는 우리가 dinosaur 물리분해를 통해서 물리적 core를 통해서 얻고)로 공간장을 나누면?"

- 지금 core(stage18)는 위경도 격자의 면 그래프 위에 있고, 쌍대는 자명합니다. 셀 중심을 잇는 간선이 들로네, 셀 경계면이 보로노이 변이며, flux 발산 div_i J = Σ J_f ℓ_f / A_i 가 보로노이 변 길이 ℓ와 셀 면적 A를 씁니다.
- stage18g는 이 분할을 지형·해륙·위도로 정한 **정적** power diagram으로 바꿉니다(들로네 연결 유지, 면 이동 ≤ 0.25 d).
- stage18h는 같은 power diagram의 가중치를 **매 6 h sub-step의 상태 에너지**로 다시 잡습니다. MPAS가 SCVT 밀도 함수로 셀 크기를 정하는 것, 수치해석의 r-적응 격자(균등분배 원리: 셀마다 "감시 함수"가 고르게 담기도록 나눔)와 같은 원리입니다.

**Dinosaur에 대해:** Dinosaur(및 NeuralGCM)는 JAX이고 연직 기둥 전체(σ 층, 지표 기압)를 상태로 요구하므로 11변수 데이터로는 직접 돌릴 수 없고, 현재 코드에서 쓰지 않습니다(`docs/IMPLEMENTATION_NOTES_KO.md` §2). 같은 역할의 물리 분해는 PyTorch에 있습니다. guide의 축소 원시방정식 `gpclimate/nn/primitive2d.py`(이류, 코리올리, 기압경도, 질량 보존; 연직 질량 가중 0.30/0.65/0.05)와 core의 guide 항(수송 변화 + source)입니다. 에너지 성분은 이것들로 계산합니다.

## 2. 에너지 성분 (셀별, 매 sub-step, 기울기 없음)

| 성분 | 정의 | 물리 |
|---|---|---|
| ke850, ke500, ke10 | \|u_l\|² (m²/s²) | 층별 운동에너지(×2) |
| ape850 | (t850 − 띠 평균 t850)² | 가용위치에너지 근사(Lorenz) |
| latent700 | q700 (kg/kg, 물리 단위) | 잠열 에너지 근사 L q |
| ke_rate | Σ_l w_l \|u_l · Δu_l\|, Δu = guide core 항(수송 + source)의 바람 변화 | core가 주는 운동에너지 변화율 |

각 성분은 표본마다 면적 평균으로 나눕니다: L_c = log(e_c / ⟨e_c⟩ + 0.1) − log(1.1) (균일장이면 0). 단위와 정규화 상수는 비율에서 상쇄됩니다.

## 3. 분할

- power 가중치 a_i = −Σ_c α_c L_c,i (+ parent가 18g를 채택했으면 정적 가중치), α_c = 4·raw_c, raw는 0 초기화 → 시작은 parent와 같음.
- α_c > 0이면 그 에너지가 큰 곳의 셀이 작아집니다. 학습된 α_c의 부호와 크기가 "어떤 에너지가 공간 분할을 정하는가"의 답입니다.
- `GridFaceGraph.deformed(a [B, N], 0.25, fixed_areas=True)`: 표본별 power diagram, 들로네 연결 유지, 면 이동 δ_f/d_f = 0.25·tanh(a_i − a_j).

## 4. 보존 (시간에 따라 바뀌는 분할)

분할이 sub-step마다 바뀌면 "새 면적에 대한 적분"은 단계 사이에 보존되지 않습니다. 그래서 셀 변화는 **고정 격자 셀 면적**으로 누적합니다(새 셀 내용의 변화를 생성점의 격자 셀로 되돌리는 ALE 방식). 연산자는 새 분할의 면 길이 ℓ̃과 덮개 가중치 ℓ̃/d만 쓰고, 안정성 상한의 face mass도 격자 것을 씁니다. 따라서 Σ_i A_i Δq_i = 0이 정확히 성립하고, 확산은 여전히 단조(볼록 결합)입니다. 새 면적은 진단용으로만 남깁니다.

## 5. 한계와 다음 단계

- 생성점(데이터 점)은 64×32에 그대로이므로 해상도는 늘지 않습니다. 바뀌는 것은 에너지에 따른 이웃 간 flux·확산 세기(면 길이)입니다. 에너지가 큰 곳은 셀이 작아져 면이 짧아지고(α > 0), 전선·제트에서 수치적 평활이 줄어드는 방향입니다.
- 생성점까지 에너지 밀도로 옮기는 진짜 적응 격자(Lloyd 재배치 + 매 단계 보존형 재보간)는 결정 카드의 B안입니다(코드 약 2일, 비용 약 2배).
- 예상: RMSE ±0.5 % (18g와 같은 크기). 주된 가치는 해석(α, 면적–에너지 상관).

## 6. 실행과 판정

- 구현: `gpclimate/nn/graph_core.py`(`deformed` 표본별·`fixed_areas`), `gpclimate/models/dynamics_vnet.py`(`core_energy_geom`, `ENERGY_COMPONENTS`, `core_energy`, `core_energy_weights`), `scripts/train_eval.py`(`--core-energy-geom`, 로그 `energy a … corr …`). 테스트 `tests/test_core_energy.py`.
- 체인: `runs/era5_64x32_s18h_energy/chain_s18h.sh` — CHAIN_S18G_DONE 대기, parent = 18g 채택 시 18g, 아니면 18g의 parent; config = parent + `core_energy_geom`; 같은 쌍 미세조정(V-Net stage9b에서 1000 step); 판정 parent 대비(+ base 대비); 마커 CHAIN_S18H_DONE. stage21 체인은 결정 카드에서 A가 선택되면 CHAIN_S18H_DONE을 기다리도록 바꿉니다.
- 보고: 학습된 α_c, 성분별 면적–에너지 상관(`energy_area_corr`), 면 이동·면적 비 범위, RMSE 판정(2019/2020/2022), ClimODE 대비 → `reports/compare_stage18h.md`.
