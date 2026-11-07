# 구현 노트: 설계안 → 코드 대응

기준 설계안: [`docs/design/Lie_Dynamics_Diffusion_VNet_Design_KO.md`](design/Lie_Dynamics_Diffusion_VNet_Design_KO.md) (2026-11-02).
이번 구현 범위는 설계안 §12의 최소 범위(Stage A + Stage B)와 Stage C 골격이다.

## 1. 설계 절 ↔ 코드

| 설계안 | 코드 | 비고 |
|---|---|---|
| §2.2 구면 그래프 \(\mathcal G_\ell\) (Fine/Medium/Coarse) | `gpclimate/geometry/graphs.py` `SphereGraphs` | physicsnemo(GraphCast) icosahedral 계층. 기본 `mesh_splits=(4,3,2)` = 2562/642/162 노드 |
| §3.2 국소 접평면, 평행이동 \(P_{j\to i}\) | `gpclimate/geometry/sphere.py` `edge_geometry`, `transport_rotation` | Levi-Civita 평행이동을 게이지 좌표의 SO(2) 각도 \(\alpha_{ij}\)로 저장. 2×2 행렬의 log는 LieConv `SO2.log` |
| §3.2 등변 메시지 \(m_{ij}=\psi(h_i,P_{j\to i}h_j,e_{ij})\) | `gpclimate/nn/gauge_conv.py` `GaugeConv` | 평행이동 후 edge frame으로 회전 → 불변 좌표에서 MLP → 원래 게이지로 복귀. 조건식 \(K(\phi-\theta)=\rho_{out}(-\theta)K(\phi)\rho_{in}(\theta)\)의 일반해이므로 함수 구성 자체가 등변 |
| §3.2 비선형성·정규화의 등변성 | `gpclimate/nn/so2.py` `Gate`, `EqNorm`, `EqLinear` | 벡터에는 불변량으로 만든 sigmoid gate, RMS norm, 복소 가중치 선형층 |
| §3.1 \(E_\theta(\widetilde X_t, T_g(c_t), \mathcal G, \sigma)\) → \(\{P_0,P_1,P_2,P_1^*,P_0^*\}\) | `gpclimate/models/lie_physical_gnn.py` `LiePhysicalGNN` | \(\sigma\)를 스칼라 조건 채널로 입력 |
| §3.3 \(\mathcal L_A\) (복원 + 물리 보조 + 등변성) | `scripts/smoke_test.py` Stage A, `gpclimate/training/losses.py` | 복원 목표는 \(T_g(X_t)\). 물리 보조 = 바람쌍별 상대와도·발산. 등변성은 구조적으로 보장되므로 진단값으로만 기록 |
| §4.2 \(F_\ell = H_\ell + \beta_\ell\,\mathrm{CrossAttn_{eq}}(Q=H_\ell, K=V=P_\ell)\) | `gpclimate/nn/attention.py` `EqCrossAttention`, `PhysicsFusion` | 1-ring + self loop 이웃. K/V 벡터는 수신 노드 게이지로 평행이동 |
| §4.3 Forward 순서 | `gpclimate/models/dynamics_vnet.py` `DynamicsVNet.forward` | 의사코드와 같은 순서·skip(H1, H0) |
| §4.3 pool·unpool (면적 가중 + 평행이동) | `GaugeTransfer` + `transport_interpolate` | pooling은 면적×가우시안 가중 평균, unpooling은 IDW. 둘 다 평행이동 후 합산하고 학습형 gauge conv로 보정 |
| §4.4 고정 간격 변화량 \(\bar X = X_t + \Delta X_\phi\) | `LieGuidedForecaster.forward` | \(\Delta\)는 이미 간격 변화량(\(\Delta t\) 재곱 없음). head는 0으로 초기화 → 시작점 = persistence |
| §4.1 가이드 가중치 고정, 입력 미분은 유지 | `LieGuidedForecaster.freeze_guide` | `requires_grad=False`만 설정, `no_grad` 미사용(rollout 손실에서 입력 gradient 필요) |
| §4.5 \(\mathcal L_{phys}\) | `global_tendency_consistency` | **약한 형태**: 면적 적분 변화량이 관측과 일치하도록. 0으로 강제하지 않음. 보존형 flux(\(\Phi_{ij}=-\Phi_{ji}\))는 아직 미구현 |
| §5 Conditional Graph Diffusion | `gpclimate/models/diffusion.py` | cosine schedule, \(\epsilon\)-예측, DDIM(\(\eta=1\)) 재간격 샘플링. 벡터 잡음은 등방 복소 가우시안 |
| §5.3 앙상블 조립 \(X^{(k)}=\bar X + R^{(k)}\) | `assemble_ensemble` | 잔차 평균을 강제로 0으로 만들지 않음 |
| §7 잔차 분포는 결정론 모델이 보지 않은 기간에서 | smoke test `calib` 기간 | Stage C 학습과 잔차 스케일 추정에 train과 다른 해(2019년 1월)를 사용 |
| §8 다단계 rollout, 매 시점 물리 특징 재계산 | `LieGuidedForecaster.rollout` | 결정론 rollout만 구현. 멤버별 rollout은 같은 함수를 멤버마다 호출하면 됨 |
| §9 PG-Flow 입출력 계약 | `gpclimate/contracts/pgflow_kai.py` `KAIAdapter` | KAI 67채널 ↔ 내부 표현. u/v를 공통 스케일로 재정규화(설계안 §9 경고 반영), 역변환 정확 |
| §10 평가 | `gpclimate/training/metrics.py` | WeatherBench2 `RMSESqrtBeforeTimeAvg`, `CRPS`, `EnsembleMeanRMSESqrtBeforeTimeAvg`, `EnsembleStddevSqrtBeforeTimeAvg` |
| §11.1–11.2 등변성 범위·확률적 등변성 | `tests/test_equivariance.py` | GNN, pool/unpool, 융합층, 출력층, 전체 예측기, denoiser 모두 무작위 국소 게이지 회전으로 검증 |

## 2. 연결 저장소 재사용 방식

| 저장소 | 재사용한 것 | 방식 |
|---|---|---|
| physicsnemo | GraphCast icosahedral 계층, `faces_to_edges`, `latlon2xyz`/`xyz2latlon` | fork 소스 파일을 직접 실행(`gpclimate/external/repos.py`). 패키지 `__init__`이 warp·hydra 등을 끌어오므로 해당 파일만 로드하고, 그 파일이 쓰는 `check_version_spec` 하나만 대체 |
| LieConv | `SO2.log`, `SO2.lifted_elems`, `Swish` | sys.path에 fork 추가 후 import |
| lie-transformer | `MultiheadLinear` (위치 커널) | 같은 방식 |
| graphtransformer | 레이어 구조 (Q/K/V/E, edge 변조, clamp, O, residual+norm+FFN) | DGL 기반이라 import하지 않고 PyG로 재구성 |
| pytorch_geometric | `MessagePassing`, `MLP`, `softmax`, `scatter` | fork에서 pip 설치 |
| e3nn | `soft_one_hot_linspace`, `FullyConnectedNet` | fork에서 pip 설치 |
| weatherbench2 | metrics, ERA5 64×32 공개 zarr | fork를 path import (apache-beam 불필요) |
| neuralgcm, dinosaur | 사용 안 함 | JAX 기반. 스펙트럴 물리 진단(정확한 와도·발산, 물리 잔차)의 후보 |

## 3. 내린 결정과 설계안과의 차이

1. **게이지 선택**: 격자 노드는 ERA5 u/v 정의와 같은 (동, 북) 게이지, 메쉬 노드는 위경도 기반 (동, 북) 게이지를 기본값으로 쓴다. 네트워크가 게이지 등변이므로 선택은 결과에 영향이 없고, 테스트는 노드마다 무작위 게이지로 바꿔 확인한다. 극점의 격자점도 자기 경도로 정의된 게이지를 가지므로 KAI 121×240(극 포함)에서도 정의된다.
2. **\(T_g\)의 의미**: 첫 구현에서 \(T_g\)는 국소 좌표계(게이지) 회전이다. 지구 전체를 회전시키는 SO(3) 대칭은 주장하지 않는다(고정 메쉬, 자전축, 지형). 정확히 등변인 네트워크에서는 \(T_g\) 증강이 학습에 정보를 더하지 않지만, 설계안의 학습 목표를 그대로 따르고 구현 검증 역할을 하도록 남겨 두었다.
3. **입력 교란 \(L\)**: 정규화 단위에서 \(L=I\), 벡터는 등방 복소 잡음, \(\sigma\sim\{0\ (25\%),\ U(0,0.5)\}\).
4. **물리 보조 과제**: 바람쌍별 상대와도·발산(위경도 격자 유한차분, 극 근처 행은 마스크). 미래 관측은 쓰지 않는다.
5. **복소 가중치**: 벡터 채널 선형층은 복소 가중치이므로 90° 회전(\(\hat k\times\))을 표현할 수 있다. SO(2)에는 등변이지만 반사(O(2))에는 등변이 아니다. 구면의 방향(바깥 법선)이 고정되어 있으므로 SO(2)가 맞는 군이다.
6. **융합 게이트 \(\beta_\ell\)**: 채널별 실수(불변) 파라미터, 초기값 0.1.
7. **Grid ↔ graph adapter**: 고정 평행이동 보간 + 학습형 bipartite gauge conv(GraphCast의 grid2mesh/mesh2grid 역할). head의 수신 상태에는 현재 상태의 격자 임베딩을 함께 넣는다.
8. **메쉬 해상도**: 기본 `(4,3,2)`의 가장 고운 메쉬(평균 간격 4.3°)는 KAI 1.5° 격자보다 거칠다. 실제 KAI 학습에서는 `mesh_splits=(5,4,3)`(10242 노드, 약 2.2°) 이상을 권장한다. KAI 121×240 텐서 전체 경로(어댑터 → 모델 → 어댑터)는 `tests/test_contract.py`에서 확인한다.
9. **PG-Flow 계약**: `KAIAdapter`는 텐서 수준 변환까지만 제공한다. PG-Flow의 데이터 로더·exporter와의 연결, 일 평균(\(\Delta t\)=1일) 데이터로의 학습은 아직 하지 않았다. 스모크 테스트는 WB2 6시간 간격 순간값에서 24시간 lead를 사용했다.

## 4. 아직 하지 않은 것

- 보존형 flux 구조와 질량·수분·에너지 budget 손실 (PG-Flow `pgcfm/physics` 잔차 연결 포함)
- 다단계 rollout 손실 \(\mathcal L_{rollout}\)의 학습 루프 (rollout 함수는 있음)
- 멤버별 앙상블 rollout (§8), rank histogram·spread–skill·스펙트럼 평가
- 학습 가능한 Lie generator, 물리시간 SDE, 기억 구조, 공동 미세조정, 고해상도 확장 (설계안 §12 후속 실험)
- 주파수 2 이상의 SO(2) 표현 (현재 스칼라 + 접벡터만)
