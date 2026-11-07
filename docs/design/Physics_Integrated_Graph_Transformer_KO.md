# 물리 통합 Graph Transformer (PIGT) 설계안

Lie(물리 가이드) → Dynamics(V-Net) → Noise(그래프 확산)의 세 단계에 **같은 등변 graph-transformer 블록**으로
물리 정보를 통합하고, 기여도 진단을 구조 안에 내장하기 위한 설계안. 2026-11-02 작성, 아직 구현 전.
기준 설계안 [`Lie_Dynamics_Diffusion_VNet_Design_KO.md`](Lie_Dynamics_Diffusion_VNet_Design_KO.md)의 §4.2(융합)·§5(확산)를 대체/확장한다.

## 0. 왜 필요한가 (진단 결과)

`runs/era5_64x32`(2016–17 학습, 2019 테스트)에서 학습된 V-Net의 가이드 의존도를 재어 보면

| 진단 | 값 | 해석 |
|---|---|---|
| 가이드 특징 P를 0으로 | 24h MSE +19 % | 쓰기는 하지만 크지 않다 |
| P를 다른 시각 것으로 교체 | +8 % | P의 절반 가까이는 "현재 상태의 공간 문맥"이고 시각 고유 정보가 적다 |
| 24h 경향 선형 probe R² (상태만 / P만 / 둘 다) | 0.09 / 0.17 / 0.18 | P는 역학(경향) 정보를 조금만 담는다 |
| 융합 게이트 β (초기값 0.1) | 0.10–0.125 | 학습 동안 거의 움직이지 않음 |

원인은 두 겹이다.

1. **내용**: Stage A 목표(복원·와도/발산)가 "현재 상태를 잘 설명"하게 할 뿐 "다음 상태를 설명"하게 하지 않는다.
   → 1차 보강(이 PR): 고정 등변 물리 연산자(기울기·발산/회전·이류·코리올리·절대와도 이류·신장·라플라시안)를
   가이드 입력에 넣고, 6h 경향 예측 head를 추가. 연산자만으로도 24h 경향 probe R²가 0.22–0.34로 올라간다(상태만 0.06–0.14).
2. **구조**: 융합이 (i) 한 방향(H가 P를 읽기만 함), (ii) 레벨당 cross-attention 한 층 + 작은 게이트, (iii) Stage C에서는
   attention 없이 concat(`Merge`)이다. 물리가 역학 표현을 "빚어낼" 통로가 좁고, 노이즈 단계는 물리를 거의 보지 못한다.
   → 2차 보강(이 문서): **PIGT**.

## 1. 토큰과 블록

레벨 ℓ(Fine/Medium/Coarse)마다 세 종류의 노드 토큰을 둔다. 모두 `Field(s, v)`(스칼라 = 주파수 0, 벡터 = 주파수 1).

| 토큰 | 출처 | 학습 여부 |
|---|---|---|
| **H_ℓ** dynamics | V-Net 인코더/디코더 | 학습 |
| **P_ℓ** physics | 고정된 Stage A 가이드 `{P0,P1,P2,P1*,P0*}` | 고정 |
| **O_ℓ** operators | `PhysicsOperators`를 레벨 ℓ 메시(`mesh_edges[ℓ]`, pooling된 상태)에 적용 | 파라미터 없음 |

**PIGT layer**(레벨 ℓ, `attn_edges[ℓ]` = 1-ring + self loop 위에서):

```
H ← H + Attn_eq(Q = H,  K/V = [H, P, O] + type-embedding)      # dynamics가 물리를 읽는다
H ← H + FFN_eq(Norm(H))
P ← P + Attn_eq(Q = P,  K/V = [P, H])                            # 물리 표현이 역학 문맥으로 다듬어진다 (양방향)
P ← P + FFN_eq(Norm(P))
```

* Attention은 기존 `EqCrossAttention`(graphtransformer 구조, LieTransformer 위치 커널)을 재사용한다. K/V 벡터는
  수신 노드 게이지로 평행이동하고, logit은 불변량(|·|, Re⟨q, e^{iα}k⟩, 거리/방향 기저)으로만 만든다 → 등변성 유지.
* type-embedding은 소스(H/P/O)별 학습 스칼라 벡터를 K의 스칼라 채널에 더한다(스칼라이므로 등변).
* P-stream의 가중치는 Stage B(또는 C) 소유다. 가이드 자체는 계속 고정이므로 Stage A의 재사용성과 "가이드 vs 없음" 비교는 유지된다.
* 레벨당 2층(설정값). 게이트 β는 없앤다. 대신 잔차 크기 비 ‖ΔH‖/‖H‖와 소스별 attention 질량을 기록한다.
* 선택 사항 `source_dropout`(p≈0.1): 학습 중 P 또는 O 토큰을 통째로 0으로. 켜면 "P를 0으로" 반사실이 분포 밖 입력이 아니게 되어
  기여도 측정이 정당해지지만 의존도 자체는 낮아질 수 있으므로, 기본은 끄고 진단용 실험에서만 켠다.

## 2. 단계별 적용

| 단계 | 현재 | PIGT 적용 후 |
|---|---|---|
| A (Lie guide) | 복원 + 와도/발산 (+ 1차 보강: 연산자 입력, 6h 경향) | 변경 없음 |
| B (Dynamics V-Net) | 레벨당 `F = H + β·CrossAttn(Q=H, K=V=P)` 1층 | 레벨당 PIGT 2층, 토큰 [H, P, O] |
| C (Noise, 잔차 확산) | `Merge(concat)`으로 P0*만 주입 | 레벨 0·1에서 PIGT, 토큰 [R(노이즈 잔차), H*(결정론 예보의 V-Net 특징, detach), P, O] |

Stage C까지 같은 블록을 쓰면 앙상블 퍼짐이 물리 상황(경압 불안정 영역, 강한 이류 구역)에 따라 달라질 통로가 생긴다.
지금은 Stage C의 CRPS가 독립 가우시안 기준선과 같은데, 이 구조 변화가 그 원인(물리 정보 부족)을 겨냥한다.

## 3. 내장 기여도 진단

학습 로그와 `scripts/diagnose_guide.py`에 다음을 추가한다.

1. **소스별 attention 질량**: 레벨·층별로 Q=H가 [H, P, O]에 준 attention 합의 평균. 학습 곡선으로 기록 → "P 기여도 곡선".
2. **반사실 제거**: P=0, O=0, P·O 모두 0, P 셔플, O 셔플 각각의 24h MSE 변화(기존 P-dependence 확장).
3. **선형 probe**: H / P / O / H+P+O 각각으로 24h 경향 R².
4. **채택 기준**: 같은 가이드·같은 step·2 seed에서 (a) P=0 열화가 커지고(물리를 실제로 더 쓰고) **그리고** (b) 2019년 RMSE가
   게이트 융합보다 좋아야 채택. (a)만 좋아지면 "쓰긴 하는데 도움은 안 됨"이므로 채택하지 않는다.

## 4. 실험 순서 (한 번에 한 가지만 바꾼다)

| # | 변경 | 비교 대상 | CPU 4코어 예상 |
|---|---|---|---|
| 1 (진행 중) | 가이드 보강(연산자 + 6h 경향), 융합은 기존 게이트 | 기존 가이드 / 가이드 없음 | A 19분 + B 50분 + 평가 10분 |
| 2 | Stage B 융합을 PIGT로 | #1 | B 약 75분(레벨당 2층, 토큰 3배) |
| 3 | Stage C에 PIGT | 가우시안 기준선, 기존 Stage C | C 약 30분 + 앙상블 평가 |
| 4 | 1.5° KAI 격자(121×240, 67채널)로 이동 | — | GPU 필요 |

## 5. 구현 계획 (파일 단위)

* `gpclimate/nn/physics_ops.py`: 레벨 ℓ용 연산자 — 상태를 `transport_interpolate(down)`으로 pooling한 뒤 `mesh_edges[ℓ]` 스텐실 적용.
  `Stencil`을 그래프별로 한 번만 만들어 캐시.
* `gpclimate/nn/attention.py`: `PIGTLayer`(위 블록), `PhysicsIntegration`(레벨별 스택, 소스 질량 반환).
  `VNetConfig.fusion = "gated" | "pigt"`로 선택, 기존 체크포인트는 `"gated"`로 로드.
* `gpclimate/models/dynamics_vnet.py`: `_fuse`에서 선택적으로 PIGT 사용, `last_attention_mass` 노출.
* `gpclimate/models/diffusion.py`: `ResidualDenoiser(fusion="pigt")`, ctx에 H*와 O 추가.
* `scripts/diagnose_guide.py`: 소스 질량·O 반사실·O probe.
* `tests/test_equivariance.py`: PIGT layer, PIGT V-Net, PIGT denoiser의 게이지 등변성(허용 오차 2e-5).
