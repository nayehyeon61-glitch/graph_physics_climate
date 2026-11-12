# stage21a·stage21b: physics-first core 설계 (물리 항이 tendency를 끌게 하기)

작성 2026-11-09 15:30 KST. 요청:

- 감자 2026-11-09 15:06 KST: "우리의 목적은 학습데이터보다는 모델의 즉 물리적 graph에 있는데 이를 최대로 할려면?" → 제안 1(physics-first core) + 2(guide를 혼자서도 도는 역학 core로 학습) → 15:07 KST "오케이 그럼 1+2로 진행해보면?"

## 0. 왜 필요한가 — stage18 로그의 진단

stage18 Stage B(step 500) 로그: `core beta 0.11/0.09/0.08 rms T 0.022 F 0.009 S 0.162`. 6 h sub-step의 상태 변화에서 guide 수송 T와 보존형 간선 flux F를 합쳐도 자유 source S의 1/7입니다. 즉 지금 구조는 "물리 항을 하나씩 더하고 V-Net이 나머지를 자유롭게 채우는" 형태라서, 18d~18g가 물리 항의 **모양**을 어떻게 바꾸든 물리 graph의 **기여**는 작게 나옵니다. 논문 주장("guide가 예측 방정식을 만든다")의 최댓값은 β·F·D가 커져 물리 항이 tendency를 실제로 끌고 가는 경우이고, 그것을 구조로 강제하는 것이 stage21입니다.

## 1. stage21a physics-first core (`VNetConfig.core_beta_init`, `--core-beta-init 1`, Stage B `--source-penalty 0.1`)

sub-step 식은 stage18 그대로입니다: X_{k+1} = SL_{β⊙V}(X_k) − div_G J_θ + S_θ (+ D Laplace block).

- **β를 1에서 시작.** `beta_head.lin_s.bias = core_beta_init`(가중치는 0 초기화 그대로). β = 1이면 첫 step부터 guide의 흐름 V가 상태를 전부 실어 나르고(stage17c의 physics_only와 같은 수송), V-Net은 그 위의 보정을 배웁니다. stage18(β = 0 시작)은 학습이 β를 0.1 근처에서 멈췄는데, 이는 S가 이미 변화를 설명해 버린 뒤라 β를 키울 이유가 없었기 때문입니다. 1에서 시작하면 "β를 낮출 이유"가 있어야 낮아집니다. β는 여전히 [−0.5, 2]로 clamp된 학습 변수입니다.
- **source 예산.** 각 sub-step에서 r = ‖S_θ‖² / ‖ΔX‖²(V-Net source의 제곱 비중; 스칼라·벡터 채널 평균, 정규화 단위) 를 재고, Stage B 손실에 λ_src · mean_k r_k(sub-step과 rollout lead 평균)를 더합니다(`LieGuidedForecaster.source_cost`, checkpoint 경로에서도 같은 값 — 테스트). 척도가 없는 비율이라 λ_src = 0.1은 "변화의 100 %가 source면 손실 0.1"인 완만한 선호이고, 변화를 β 수송·보존형 flux·확산으로 설명할 수 있을 때만 S를 줄이게 합니다. S를 0으로 밀어붙이지는 않습니다(L_f가 더 큼).
- **진단.** `last_core["phys_share"]` = rms(T + F + D) / rms(ΔX), `["source_share"]` = √r. 학습 로그 `share P x.xx S x.xx`, eval.json `physics_share`(lead별, lie_guided의 마지막 sub-step 평균), eval.md "물리 항 비중" 표, `scripts/core_physics_summary.py`(물리 항 비중, guide 단독 기술 physics_only vs 지속성, guide 이득).
- 가설: 물리 항 비중이 stage18의 ≈0.15에서 0.5 이상으로 오르고 guide 이득(+5.6 %)이 커진다. RMSE는 ±1 % 안에서 같거나 조금 나빠질 수 있다(자유도 제한의 대가).

## 2. stage21b guide를 역학 core로 학습 (Stage A `--guide-path-steps 4 --guide-path-weight 1.0`)

지금 Stage A는 guide를 "특징 생산기"로 학습합니다(재구성 + 보조 + 6 h 경향 1 step). 그래서 guide 단독 예보(physics_only)는 24 h에서 지속성 대비 +33 %에 그치고 72 h부터 지속성보다 나쁩니다(stage17c). stage21b는 Stage A의 목적에 **guide 자신의 역학 경로**를 더합니다.

- guide의 역학: x_{j+1} = SL_{V(x_j)}(x_j) + S(x_j)·tend_scale — V-Net이 받는 core 항(stage18 `core_terms`)을 guide의 **자기 경로 상태**에서 다시 평가한 것입니다. 입력(섭동 포함)에서 4 × 6 h를 돌려 관측 상태 X_{t+6j}와 맞춥니다. 오차는 6 h 경향 단위(j = 1은 깨끗한 입력의 경향 손실과 같음)로 경로 평균.
- head만 학습(`--guide-heads-only 1`, trunk = stage9b 특징 P 그대로), base의 guide(`guide.pt`, 17c Stage A 결과)에서 시작, 17c/19와 같은 섭동(`--perturb-model` 17a pair, p 0.3). 1000 step, lr 5e-4. 경로 4 step이라 Stage A 비용 ≈ 5배(17c 10분 → ≈ 50분).
- 검증 `val path`(calib 16 시점, 경로 평균), 선택 점수에 포함(가중치 1).
- 그 뒤 Stage B는 stage21a와 같게(β 1 시작 + source 예산) 새 guide로 돌립니다. 즉 21b = 21a + 역학 core guide, 판정은 21a 대비(+ base 대비).
- 가설: physics_only의 24/48 h 기술이 오르고(+33 % → +40 % 이상), V-Net은 보정만 맡게 되어 guide 이득이 커진다.

## 3. 체인·판정

- `runs/era5_64x32_s21_physfirst/chain_s21.sh`: CHAIN_S18G_DONE을 기다려 18g의 base.txt/bflags.txt(18 계열 채택본과 그 Stage B guide 플래그)를 읽고, 21a(`runs/era5_64x32_s21a_physfirst`: base config + core_beta_init 1, Stage B `$BFLAGS --source-penalty 0.1`, pair·guide는 base 것) → 21b(`runs/era5_64x32_s21b_guidecore`: Stage A 위와 같이, Stage B 21a와 같이) 순으로 돌리고 CHAIN_S21_DONE을 남깁니다. stage19 체인은 CHAIN_S21_DONE을 기다리도록 바꿨고(2026-11-09 15:28 KST), base는 stage21의 base.txt로 그대로 전달됩니다 — **stage21은 체인 base를 바꾸지 않습니다.** 물리 지표로 판단할 단계라 RMSE 채택 규칙(5 lead 중 3개 이상, 평균 ≥ 0.5 %)은 기록용이고, stage20 통합(`stage20_compose.RUNS`, EXTRA_B `--source-penalty 0.1`)에서는 다른 단계와 같은 규칙으로 후보가 됩니다.
- 보고: `reports/compare_stage21.md` — 21a/21b vs base: RMSE(11변수, 5 lead, 2019·2020·2022), **물리 항 비중**(lead별), β·T/F/S/D rms, guide 단독 기술(physics_only vs 지속성), guide 이득(lie_guided vs no_guide), P=0 손실(diagnose_guide), 보존량.
- ETA(KST, 2026-11-09 기준): 18g 종료 ≈ 11-12 저녁 → 21a ≈ 11-13 오전(Stage B 9.5 h + 평가 1.2 h) → 21b ≈ 11-13 저녁(Stage A 50분 추가) → 19a/b/c ≈ 11-15 새벽 → stage20 ≈ 11-16 밤.

## 4. 구현 요약

- `gpclimate/models/dynamics_vnet.py`: `VNetConfig.core_beta_init`; `beta_head` bias 초기화; `core_rate`가 `_src_cost`(제곱 source 비중)와 `last_core["phys_share"/"source_share"]`를 남김; `_sub_step`이 (상태, cost)를 돌려 checkpoint 재계산과 일치; `integrate`가 `source_cost`(sub-step 평균)를 둠; core가 아닌 모델은 0.
- `scripts/train_eval.py`: cfg/`--core-beta-init`; Stage B `--source-penalty`(core 필요, `L_src` 로그·jsonl `l_src`); Stage A `--guide-path-steps/--guide-path-weight`(flow·tend head 필요, `L_path`/`val path`, jsonl `l_path`/`val_path`); 학습 로그 `share P S`; eval `physics_share` + eval.md 표.
- `scripts/core_physics_summary.py`; `tests/test_physics_first.py`(5: β 초기화·비중 보고, checkpoint 유무 동일·미분 가능, 예산 최소화가 source 비중을 낮춤, core 아닌 모델 0, 파이프라인 A→B→eval).
