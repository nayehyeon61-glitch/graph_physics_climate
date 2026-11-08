# stage20 통합 버전 설계 (stage17–19의 이점 모음)

작성 2026-11-07 23:30 KST. 요청:

- 감자 2026-11-07 23:10 KST: "마지막으로 이후 이점들을 모아서 stage 20으로 해서 통합 버젼 실험해보죠"
- 결정 카드 2026-11-07 23:15 KST: "3000 step + 대조군"

stage9(stage3–15 이점 통합)와 같은 방식으로, stage17–19에서 이점이 확인된 구성만 모아 더 큰 예산으로 한 번 학습합니다.

## 1. 구성 선택 규칙 (`scripts/stage20_compose.py`)

- stage17–19의 모든 run은 같은 짝 미세조정(stage9b에서 1000 step)으로 학습했습니다. 각 run은 바로 앞 stage와 비교해 판정했고, 체인마다 앞에서 채택된 쪽 위에 쌓았기 때문에 각 run의 `config.json`에는 그 경로에서 채택된 구성이 이미 누적되어 있습니다.
- 그래서 통합 구성은 **공통 대조군 stage17-ctl(같은 예산, 새 구조 없음)에 대해 채택 기준(5개 선행 중 3개 이상 개선, 평균 ≥ 0.5 %)을 통과한 run 중 평균 이득이 가장 큰 run의 구성**으로 정합니다.
- 그 run의 guide(Stage A, PINN 전처리 `pinn_scales.json`, 그래프 임베딩 `graph_embed.npy`)는 그대로 씁니다.
- 그 run이 Stage B에서 쓴 옵션도 그대로 씁니다. guide head 경로 미세조정은 `guide_finetuned.pt`가 있으면, 경로 PINN은 stage19c인 경우입니다.
- 통과한 run이 없으면 stage20은 학습하지 않고, 통합 모델은 stage9b로 남는다고 기록합니다.
- 선택 결과와 모든 run의 대조군 대비 이득은 `runs/era5_64x32_s20_integrated/compose.json`에 남습니다.

## 2. 학습 (`runs/era5_64x32_s20_integrated/budget.env`, 결정 카드의 선택)

| 단계 | 내용 |
| --- | --- |
| guide 없는 짝 | stage9b 짝에서 3000 step, lr 3e-4, rollout 2(처음부터) |
| guide 모델 Stage B | stage9b V-Net에서 3000 step, 선택된 run의 Stage B 옵션 포함 |
| Stage C | 잔차 graph diffusion 800 step, stage9b의 noise 제약(실현 가능 투영, 물 보존, 평균 손실) |
| 평가 | 2019 결정론과 앙상블(24 사례, 8 member, CRPS), guide 진단, 2020/2022 |
| 같은 예산 대조군 | `runs/era5_64x32_s20_ctl`: stage9b 구성(delta, 새 구조 없음)을 같은 3000 step으로 짝 모두 미세조정, 평가, 2020/2022 |

## 3. 판정과 비교

- **채택 판정:** stage20 vs 같은 예산 대조군(`decide_adoption`, 평균 ≥ 0.5 %). 구조만의 이득입니다.
- **함께 보고:**
  - stage20 vs stage9b(예산이 달라 참고용)
  - stage20 vs 선택된 1000 step run(예산 효과)
  - guide 이득(guide 없는 짝 대비)
  - ClimODE 열
  - CRPS와 spread/skill
  - physics_only, β, 항별 크기(T F S D K), PINN 잔차 비율
- **보고서:** `reports/compare_stage20.md`(`scripts/compare_runs.py`)
- **실행:** `runs/era5_64x32_s20_integrated/chain_s20.sh`(CHAIN_S19_DONE 대기, 표지 CHAIN_S20_DONE)
- **예산 변경:** `budget.env`만 고칩니다(실행 중인 체인 스크립트는 고치지 않음).

## 4. 비용과 위험

- ODE 4 sub-step(guide와 core를 매 sub-step 다시 계산)은 delta보다 step당 약 5배 비쌉니다. guide 모델 3000 step은 약 17 h, 짝은 약 5 h, 대조군은 약 4 h, Stage C와 평가는 약 2 h로 전체 약 1.2일입니다. stage17a 실측으로 다시 확인합니다.
- **stage17a 실측(2026-11-08 14:20 KST)으로 다시 계산:** guide 모델 ODE Stage B는 batch 4에서 메모리 14 GB를 넘겨 종료됐고, sub-step gradient checkpointing(같은 기울기, 최대 6.7 GB)을 넣은 뒤 step당 약 31초입니다(delta guide 모델 4.7초의 6.6배). guide 없는 ODE 짝은 checkpointing 전 6.4초였고, 지금은 약 8초로 봅니다. 그래서 guide 모델 3000 step은 약 26–31 h(core 구성은 더 김), 짝은 약 7 h, 대조군은 약 5 h, Stage C와 평가는 약 2 h로 전체 약 1.7–1.9일입니다.
- 누적 구성이라, 선택된 run에는 그 경로에서 단독으로는 기각된 구성이 포함될 수 있습니다. 예를 들어 19a가 기각돼도 19b에는 PINN이 남습니다. 그래도 대조군 대비 가장 좋은 조합이라는 점은 실측으로 확인된 것이고, 어떤 구성이 들어갔는지는 `compose.json`에 그대로 기록합니다.
