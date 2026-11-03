# 실험 stage 목록 (구조·가설이 바뀔 때마다 한 stage)

원칙(감자, 2026-11-03): 모델 구조나 가설이 바뀌면 **별도 stage = 별도 run 디렉터리**로 나누고, 바로 앞 stage와 같은 데이터·step 수로 비교한다. 한 run에 두 가지 변경을 섞지 않는다.

공통 조건: ERA5 5.625° (64×32) 11변수, 학습 2016–2017, calib 2018, 테스트 2019 (00 UTC 4일 간격 90개 초기시각), 메시 (4,3,2), Stage A 1000 step / Stage B 1500 step, batch 4, 단일 seed, CPU 4코어. 점수는 WeatherBench2 면적가중 RMSE.

| stage | 가설 / 구조 변경 | run | 앞 stage 대비 결과 | 보고서 |
|---|---|---|---|---|
| S1 | 기본 Lie guide(재구성·와도/발산 보조) + 게이트 융합 V-Net + rollout 없음 | `runs/era5_64x32` | guide 의존도 낮음(P=0 → MSE +19 %), guide 유무 차이 1 % 미만 | `reports/`(첫 학습 보고) |
| S2 | guide 보강: 고정 등변 물리 연산자 입력 + 6 h 경향 head | `runs/era5_64x32_physops` | 의존도 확보(P=0 → +121 %), 24 h 일부 개선·72–120 h 1–3 % 악화(자기 예보 상태에서 오차 증폭) | `reports/GUIDE_STRENGTHENING_KO.md` |
| S3 | 학습식 변경: Stage A 예보상태 섭동 + Stage B 2-step rollout 손실 (guide 없음 baseline도 같은 손실로 재학습) | `runs/era5_64x32_physops_roll` | guide 있음이 모든 리드에서 guide 없음보다 낫고(0.5–5 %), 72–120 h −6–9 % | `reports/ROLLOUT_LOSS_KO.md` |
| S4 | 융합 구조 변경: 게이트 융합 → PIGT(물리 통합 graph transformer, H/P/O 세 소스 공동 softmax, 1 layer) | `runs/era5_64x32_pigt` | 학습 중 (2026-11-03) | `reports/PIGT_CLIMODE_KO.md` (예정) |
| 외부 기준 | ClimODE 공개 체크포인트를 같은 2019 프로토콜로 채점 (eval 모드 / 저자 프로토콜) | `runs/climode_2019`, `runs/climode_2019_trainmode` | 저자 프로토콜: 24–48 h z500은 ClimODE 우세, 72 h 이후 우리 모델 우세; eval 모드는 72 h부터 발산 | 〃 |
| S5a | **층별 물리 제약 1 — guide 구조 변경**: 경향 = 고정 해석(resolved) 역학 F(X) + 학습 닫힘 S_θ (메시 0), F는 보정계수가 붙은 2D 원시방정식 항 | `runs/era5_64x32_s5a_closure` | 대기 (PIGT 종료 후 시작) | `docs/design/Layerwise_Physical_Constraints_KO.md` |
| S5b | **층별 물리 제약 2 — Dynamics 학습식 변경**: Stage B에 PDE 잔차 손실 ‖(X̂−X_t)/Δt − [F+S_θ](X̄)‖² 추가 (S5a guide 재사용) | `runs/era5_64x32_s5b_pde` | 대기 | 〃 |
| S5c-0 | **Noise 층 추가(대조군)**: S5a/S5b 중 결정론 점수가 나은 run의 guide·V-Net 위에 Stage C residual graph diffusion(800 step, 제약 없음 = S1의 Stage C) | `runs/era5_64x32_s5c0_diffusion` | 대기 (S5b 종료 후 자동) | 〃 |
| S5c | **층별 물리 제약 3 — Noise 층 허용조건**: 같은 forecaster, Stage C에 사영 P_C(q700 ≥ 0, 전구 질량·p_s 가중 기둥수분 = μ) + L_feas + L_mean(비편향 쌍 추정량) | `runs/era5_64x32_s5c_noise` | 대기 | 〃 |
| S6 (계획) | **5층 기하·확률 동역학 GNN**(감자 설계서 2026-11): 1 discrete flux 보존형 핵 → 2 Lie 군 표현(기존 guide) → 3 Lie 대수 생성원 ψ_θ → 4 수송형 확률장 → 5 분할 적분기; S6a–S6f로 분할(설계서 실험 A–F) | 미정 | PIGT·S5 결과 후 착수 | `docs/design/Five_Layer_Geometric_Stochastic_GNN_Plan_KO.md` |

## 데이터로 확인한 사실 (S5a 설계 근거, 2026-11-03)

`scripts/resolved_fraction.py` (학습 기간 160시각): 관측 6 h 경향 T 대비 고정 항 F의 설명분산 `var(T−F)/var(T)`.

- 코리올리 항과 기압경도력 항은 **각각** 6 h 경향 분산의 50–80배 크기이며 지균 균형으로 상쇄된다. 두 항을 더한 비지균 가속도는 그래도 경향 분산의 2–23배(850/500/10 m)여서, 그대로 더하면 닫힘이 그 오차를 다시 상쇄해야 한다.
- 3개 바람 층으로 근사한 질량 연속방정식의 지표기압 경향은 경향 분산의 200배(쓸 수 없음). t2m 이류도 설명력 없음.
- 그래서 F의 각 항에 **학습 전 1회 최소제곱으로 맞춘 고정 보정계수**를 둔다(코리올리와 기압경도력은 같은 계수 → 상쇄 유지). 보정 후 설명분산: t850 13 %, q700 27 %, u/v850 19 %, u/v500 29 %, u/v10 9 %; t2m·msl 항은 제외. 계수: `runs/era5_64x32_s5a_closure/resolved_coef.json`.
