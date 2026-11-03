# 실험 stage 목록 (구조·가설이 바뀔 때마다 한 stage)

원칙(감자, 2026-11-03): 모델 구조나 가설이 바뀌면 **별도 stage = 별도 run 디렉터리**로 나누고, 바로 앞 stage와 같은 데이터·step 수로 비교한다. 한 run에 두 가지 변경을 섞지 않는다. 모든 stage 비교표에는 외부 기준 **ClimODE**(저자 프로토콜, `runs/climode_2019_trainmode`, 공유 5변수) 열을 함께 넣는다(감자, 2026-11-03 04:52).

공통 조건: ERA5 5.625° (64×32) 11변수, 학습 2016–2017(데이터 stage S7은 예외, 행에 명시), calib 2018, 테스트 2019 (00 UTC 4일 간격 90개 초기시각), 메시 (4,3,2), Stage A 1000 step / Stage B 1500 step, batch 4, 단일 seed, CPU 4코어. 점수는 WeatherBench2 면적가중 RMSE.

| stage | 가설 / 구조 변경 | run | 앞 stage 대비 결과 | 보고서 |
|---|---|---|---|---|
| stage1 | 기본 Lie guide(재구성·와도/발산 보조) + 게이트 융합 V-Net + rollout 없음 | `runs/era5_64x32` | guide 의존도 낮음(P=0 → MSE +19 %), guide 유무 차이 1 % 미만 | `reports/`(첫 학습 보고) |
| stage2 | guide 보강: 고정 등변 물리 연산자 입력 + 6 h 경향 head | `runs/era5_64x32_physops` | 의존도 확보(P=0 → +121 %), 24 h 일부 개선·72–120 h 1–3 % 악화(자기 예보 상태에서 오차 증폭) | `reports/GUIDE_STRENGTHENING_KO.md` |
| stage3 | 학습식 변경: Stage A 예보상태 섭동 + Stage B 2-step rollout 손실 (guide 없음 baseline도 같은 손실로 재학습) | `runs/era5_64x32_physops_roll` | guide 있음이 모든 리드에서 guide 없음보다 낫고(0.5–5 %), 72–120 h −6–9 % | `reports/ROLLOUT_LOSS_KO.md` |
| stage4 | 융합 구조 변경: 게이트 융합 → PIGT(물리 통합 graph transformer, H/P/O 세 소스 공동 softmax, 1 layer) | `runs/era5_64x32_pigt` | stage3 대비 24–72 h 11/11 변수 개선(11변수 평균 −1.0/−1.0/−0.6 %), 96–120 h 동률(−0.3/0.0 %); 검증 MSE 0.0822 vs 0.0842; 어텐션 질량 O 29–47 %·P 30–47 %·H 15–31 %; P=0 → +69 %(stage3 +102 %: 연산자 토큰이 물리 일부를 직접 공급); V-Net 파라미터 +22 %, 단일 seed | `reports/PIGT_CLIMODE_KO.md` |
| 외부 기준 | ClimODE 공개 체크포인트를 같은 2019 프로토콜로 채점 (eval 모드 / 저자 프로토콜) | `runs/climode_2019`, `runs/climode_2019_trainmode` | 저자 프로토콜: 24–48 h z500은 ClimODE 우세, 72 h 이후 우리 모델 우세; eval 모드는 72 h부터 발산 | 〃 |
| stage5a | **층별 물리 제약 1 — guide 구조 변경**: 경향 = 고정 해석(resolved) 역학 F(X) + 학습 닫힘 S_θ (메시 0), F는 보정계수가 붙은 2D 원시방정식 항 | `runs/era5_64x32_s5a_closure` | stage3 대비 11변수 평균 RMSE +0.7/+0.7/+0.8/+0.6/+0.2 %(24/48/72/96/120 h; 소폭 악화, 개선 변수 2/1/0/1/4), guide 없음 대비 -0.7/-0.6/-0.3/-0.5/-0.5 %; 검증 MSE 0.0856 vs 0.0842(2-step 0.1567 vs 0.1546); 닫힘 구조는 설계대로 작동 — 6 h 경향 MSE 해석항만 0.61 → 해석항+닫힘 0.22(0 예측 0.70), 닫힘 기여 z500 24 %·t850 37 %·q700 55 %; P=0 악화 +89 %(stage3 +102 %), 섞은 P +140 %(stage3 +139 %); ClimODE는 24–48 h z500 우위(212/373 vs 253/404), 96 h부터 열세. **판정: 단독 채택 보류**, F+S_θ는 stage8 drift 후보(stage8d) | `docs/design/Layerwise_Physical_Constraints_KO.md` |
| stage5b | **층별 물리 제약 2 — Dynamics 학습식 변경**: Stage B에 PDE 잔차 손실 ‖(X̂−X_t)/Δt − [F+S_θ](X̄)‖² 추가 (stage5a guide 재사용) | `runs/era5_64x32_s5b_pde` | 대기 | 〃 |
| stage5c-0 | **Noise 층 추가(대조군)**: stage5a/stage5b 중 결정론 점수가 나은 run의 guide·V-Net 위에 Stage C residual graph diffusion(800 step, 제약 없음 = S1의 Stage C) | `runs/era5_64x32_s5c0_diffusion` | 대기 (stage5b 종료 후 자동) | 〃 |
| stage5c | **층별 물리 제약 3 — Noise 층 허용조건**: 같은 forecaster, Stage C에 사영 P_C(q700 ≥ 0, 전구 질량·p_s 가중 기둥수분 = μ) + L_feas + L_mean(비편향 쌍 추정량) | `runs/era5_64x32_s5c_noise` | 대기 | 〃 |
| stage7 | **데이터 stage(구조·학습식 동일)**: stage3 구성 그대로(게이트 융합, 물리 연산자, 경향 head, Stage A 예보상태 섭동, Stage B 2-step rollout, A 1000 / B 1500 step) 학습 데이터만 2년 → 10년(2008–2017, 14,612 시각; `data/era5_wb2_64x32_2008_2019.npz`). guide 없음 baseline과 섭동 모델도 10년으로 재학습. 가설: 현재 학습은 데이터·step 제한(6000 샘플 ≈ 2년 2 epoch, 검증 손실 계속 감소) → 같은 step 수에서 S3보다 좋아지면 다음은 step 3배 변형 | `runs/era5_64x32_s7_data10y` | 대기 (stage5c 종료 후 자동, `runs/era5_64x32_s7_data10y/chain_data.sh`); 비교 대상 stage3 | — |
| stage8a | **예측층 구조 변경: one-shot delta → graph neural ODE**(Poli et al. 2019; 감자 제안 2026-11-03): X̄ = X_t + Σ_k f_φ(X_k, P(X_k))/K, K = 2 sub-step(12 h) Euler, 물리 특징 sub-step마다 재계산. stage3 guide 재사용, 나머지 S3와 동일. 가설: 적분 구조가 72–120 h 오차 증가를 줄임 | `runs/era5_64x32_s8a_ode` | 대기 (stage7 종료 후 자동, `runs/era5_64x32_s8a_ode/chain_s8.sh`); 비교 대상 stage3 | `docs/design/Graph_Neural_ODE_Dynamics_KO.md` |
| stage8b | **학습된 물리 위에 ODE**: stage8a + 알려진 drift T_guide(동결 guide의 6 h 경향) — X_{k+1} = X_k + T_guide(X_k)·τ·(L/K) + δ f_φ/K, V-Net은 보정만 학습(초기 = 순수 물리 적분). 평가에 `physics_only` 행(V-Net 끔) 추가. 가설: 24 h 개선 + guide 의존도 상승 | `runs/era5_64x32_s8b_ode_phys` | 대기 (stage8a 다음); 비교 대상 stage8a | 〃 |
| stage8c (후속) | 6 h 중간 관측 감독(hybrid arc) + K = 4 — CPU 비용 5.4×라 GPU/야간 | — | stage8b 결과 후 | 〃 |
| stage10a | **물리 feature matrix 확장(FourCastNet 2022 / SFNO 2023 참고; 감자 제안 2026-11-03)**: 상태 채널 base11(5 스칼라 + 3 바람쌍) → fcn21(FourCastNet 20채널 + q700 = 13 스칼라 + 4 바람쌍; 추가 sp, z1000, z850, t500, z50, r500, r850, tcwv, u/v1000). 등변 분할(스칼라 / 접벡터 u+iv)·채널별 z-score(벡터쌍은 공통 스케일)·고정 조건(지형·해륙·∇지형·TOA·코리올리)은 유지, guide의 고정 물리 연산자 행렬은 47+19 → 101+30 채널, 보조 head 6 → 8(와도·발산 ×4층), 경향 head 11 → 21. 구조·학습식·데이터 기간은 stage3과 동일(A 1000 / B 1500, 2016–2017, guide 없음 baseline·섭동 모델도 fcn21로 재학습). 가설: 기둥 정보(지표기압, 1000/850 hPa 지위, t500, 습도, tcwv)가 P 수준과 6 h 경향 head를 개선해 11개 공통 변수의 RMSE가 내려간다(특히 t2m·msl·q700) | `runs/era5_64x32_s10a_fcn21` | 대기 (stage8b 종료 후 자동; 비교는 11개 공통 변수 + ClimODE 5변수) | `docs/design/Physical_Guide_Feature_Matrix_KO.md` |
| stage6 (계획) | **5층 기하·확률 동역학 GNN**(감자 설계서 2026-11): 1 discrete flux 보존형 핵 → 2 Lie 군 표현(기존 guide) → 3 Lie 대수 생성원 ψ_θ → 4 수송형 확률장 → 5 분할 적분기; stage6a–S6f로 분할(설계서 실험 A–F). 호몰로지/DEC 보강(감자 자료 2건): 메시는 이미 심플리셜 복합체, b=(1,0,1)이라 조화 성분 없음, 게이지 연결의 홀로노미 = 면적(가우스–보네) 확인 → DEC flux 핵·ψ_θ=⋆d₀ 생성원·L₀ 고유기저 잡음·persistent-homology 저기압 지표(설계 문서 8절) | 미정 | PIGT·stage5 결과 후 착수 | `docs/design/Five_Layer_Geometric_Stochastic_GNN_Plan_KO.md` |

## 데이터로 확인한 사실 (stage5a 설계 근거, 2026-11-03)

`scripts/resolved_fraction.py` (학습 기간 160시각): 관측 6 h 경향 T 대비 고정 항 F의 설명분산 `var(T−F)/var(T)`.

- 코리올리 항과 기압경도력 항은 **각각** 6 h 경향 분산의 50–80배 크기이며 지균 균형으로 상쇄된다. 두 항을 더한 비지균 가속도는 그래도 경향 분산의 2–23배(850/500/10 m)여서, 그대로 더하면 닫힘이 그 오차를 다시 상쇄해야 한다.
- 3개 바람 층으로 근사한 질량 연속방정식의 지표기압 경향은 경향 분산의 200배(쓸 수 없음). t2m 이류도 설명력 없음.
- 그래서 F의 각 항에 **학습 전 1회 최소제곱으로 맞춘 고정 보정계수**를 둔다(코리올리와 기압경도력은 같은 계수 → 상쇄 유지). 보정 후 설명분산: t850 13 %, q700 27 %, u/v850 19 %, u/v500 29 %, u/v10 9 %; t2m·msl 항은 제외. 계수: `runs/era5_64x32_s5a_closure/resolved_coef.json`.
