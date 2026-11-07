# 층별 물리 제약으로 physical guide 보강 (stage5a / stage5b / stage5c)

출처: 감자 첨부 제안서 「물리제안.txt」(2026-11-03) — Lie 층에는 등변성 제약, Dynamics 층에는 운동량·열역학·수분·질량 방정식의 잔차, Noise 층에는 분포 학습과 물리적 허용 조건을 두고, 전체를 `dX/dt = F_primitive(X,c) + Ψ_θ(X,Z,c)` (고정 역학 + 학습 닫힘) 구조로 쓰자는 제안.

## 1. 제안 ↔ 우리 구조 대응

| 제안서 항목 | 우리 구현 | 방식 | stage |
|---|---|---|---|
| Lie 층 등변성 잔차 L_eq | SO(2) 게이지 등변성이 **구조적으로 정확**(테스트 18+2개, 상대오차 < 2e-5). 손실이 아니라 진단치로만 보고 | hard | 기존 |
| Lie 층 입력에 δ, ζ, ∇ℓ, ∇Φ 등 물리량 | 고정 연산자(발산·회전·기울기·이류·코리올리·절대와도·라플라시안)를 메시 0 입력으로 이미 사용 (stage2) | hard | 기존 |
| Dynamics: `dX/dt = F_primitive + Ψ_θ` | **guide의 경향 head를 닫힘 head로 교체**: 경향 = F(X) + S_θ. F는 `gpclimate/nn/primitive2d.py`의 고정 등변 연산자, S_θ는 메시 0의 선형 등변 읽기(`closure_out`). Stage A 목표 = 관측 6 h 경향 − F(X_t) | 구조(hard F) + 손실 | **stage5a** |
| 운동량 방정식 잔차 R_u | u500: −(u·∇)u − f k×u − ∇Φ500; u850: Φ850 = R_d T̄ ln(msl/850 hPa) (T̄ = (t2m+t850)/2); u10: −R_d t2m ∇ln msl. 연직 이류·마찰은 닫힘 | F 항 | stage5a |
| 열역학 방정식 잔차 R_T | t850: −u850·∇T (단열항 κTω/p는 ω가 없어 닫힘), t2m: 이류항은 설명력 없어 제외 | F 항 | stage5a |
| 수분 방정식 (보존형) | q700: −u700·∇q (u700은 850/500의 log-p 보간). 플럭스 보존형은 연직 구조가 없어 이류형으로 | F 항 | stage5a |
| 질량 연속 ∂ln p_s/∂t = −∫(δ+u·∇ln p_s)dσ | 3개 바람 층 구적으로 시도 → 경향 분산의 200배(상쇄 실패)라 **제외**, 닫힘이 담당 | 제외(기록) | stage5a |
| 정역학 Φ = H(T_v) 하드 복원 | 기둥 온도가 없어(t850, t2m뿐) 복원 불가. z500은 닫힘 전용 | 불가(기록) | — |
| PDE 잔차 손실 L_PDE (관측 궤적) | Stage A의 닫힘 손실이 곧 관측 궤적에서의 PDE 잔차 ‖T_obs − F − S_θ‖² | 손실 | stage5a |
| PDE 잔차를 rollout 결과에 평가 | Stage B: ‖(X̂−X_t)/Δt_lead − [F+S_θ](X̄)‖², X̄ = (X_t+X̂)/2 (사다리꼴). guide 가중치 고정, 기울기는 X̂로만 흐름 | 손실 | **stage5b** |
| Noise 층: 허용 조건 사영 P_C, L_mean, 질량 가중 보존 | Stage C(residual graph diffusion)에 **사영 P_C(μ)**: q700 ≥ 0, 전구 질량 ∫msl dA = μ의 값, p_s 가중 기둥수분 ∫msl·q700 dA = μ의 값 (hard). 손실: L_DSM + λ_feas L_feas(x̂₀ 예측의 q<0·질량 drift) + λ_mean L_mean(최종 사영 샘플의 앙상블 평균 − μ, 비편향 쌍 추정량). 6절 | 사영(hard) + 손실 | **stage5c-0 / stage5c** |
| 상변화 결합 (q_v+q_l+q_i, c_p s_T + L_v s_q) | q_l, q_i가 없어 불가 | 불가(기록) | — |

## 2. S5a의 식 (우리 11변수 2D 축소형)

메시 0에서 고정 연산자(1-ring 최소제곱 기울기·야코비안, 쌍선형 격자→메시 보간)로

- u500: F = −c_adv (u·∇)u − c_ageo [ f k×u + ∇Φ500 ]
- u850: F = −c_adv (u·∇)u − c_ageo [ f k×u + ∇Φ850 ],  Φ850 = R_d T̄ ln(msl/850 hPa)
- u10:  F = −c_adv (u·∇)u − c_ageo [ f k×u + R_d t2m ∇ln msl ]
- t850: F = −c_T u850·∇t850,  q700: F = −c_q u700·∇q700,  z500·t2m·msl: F = 0

을 물리 단위(SI/s)로 계산해 6 h·정규화 경향 단위로 바꾸고, Stage A에서 닫힘 head S_θ를 `S_θ ≈ T_obs − F` 로 학습한다. 추론 시 guide의 물리 경향은 F + S_θ. 보정계수 c는 학습 전 학습 데이터에서 변수별 최소제곱으로 1회 결정하고 고정한다(코리올리와 기압경도력은 같은 계수 → 지균 상쇄 유지). 이유: 순간 경향과 6 h 전진차분, 5.6° 해상도의 차이로 c = 1이면 F가 경향 분산보다 커져(바람 2–23배) 닫힘이 F의 오차를 상쇄하는 데 용량을 쓴다(아래 3절).

연산 정확도: 격자→메시를 기존 가우시안 풀링으로 쓰면 기울기 오차가 7–12 %였고(풀링 가중치의 노드별 불규칙성이 미분에서 증폭), **쌍선형 보간(`grid2mesh_lin`)** 으로 바꾸면 0.1–0.5 %, 지균 잔차는 기압경도력의 0.7 %로 떨어진다. 닫힘 head와 PDE 잔차는 모두 메시 0에서 계산해 메시→격자 역보간(오차 9 %)을 제약식에 넣지 않는다.

## 3. 데이터 점검 결과 (`scripts/resolved_fraction.py`, 학습 160시각)

| 변수 | 항 | 보정계수 | 보정 후 var(T−F)/var(T) | 비고 |
|---|---|---|---|---|
| t850 | u850·∇T | 0.45 | 0.87 | c=1이면 1.06 |
| q700 | u700·∇q | 0.81 | 0.73 | 가장 물리적으로 깨끗한 항 |
| t2m | u10·∇T2m | 0.13 | 1.00 | 제외 |
| msl | 질량 연속(3층 구적) | ≈0 | 1.00 | 제외 (c=1이면 227) |
| u/v850 | (u·∇)u, f k×u+∇Φ | 0.53, 0.09 | 0.81 | 코리올리·PGF 각각은 경향 분산의 50배 |
| u/v500 | 〃 | 0.48, 0.12 | 0.71 | |
| u/v10 | 〃 | 0.63, 0.005 | 0.91 | 표면 PGF는 마찰 없이는 설명력 없음 |

해석: 이 해상도·변수 집합에서 고정 역학이 설명하는 6 h 경향은 10–30 %이고, 나머지(연직 운동, 비단열, 마찰, 해상도 이하 과정, 분석 증분)는 닫힘 S_θ가 맡는다. 그래서 S5a는 "물리가 최종 결정"이라기보다 "물리가 먼저 결정하고 학습이 나머지를 채우는" 구조이며, 효과는 S3와의 비교로 판단한다.

## 4. 진단 (`diagnose_guide.py` `layerwise_physics`)

- `resolved_fraction_6h`: 테스트에서 F만의 설명분산, `closure_fraction_6h`: F+S_θ의 설명분산
- `mse_6h_tendency`: 0 / F / F+S_θ 의 경향 MSE
- `pde_residual_24h`: 24 h 예보의 PDE 잔차 ‖(X̂−X_t)/Δt − [F+S_θ](X̄)‖² — guide 있음 / 없음 / 관측(truth) 비교. S5b가 이 값을 줄이면서 RMSE를 유지·개선하는지가 판단 기준.

## 5. 실행 계획

1. stage5a: `CLOSURE_TEND=1 RESOLVED_TERMS=adv_u,adv_t850,adv_q700,coriolis,pgf RESOLVED_COEF=runs/era5_64x32_s5a_closure/resolved_coef.json`, 나머지는 S3와 동일(섭동, 2-step rollout, guide 없음 baseline은 stage3 것 재사용). 비교: stage3(보강 guide+rollout) vs stage5a, 같은 90 초기시각.
2. stage5b: S5a의 guide 재사용(`GUIDE_FROM`), `PDE_WEIGHT=0.1`. 비교: stage5a vs stage5b, PDE 잔차 진단 포함.
3. stage5c-0: stage5a/stage5b 중 결정론 점수가 나은 run의 guide·V-Net을 그대로 두고 Stage C(residual graph diffusion, 800 step, 제약 없음)를 얹는다 — Noise 층을 추가하는 구조 변경이므로 별도 stage(대조군). 비교: 결정론 vs 앙상블(CRPS, 앙상블평균 RMSE, spread), 24 h, 45 초기시각 × 8 멤버.
4. stage5c: 같은 forecaster, Stage C에 사영 P_C + L_feas + L_mean (6절). 비교: stage5c-0 vs stage5c — 허용조건 위반(q700<0 면적비, 질량·수분 drift)이 0이 되는지, 앙상블평균 RMSE가 결정론 RMSE에 가까워지는지(평균 제약), CRPS·spread가 유지되는지. 같은 샘플의 사영 전/후 점수도 함께 기록한다(`eval.md`의 "사영 전/후" 표).

## 6. S5c의 식 (Noise 층 허용조건)

Stage C의 잔차 확산(`gpclimate/models/diffusion.py`)은 제안서 5.1절 그대로 관측 잔차 r₀ = X_obs − sg(μ)를 오염시킨 r_s를 입력으로 ε을 맞히는 L_DSM을 쓰고, 조건 c_noise = sg(X_t, Z_t(=guide P0u), μ, c_t)이다. S5c는 여기에 세 가지를 더한다.

**(1) 허용조건 사영 P_C(μ) (hard, `FeasibilityProjection`)** — 생성 상태 X^(k) = P_C(μ + R^(k)) 에 대해

| 조건 | 우리 11변수에서의 형태 | 사영 |
|---|---|---|
| q ≥ 0 | q700 ≥ 0 | 정규화 단위의 임계값 (0 − mean_q)/std_q 에서 clamp |
| M(X^(k)) = M(μ) | 전구 질량 ∝ ∫ p_s dA, p_s 대용은 msl | msl에 상수를 더해 면적평균을 μ와 일치. 질량 적분이 msl에 선형이므로 상수 이동이 **정확한 최소노름 사영** |
| W(X^(k)) = W(μ) | p_s 가중 기둥수분 ∫ msl·q700 dA (한 층이므로 σ 적분은 한 항) | q700(≥ 0으로 자른 뒤)에 양의 상수를 곱해 μ의 값과 일치 — 곱셈이므로 q ≥ 0 유지 |
| T > 0, p_s > 0 | 대기 값에서 항상 성립 | 적용 안 함 |
| Φ = H(T_v; Φ_s) | 기둥 온도가 없어 복원 불가 | z500은 예단 변수로 둠(기록) |

사영은 스칼라 채널만 바꾸므로 게이지 변환과 가환이고(테스트), 멱등이다. 평가에서는 같은 잔차 샘플의 **사영 전/후** 점수와 위반량(q700<0 면적비, |⟨msl−μ⟩| Pa, |W/W_μ−1|)을 함께 기록한다.

**(2) L_feas (soft)** — DSM 학습 중 denoiser의 x̂₀ 예측으로 만든 상태 μ + σ_R x̂₀ 에서, hard 사영이 나중에 고칠 위반을 미리 줄이도록 `ā(s)·[ ⟨relu(q_min − q)²⟩_A + ⟨msl − μ_msl⟩_A² ]`(잔차 단위)를 더한다. 사영이 분포를 덜 비틀게 하려는 것이며, 제안서의 "hard로 구현하지 않은 조건만 L_feas에" 원칙에 따라 사영과 같은 조건의 soft 판만 둔다.

**(3) L_mean — 앙상블 평균 제약 (제안서 5.3절)** — E[X^(k) | c] = μ 를 **최종(사영 후) 샘플**에서 평가한다. K개 멤버 평균의 제곱오차 ‖(1/K)Σ X^(k) − μ‖² 에는 멤버 분산/K 가 들어 있어 그대로 최소화하면 spread도 줄어든다. 그래서 서로 다른 멤버 쌍의 내적만 쓰는 비편향 추정량

  L_mean = (1 / K(K−1)) Σ_{k≠l} ⟨X^(k) − μ, X^(l) − μ⟩_W  (잔차 단위, 면적가중)

을 쓴다(조건부 독립인 멤버에 대해 E = ‖bias‖², spread 항 없음; 테스트에서 확인). 비용 때문에 K = 2 멤버 × 4 샘플링 step으로 5 step마다 한 번 기울기를 흘린다(`--noise-mean-every 5 --noise-mean-members 2 --noise-mean-steps 4`). 제안서의 경고대로 생성된 한 시점의 상태에 PDE 잔차를 요구하지는 않는다.

손실: L_Noise = L_DSM + λ_feas L_feas + λ_mean L_mean, 기본 λ_feas = λ_mean = 1. stage5c-0은 λ = 0, 사영 없음(S1의 Stage C와 동일).

S1의 Stage C 결과(`runs/era5_64x32`, 800 step)에서 앙상블평균 RMSE(z500 287)가 결정론 RMSE(249)보다 나빴고 CRPS는 독립 가우시안 잡음 baseline과 같았다. 이 "잔차 생성기가 평균을 끌고 간다"는 증상이 바로 L_mean이 겨냥하는 것이므로, S5c의 판단 기준은 (i) 위반량 0, (ii) 앙상블평균 RMSE → 결정론 RMSE, (iii) CRPS ≤ stage5c-0, spread 유지이다.
