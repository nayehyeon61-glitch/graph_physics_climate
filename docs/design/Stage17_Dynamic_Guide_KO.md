# stage17 — dynamic guide (6시간 경로 위에서 다시 계산되는 물리 guide)

작성 2026-11-07 19:55 KST. 감자 요청: "우선 추천방향 진행", "아직 논문으로 하기엔 역시 guide 가 약해요" (19:31 KST).

## 문제: 지금 guide는 왜 약한가

| 관찰 (stage9b) | 뜻 |
|---|---|
| guide 이득 +4.26 / +4.19 / +3.50 / +2.37 / +1.75 % (24–120 h) | 리드가 길수록 줄어듦 |
| guide 특징의 6 h 경향 probe R² 0.336 (상태만 0.086) | guide는 **6시간** 경향을 잘 앎 |
| guide는 X_t에서 한 번만 계산되고 V-Net은 24 h를 한 번에 감 | 6 h 정보가 24 h 한 걸음에 묻힘 |
| 비교 짝은 guide·고정 연산자·PIGT 융합층이 모두 없음 | +4 %는 "물리 통합 모듈 전체"의 몫 |

가설: guide가 아는 시간 척도(6 h)와 모델이 쓰는 시간 척도(24 h)가 어긋나서 guide가 약하다. guide를 6시간
경로의 매 상태에서 다시 계산하면(dynamic guide), guide가 예보 경로 전체에 관여하고 guide의 몫이 커진다.

## 구조

24 h 리드를 6 h sub-step 4개로 적분한다(graph neural ODE, Euler):

    X_{k+1} = X_k + (1/4) · δ · V-Net(X_k, P(X_k), O(X_k), c_k, prev, col_{k=0})          (stage17a)
                  + γ ⊙ T_guide(X_k) · s_tend                                             (stage17b 추가)

- **P(X_k), O(X_k)**: 학습된 guide 특징과 고정 물리 연산자를 매 경로 상태에서 다시 계산한다.
- **c_k**: 태양 강제력(toa_t, toa_t1)을 sub-step 시각으로 옮긴다. 삽입 일사 근사는 경도 방향으로 시간당 15°
  움직이므로 6 h = 90° = 64격자에서 16칸 회전이 정확하다(적위 변화 하루 0.4° 미만만 무시). 고정 조건(지형, 해륙,
  코리올리)은 그대로.
- **prev**(stage14 이전 변화)는 모든 sub-step에, **기둥 입력**(stage15a)은 관측 시각인 sub-step 0에만 준다.
- **경로 감독**: X_{t+6h}, X_{t+12h}, X_{t+18h} 경로 상태를 ERA5와 맞춘다(가중치 1.0, 24 h 손실과 같은 delta 단위).
  stage15b의 보조 head와 달리 **실제 예보 경로**를 감독하므로 guide가 보는 상태가 물리적으로 의미 있는 6 h 상태가 된다.
- **stage17b**: guide 자신의 6 h 경향 T_guide를 drift로 더하고, 변수별 이득 γ를 0에서 학습한다. 학습된 γ가
  "동역학이 guide 경향을 얼마나 쓰는지"를 직접 보여 주고, guide drift만 적분한 예보(physics_only 행)도 나온다.
  stage8b는 12 h sub-step에서 6 h 경향을 2배로 늘려 써서 발산했는데, 여기서는 sub-step이 정확히 6 h다.

## 실험 설계 (CPU 예산)

ODE 4 sub-step은 한 step이 delta보다 약 5.4배 비싸서(stage8 측정) 9000 step 처음부터 학습은 CPU로 이틀 넘게 걸린다.
그래서 **짝 지은 미세조정**으로 비교한다. 모두 stage9b 가중치에서 시작해 같은 데이터·lr(3e-4, cosine)·rollout 2로
1000 step을 더 학습한다.

| run | 적분 | guide | 비교 대상 |
|---|---|---|---|
| stage17-ctl `runs/era5_64x32_s17_ctl` | delta (stage9b 그대로) | 정적 | 기준 |
| stage17a `runs/era5_64x32_s17a_dyn` | ODE 4×6 h + 경로 감독 + sub-step 강제력 | 동적 | stage17-ctl |
| stage17b `runs/era5_64x32_s17b_drift` | stage17a + guide drift(γ, 0 시작) | 동적 + drift | stage17a |

각 run에 guide 없는 짝도 같은 방식으로 미세조정한다(stage17b는 guide가 없으면 drift도 없으므로 stage17a의 짝을 씀).

## 판정

1. 정확도: decide_adoption(11변수 평균, 5개 리드 중 3개 이상 개선, 평균 +0.5 % 이상), 2019 + 2020/2022, ClimODE 열.
2. **guide가 강해졌는가** (이번 단계의 핵심): 같은 적분기 안에서 guide 있음 vs 없음의 이득이 stage17-ctl의 정적
   guide 이득보다 큰지, 특히 72–120 h에서 줄어들지 않는지. P=0·교체 진단, probe, PIGT 어텐션 비중도 함께.
3. stage17b: 학습된 γ(변수별), physics_only 예보의 기술(지속성 대비).

## 비용과 일정

stage16(문헌 분할)과 그 ClimODE CRPS 평가가 끝난 뒤(11-08 오전) 시작한다. 예상: stage17-ctl 약 1.7 h, stage17a
약 9 h(guide 있는 ODE 7 h + 짝 2 h), 평가 약 1 h → stage17a 결과 11-08 저녁, stage17b(약 8 h)는 11-09 새벽.

## 코드

- `gpclimate/models/dynamics_vnet.py`: `VNetConfig.ode_path_forcing`, `ode_drift_gain`; `LieGuidedForecaster.sub_cond`,
  `integrate(..., prev, col)`(경로를 `last_path`에 보관), `inter_forecasts`(ODE면 경로 반환), drift 이득.
- `scripts/train_eval.py`: `--ode-path-forcing`, `--ode-drift-gain`, `--init-vnet`(짝 지은 미세조정), ODE에서
  `--inter-times 3` = 경로 감독, 검증 로그에 drift 이득.
- 테스트 `tests/test_dynamic_guide.py`(6개): 강제력 회전 = 실제 일사 근사, prev/기둥 입력 전달, 경로 감독, 이득 0 시작,
  delta 가중치 이어받기, 짝 로딩, 파이프라인(미세조정 → 평가 → 진단).
