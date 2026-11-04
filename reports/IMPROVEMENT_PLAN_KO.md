# 개선점 조사 보고서와 실행 계획 (stage12, 2026-11-04 09:00 KST)

감자 요청(2026-11-04 08:28 KST): "개선점을 조사해서 수행해주세요". 기준 모델은 stage3(`runs/era5_64x32_physops_roll`, 게이트 융합 + 물리 연산자 guide + 예보상태 섭동 + 2-step rollout). **부호 규약: 모든 수치는 개선율 100·(1 − 오차/기준 오차), + = 개선(좋음), − = 악화(나쁨).** 진단 원자료: `reports/error_diagnostics_era5_64x32_physops_roll.md`(`scripts/diagnose_errors.py`, 2019년 90개 초기시각 × 5 lead).

## 1. 요약

1. ClimODE에 뒤지는 곳은 **24–48 h의 중위도·극지방 종관규모**뿐이다(z500 24 h: 열대 +29.6 %, 중위도 −22.2 %, 극 −21.8 %). 72 h 이후와 t2m은 모든 위도대에서 우리가 앞선다.
2. 그 오차는 편향이 아니라 **종관규모(동서 파수 8–15) 정보 손실**이다: 24 h에 관측 파워의 73 %(ClimODE 82 %), 120 h에 43 %(80 %)만 남는다. 계통 편향 비중은 0.17–0.24로 작다(ClimODE 0.36–0.76).
3. 학습은 여전히 **step 제한**이다: 검증 MSE가 마지막 500 step에서 +5.5 % 더 내려가고(stage7의 10년 데이터는 같은 step에서 곡선이 동일), 모델은 ClimODE의 1/3 크기(V-Net 0.55 M + guide 0.28 M vs 2.75 M), batch 4, EMA·가중치 감쇠 없음.
4. 따라서 개선안은 (a) 최적화 품질(EMA), (b) V-Net 용량(폭), (c) step 예산(stage7b, 이미 예약), (d) guide를 예보 목적에 맞춰 미세조정, (e) 장기 lead 감쇠 완화(rollout 커리큘럼) 순으로 **한 stage에 한 변경**씩 실행한다. 각 stage는 stage3과 같은 데이터·step으로, ClimODE 열과 함께 비교한다.
5. 코드(`scripts/train_eval.py` Stage B 옵션 `--ema`, `--rollout-schedule`, `--guide-lr`; 기본값 꺼짐, 이전 recipe 그대로 재현)와 스모크 테스트(`tests/test_train_eval_smoke.py`, 합성 12×24 데이터로 4개 stage 실행)는 완료, 전체 테스트 47개 통과. 체인은 stage11a 직후 stage12a → stage12b, stage11b 뒤 stage12c → stage12d로 예약했다(4절).

## 2. 증거 (stage3 기준)

### 2.1 ClimODE 대비 lead별 개선율 (+ = stage3이 좋음)

| 변수 | 24 h | 48 h | 72 h | 96 h | 120 h |
|---|---|---|---|---|---|
| z500 | −17.1 % | −6.8 % | −1.1 % | +4.0 % | +8.6 % |
| t850 | −6.9 % | +4.9 % | +8.9 % | +12.2 % | +15.8 % |
| t2m | +42.2 % | +57.8 % | +47.4 % | +45.9 % | +46.2 % |
| u10 | −7.4 % | −5.7 % | −1.6 % | +1.9 % | +4.9 % |
| v10 | −10.2 % | −3.0 % | +4.4 % | +9.1 % | +12.8 % |

ClimODE 주의: 10년 학습(2006–2015), 2.75 M 매개변수, 저자 프로토콜(train-mode BatchNorm). 우리: 2016–2017 2년, 0.83 M.

### 2.2 어디서 지는가 — 위도대 (24 h, + = stage3이 좋음)

| 변수 | 열대 (|φ|<20°) | 중위도 (20–60°) | 극 (>60°) |
|---|---|---|---|
| z500 24 h | +29.6 % | −22.2 % | −21.8 % |
| t850 24 h | +13.2 % | −10.1 % | −14.4 % |
| z500 120 h | +17.0 % | +8.6 % | +7.7 % |

열대와 장기 lead는 우리가 앞서고, 격차는 **중위도·극의 초기 1–2일 종관규모 발달**에 집중된다. 오차 증가율(RMSE(120 h)/RMSE(24 h))은 우리 2.88, ClimODE 3.69(z500) — 우리는 출발 오차가 크고 증가가 느리다(평활 예보의 전형).

### 2.3 왜 지는가 — 편향 아님, 종관규모 평활화

- **편향**: 사례평균 오차장 RMS / RMSE = 0.17–0.24(모든 변수·lead), 전구평균 오차 비중 ≈ 0. ClimODE는 0.36–0.76(t2m 0.76). 편향 보정류 개선안은 우리에게 효과가 없다.
- **진폭비**(예보 아노말리 std / 관측): 24 h 바람 0.88, q700 0.88; 120 h msl 0.74, v850 0.63, u850 0.69. z500은 0.97 → 0.90. lead가 길수록 감쇠한다(MSE 학습의 평균화).
- **동서 스펙트럼**(30–60°, 예보/관측 파워): z500 24 h k1-3 0.97, k4-7 0.95, **k8-15 0.73**(ClimODE 0.82); 120 h k4-7 0.67(ClimODE 0.96), k8-15 0.43(0.80). u850은 24 h에 이미 k8-15 0.48. 즉 메시 2562 노드(≈480 km)에서 파장 2,500–5,000 km 대역이 첫 step부터 깎인다 — 한 level당 depth 1인 V-Net과 grid↔mesh 보간의 용량·수용영역 문제.
- **격자규모 잡음**: z500 k16-32는 120 h에 1.61(no_guide 1.87, ClimODE 0.43) — rollout 중 최소 규모 잡음이 자란다. 다른 변수(t850 0.39, u850 0.22)는 반대로 과평활. 후순위 후보(스펙트럼 벌칙)의 근거.

### 2.4 학습 예산·용량·절차

- Stage B 검증 MSE: 0.325 → 0.1331(250) → 0.1063(500) → 0.0964(750) → 0.0891(1000) → 0.0852(1250) → 0.0842(1499); 마지막 499 step 개선 +5.5 %(guide 없음 +5.6 %). stage7(10년, 같은 step)은 step별 곡선이 동일 → **데이터가 아니라 step이 병목**. stage7b(4500 step)가 이를 직접 검증한다.
- rollout 손실의 대가: stage2(rollout 없음) 1-step 검증 MSE 0.0787 vs stage3 0.0842(−7 %)이지만 72–120 h는 stage3이 낫다 → 24 h와 장기 lead의 trade-off가 있으므로 커리큘럼(늦게, 짧게)이 합리적.
- 매개변수: V-Net 554,665(cs 48 / cv 12, depth 1/level) + guide 275,817; 폭 64/16이면 825,505(+33 % step 시간, 사전 측정). batch 4, lr 1e-3 cosine, EMA·가중치 감쇠 없음 → 마지막 가중치가 잡음에 민감.
- guide 기여(stage3 vs guide 없음, + = guide가 좋음): z500 +1.3/+1.0/+0.7 %(24/72/120 h), t2m +4.7/+3.1/+2.5 %, u850 +1.9/+0.8/+0.2 %; P=0 열화 +102 %(진단값). guide는 6 h 경향 목적으로만 학습되고 Stage B에서 동결되므로 예보 목적과 정렬되지 않았다(guide 단독 probe R² 0.335).

## 3. 개선 후보와 우선순위

| 순위 | stage | 변경(한 가지) | 근거(2절) | 비용 | 기대 |
|---|---|---|---|---|---|
| 1 | **stage12a** | Stage B 가중치 EMA(감쇠 0.995, 검증·저장은 EMA 가중치) | 2.4 step 제한, batch 4 잡음 | 0 | 전 lead +1–3 % |
| 2 | **stage12b** | V-Net 폭 48/12 → 64/16 | 2.3 종관규모 용량, 2.4 크기 1/3 | +33 % step 시간 | 24–48 h 중위도 개선 |
| 3 | stage7b(예약됨) | Stage B 1500 → 4500 step(10년 데이터) | 2.4 step 제한 | 3× | 전 lead |
| 4 | **stage12c** | Stage B에서 guide 미세조정(lr 1e-4, V-Net 1e-3) | 2.4 guide-예보 목적 불일치 | +10 % 내외 | guide 여유폭↑, 24 h |
| 5 | **stage12d** | rollout 커리큘럼 500:2, 1000:3 | 2.3 장기 감쇠, 2.4 trade-off | 마지막 1/3 +50 % | 72–120 h +, 24 h − 가능 |

채택 기준(기존 규칙): 11변수 평균 개선율이 5 lead 중 3개 이상 +이고 P=0 열화가 stage3의 +102 % 아래로 무너지지 않을 것. 채택된 변경은 stage9 통합 구조에 들어간다.

후순위(이번에 예약하지 않음, 결과에 따라 stage12e–): 최상위 메시 depth 1 → 2(종관규모 수용영역; 폭보다 비싸 폭 결과를 보고 결정), 격자규모 잡음 억제(k16-32 스펙트럼·라플라시안 벌칙), batch 8, 가중치 감쇠, lead 가중 손실, 메시 (5,4,3)(GPU 필요), 10년 데이터 + 3× step 결합(stage7b 결과가 결정), 상위 stage 결합 = stage9. 기각된 방향: ODE 예측층(stage8a/b), fcn21 21채널 출력(stage10a; stage10b 대기).

## 4. 실행 순서와 예상 시각 (KST, CPU 4코어 직렬)

| 순서 | stage | 체인 | 예상 완료 |
|---|---|---|---|
| 1 | stage11a(진행 중, Stage B 575/1500) | `runs/era5_64x32_s11a_curv/chain_s11.sh` | 11-04 10:35 |
| 2 | stage12a EMA | `runs/era5_64x32_s12a_ema/chain_s12a.sh` | 11-04 13:15 |
| 3 | stage12b 폭 64/16 | `runs/era5_64x32_s12b_width64/chain_s12b.sh` | 11-04 16:30 |
| 4 | stage7b 4500 step | `runs/era5_64x32_s7b_steps3x/chain_s7b.sh`(stage12b 뒤로 이동) | 11-04 23:00 |
| 5 | stage10b 손실 가중 | `runs/era5_64x32_s10b_fcn21_w/chain_s10b.sh` | 11-05 01:30 |
| 6 | stage11b 심플리셜 | `runs/era5_64x32_s11b_simp/chain_s11b.sh`(stage11a 결과 후 생성) | 11-05 04:30 |
| 7 | stage12c guide 미세조정 | `runs/era5_64x32_s12c_guideft/chain_s12c.sh` | 11-05 06:10 |
| 8 | stage12d rollout 커리큘럼 | `runs/era5_64x32_s12d_roll3/chain_s12d.sh` | 11-05 09:15 |
| 9 | stage13a 구면 neural operator 층(감자 제안 09:28 KST) | `runs/era5_64x32_s13a_sfno/chain_s13a.sh` | 11-05 12:30 |
| 10 | stage13b + 스펙트럼 에너지 손실 | `runs/era5_64x32_s13b_sfno_spec/chain_s13b.sh` | 11-05 15:45 |

stage12a/12b/12d는 guide 없는 baseline도 같은 절차로 다시 학습한다(절차가 바뀌면 baseline도 바꾼다는 규칙). stage12c는 baseline이 바뀌지 않으므로 stage3의 것을 복사한다. 모든 stage12는 stage3의 guide(`guide.pt`)·정규화기를 재사용하고 Stage A를 생략한다(Stage B만 다름).

## 5. 구현 메모

- `scripts/train_eval.py` Stage B: `--ema d`(EMA 복사본 `copy.deepcopy(model)`, 워밍업 d_t = min(d, (1+t)/(10+t)), 검증과 `vnet.pt` 저장에 EMA 사용), `--rollout-schedule "500:2,1000:3"`(`parse_rollout_schedule`; `--rollout-steps/--rollout-start`를 덮어씀), `--guide-lr`(guide 매개변수 그룹을 별도 base lr로 cosine 스케줄, 개선 시 `guide_finetuned.pt` 저장; `Run.load_guide(m, finetuned=True)`가 추론에서 우선 사용, Stage B 시작은 `guide.pt`). 모두 기본값 꺼짐.
- `scripts/diagnose_errors.py`: 위도대 RMSE, 편향, 진폭비, 동서 스펙트럼, 오차 증가율, 학습 곡선, ClimODE 정렬. 첫 실행은 npz 지연 슬라이싱 때문에 메모리 9.4 GB 초과로 중단 → 배열을 한 번만 적재하도록 수정.
- 비용 교훈: 학습 중에 `nice -n 19`로 돌린 진단(2 thread)·테스트(1 thread)만으로도 Stage B 속도가 4.6 → 17 s/step로 3.7배 느려졌다(OpenMP 과구독). 앞으로 부수 작업은 학습 Stage 사이에만.
