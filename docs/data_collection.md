# AC 완속 충전기 실험 데이터 수집 프로토콜

> **목적**: "I_actual = f(SoC, PWM_cap)" 매핑 학습 및 PWM 스케줄링 기반 피크 저감 연구 (MS thesis) 을 위한 실험 데이터를 체계적으로 수집한다.

## 0. 배경 — 필드 현실

- SAE J1772 / IEC 61851 파일럿 공식: `I_max(A) = duty_cycle(%) × 0.6`
- 한국 AC 완속 충전기 **하드웨어 상한**: 일반적으로 11 kW (단상 50 A ≈ PWM 83%)
- **필드 기본 운용값**: 7 kW @ 220 V ≈ 31.8 A ≈ **PWM 53%** (계약·그리드 부하 고려)
- PWM 하한: **약 10% (6 A)** — 그 이하에선 EV가 세션 거부
- 연구 대상 PWM 범위: **[15%, 83%]** — 2 kW(저감 하한) ~ 11 kW(하드웨어 상한)

## 1. PWM × SoC 샘플링 격자

### 🔑 수집 전략 — Full vs Taper 분리

**오늘 PWM 52% 세션에서 확인된 사실**: SoC 22–89% 구간 binding_ratio = 0.983 ± 0.001 (사실상 상수). 즉 저·중 SoC는 학습할 정보가 거의 없고 **변동은 고 SoC에 집중**.

이에 따라 세션을 두 유형으로 나눠 시간을 대폭 절감:

| 유형 | SoC 범위 | 용도 | 세션당 시간 |
|---|---|---|---|
| **Full session** | ≤10% → 100% | ETA 라벨, flat 구간 재확인, 저PWM flat 가정 검증 | 6~27h |
| **Taper session** | 85% → 100% | f(SoC, PWM) 고해상도 매핑 (핵심 데이터) | 0.5~3h |

### Phase A — 단일 차량 f(SoC, PWM) 격자 (최우선)

**4-레벨 전력 기반 설계** — 물리적으로 의미 있는 regime 4개를 대표:

| 레벨 | 전력 | I_cap | **PWM** | OBC regime | 역할 |
|---|---|---|---|---|---|
| L11 | 11 kW | 50.0 A | **83%** | **over-OBC** (I_cap > OBC rating) | 하드웨어 상한 |
| L7 | 7 kW | 31.8 A | **53%** | **at-OBC** (전환점, 오늘 세션) | 필드 기본 |
| L4 | 4 kW | 18.2 A | **30%** | mid (regime 전환 중간) | 감량 중간값 |
| L2 | 2 kW | 9.1 A | **15%** | **under-OBC** (PWM-bound) | 저감 하한 |

> 3레벨(L11/L7/L2)만으로도 3-regime 대표는 가능하지만 L4 추가 시 XGBoost가 전환 경계를 보간할 포인트 확보. **4레벨 권장**.

동일 차량 1대 고정. **모든 세션 SoC 100% 완충 후 30분 추가 유지**.

**A-Full (full-range 세션, 4개 — regime별 기본 곡선)**

| 세션 | 전력 | PWM | SoC 범위 | 예상 시간 | 목표 |
|---|---|---|---|---|---|
| A-L11-F | 11 kW | 83% | ≤10 → 100% | ~6h | over-OBC regime, OBC 상한 포착 |
| A-L7-F | 7 kW | 53% | ≤10 → 100% | ~9h | at-OBC 전환점 (오늘 놓친 taper 끝단) |
| A-L4-F | 4 kW | 30% | ≤10 → 100% | ~15h | mid regime, flat 가정 검증 |
| A-L2-F | 2 kW | 15% | ≤10 → 100% | ~27h | under-OBC, 저PWM flat 가정 검증 |

**A-Taper (85→100%, 각 레벨 2회 — 고해상도 매핑)**

| 세션 | 전력 | PWM | 예상 시간/회 | 반복 |
|---|---|---|---|---|
| A-L11-T | 11 kW | 83% | ~30분 | 2 |
| A-L7-T | 7 kW | 53% | ~45분 | 2 |
| A-L4-T | 4 kW | 30% | ~1.5h | 2 |
| A-L2-T | 2 kW | 15% | ~3h | 2 |

**하드웨어 현황**: EVSE에 **릴레이 1개**만 장착된 상태 → 현재 수집 가능 레벨은 **L7 / L4 / L2** (세 개). L11은 **릴레이 2개로 업그레이드 후** 별도 수집. 따라서 Phase A는 두 단계로 분리:

- **Phase A-Now (릴레이 1개)**: L7 / L4 / L2 = Full 3 + Taper 6 = **9세션, ~62h**
- **Phase A-Later (릴레이 2개 업그레이드 후)**: L11 Full 1 + Taper 2 = **3세션, ~7h**

수집 우선순위 (★★★★★ — 즉시, 현재 하드웨어):
1. **A-L7-F** — 오늘 놓친 taper 끝단 확보 (at-OBC)
2. **A-L2-F** — under-OBC flat 가정 검증 (가장 불확실)
3. **A-L4-F** — mid regime 검증 + 격자 중앙점

3개 Full 확보 후 flat 가정 확인되면 **A-Taper 6세션**으로 전환.

### Phase B — 차종 다양성

| 세션 | PWM | 차량 | SoC 범위 |
|---|---|---|---|
| B1 | 52% | Car_2 (다른 모델) | 10 → 100% |
| B2 | 52% | Car_3 | 10 → 100% |
| B3 | 52% | Car_4 | 10 → 100% |

추가로 Car_2, Car_3에서 PWM 20% 1세션씩 확보 시 generalization 근거 강화.

### ~~Phase C — 병렬 세션~~ (제외)

**제외 사유**: 다대 동시 충전 시 피크 합이 차단기 허용치를 초과하면 자동 트립 → 실측 의미 없음. 기여③(PWM 스케줄링) 평가는 **P0/P1/P2 단일 세션 로그를 오버레이한 시뮬레이션**으로 대체. AC Level 2에서 차량 OBC 간 전기적 독립성이 성립하므로 시뮬레이션 유효성 확보 가능. (코드 TODO 참조)

### Phase D — 계절 비교 (환경 변수)

챔버가 없어 외기온 직접 통제 불가 → **계절을 활용한 자연 A/B 실험**으로 대체:

| 세션 | 계절 | 시점 | 레벨 | 역할 |
|---|---|---|---|---|
| _baseline_ | 봄 | 2026-04 (P0 수집 중) | L7 / L2 | P0의 A-L7-F / A-L2-F가 봄 baseline |
| D-Summer-L7-F | 여름 (고온) | 2026-06 ~ 08 | L7 | 필수 |
| D-Winter-L7-F | 겨울 (저온) | 2026-12 ~ 2027-02 | L7 | 필수 |
| D-Summer-L2-F | 여름 | 2026-06 ~ 08 | L2 | 옵션 (regime × 계절 교차) |
| D-Winter-L2-F | 겨울 | 2026-12 ~ 2027-02 | L2 | 옵션 |

## 2. 환경 통제 변수 (수집 시 고정 또는 기록)

| 변수 | 전략 | 이유 |
|---|---|---|
| 차량 | Phase A는 **1대 고정**, B에서 다양화 | f(SoC, PWM) 순수 추출 |
| 외기온도 | 매 세션 기록, 극단 피함 (5~30°C) | OBC 효율·배터리 저항 영향 |
| 배터리 초기 온도 | BMS 읽기 가능 시 기록 | 저온 시 CC-CV 지연 |
| 시작 SoC | ≤10% 목표 | 전 SoC 범위 확보 |
| 종료 SoC | **100% 강제** | taper 구간 확보 |
| 주행 직후 여부 | 기록 (preconditioned) | 초기 배터리 온도 영향 |

## 3. 수집 기술 체크리스트

매 세션 시작 전:
- [ ] PWM 설정 변경 적용 확인 (OCPP `SetChargingProfile` 또는 EVSE UI)
- [ ] MongoDB 로그 수집 활성 확인
- [ ] 메타 YAML 파일 작성 시작

매 세션 중:
- [ ] **MeterValues 뿐 아니라** StartTransaction / StopTransaction / StatusNotification / BootNotification **모두** 수집
- [ ] 세션 중단 없이 100% 도달까지 유지
- [ ] SoC 98% 이후 **최소 30분 추가** (end-of-charge taper 확보)

세션 종료 후:
- [ ] JSON 파일이 `]` 로 정상 종료되는지 확인
- [ ] 메타 YAML 파일 완성
- [ ] 파일명 컨벤션 준수

## 4. 파일명 / 세션 ID 컨벤션

```
data/raw/YYMMDD-{phase}-PWM{pp}-{nn}.json
data/raw/YYMMDD-{phase}-PWM{pp}-{nn}.meta.yaml

예시:
  260420-A-PWM52-01.json       # Phase A, PWM 52%, 1회차
  260420-A-PWM52-01.meta.yaml
  260421-A-PWM20-01.json
  260505-B-PWM52-01.json       # Phase B (다른 차)
  260510-C-PWM52-02.json       # Phase C 병렬 중 2번 커넥터
```

## 5. 세션 메타데이터 템플릿

```yaml
session_id: 260420-A-PWM52-01
phase: A
car:
  model: "Tesla Model 3 LR"
  vin_last4: "1234"
  obc_spec_kw: 11.5
charger:
  id: "001QATLABS016"
  evse_max_a: 32
pwm_setting_pct: 52
pwm_setting_a_calc: 31.2          # = pwm_setting_pct × 0.6
start:
  time_kst: "2026-04-20T10:00:00+09:00"
  soc_pct: 8
  pack_temp_c: 22                 # null 허용
stop:
  time_kst: "2026-04-20T16:30:00+09:00"
  soc_pct: 100
  reason: "fully_charged"         # fully_charged / user_unplug / error / target_reached
environment:
  ambient_temp_c: 15
  weather: "cloudy"
  preconditioned: false
parallel_session_ids: []          # Phase C일 때 동시 세션 ID들
notes: ""
```

## 6. 데이터 예산 요약

| Phase | 세션 수 | 누적 시간 | 비고 |
|---|---|---|---|
| **A-Now** (L7/L4/L2 Full 3 + Taper 6) | 9 | ~62h | 현재 하드웨어 |
| B (차종 다양성) | 5 | ~33h | L7 Full 위주 |
| ~~C (병렬)~~ | 0 | — | **제외**: 차단기 트립으로 실측 불가, 시뮬레이션으로 대체 |
| D (계절 비교, 필수) | 2 | ~18h | 여름·겨울 L7 각 1회 |
| D (계절 비교, 옵션) | +2 | +54h | 여름·겨울 L2 각 1회 |
| **현재 필수 합계** | **16** | **~113h** | 릴레이 1개 |
| **현재 옵션 포함** | **18** | **~167h** | |
| A-Later (L11, 릴레이 업그레이드 후) | +4 | +13h | A-L11 Full/Taper + B-Car2-L11 |

실 실험일: 주말 집중 기준 약 6~8주 (Full을 주말, Taper를 평일에 배치).

## 7. 바로 실행 가능한 "첫 3세션" (릴레이 1개)

우선순위 ★★★★★ — regime별 기본 곡선 확보:

| 순서 | 세션 | 전력 | PWM | SoC | 예상 | 핵심 목표 |
|---|---|---|---|---|---|---|
| 1 | **A-L7-F** | 7 kW | 53% | ≤10→100% | ~9h | at-OBC taper 끝단 (오늘 놓침) |
| 2 | **A-L2-F** | 2 kW | 15% | ≤10→100% | ~27h | under-OBC flat 검증 |
| 3 | **A-L4-F** | 4 kW | 30% | ≤10→100% | ~15h | mid regime 검증 |

공통 절차:
1. SoC ≤10%까지 주행 또는 방전 대기
2. PWM 설정 변경 후 적용 확인
3. 메타 YAML 작성, 수집 시작
4. **100% 도달 후 30분 추가 유지** 후 종료
5. 파일 export 시 `]` 닫힘 확인
6. 메타 YAML에 실제 종료 시각·SoC 기록

**분기 규칙**:
- A-L2-F 결과에서 저 SoC flat (binding_ratio 편차 ≤0.005) → A-Taper 6세션 진행
- flat 가정 깨짐 → Taper 윈도우 70→100%로 확장 후 재설계
- 릴레이 2개 업그레이드 완료 → A-L11-F 추가 수집 (3세션)
- A-L11-F에서 over-OBC 확인 (binding_ratio <0.9) → regime 분리 논문 핵심 그림으로 채택
