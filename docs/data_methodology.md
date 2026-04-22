# Phase B Data Methodology

> **Version**: 1.0 (2026-04-22)
> **Status**: Final for thesis Ch.3 / Appendix reference. Reproducible from scripts listed in §10.
> **Scope**: AC Level-2 charger fleet, 5.5-month window, OCPP server-log collection.

This document captures the pipeline that turned a 118.9 million–document MongoDB OCPP collection into a 58,175-session, ML-ready parquet dataset for AC slow-charger departure-time prediction and LP peak-shaving. It is intended as the authoritative record for thesis Ch.3 (Data) and as a reproducibility appendix.

---

## 1. Executive summary

| Axis | Value |
|---|---|
| Source collection | MongoDB `logs.PLATFORM-OCPP` — 118,932,913 documents |
| Collection window | 2025-11-08 00:35 UTC → 2026-04-22 03:46 UTC (≈ 5.5 months) |
| Distinct chargers (all) | ~700 (incl. null / non-conforming IDs) |
| AC chargers confirmed | **492** (`ELA*` / `E01AS*` model families) |
| Raw sessions aggregated | **69,224** |
| Clean sessions | **58,175** (84.0 %) |
| ML-usable (MV-present) clean | 45,380 (78.0 %) |
| Primary deliverable | `data/phase_b/session_dataset_clean_v2.parquet` |

The dataset is now the input for the XGBoost/LSTM departure-time track and the CVXPY LP simulation track of the thesis pipeline (see `HANDOFF_ModelPipeline_v2.md`).

---

## 2. Source & scope

### 2.1 Collection profile

| Item | Value |
|---|---|
| Collection | `logs.PLATFORM-OCPP` |
| Total documents | 118,932,913 |
| Oldest timestamp | 2025-11-08 00:35:00 UTC |
| Newest timestamp | 2026-04-22 03:46:19 UTC (at cut-off) |
| Indexes | `_id`, `timestamp_-1`, `meta.chargerId`, `(meta.chargerId, timestamp_-1)` |

The absence of an index on `meta.action` dictated a timestamp-first query strategy for all subsequent aggregations.

### 2.2 AC charger classification

An AC-only scope is mandatory: DC fast chargers (30 min sessions, 150–350 A taper) and AC slow chargers (2–12 h sessions, 6–50 A PWM-bound) have structurally different departure and current profiles. Mixing them breaks both the ML label distribution and the η linearity assumption underlying the LP.

**Classification rule** (AC allowlist, not DC blocklist):

```python
is_ac(model) := re.match(r"^ELA", model) or re.match(r"^E01AS", model)
```

Source: BootNotification `meta.payload.chargePointModel`, aggregated across the full 5.5-month window to capture chargers that booted at any time.

**AC model distribution (492 chargers)**:

| Model | Chargers | Interpretation |
|---|---|---|
| `ELA007C01` | 264 | 7 kW single-phase, series A |
| `E01AS007K10KR0101` | 147 | 7 kW single-phase, series D (PLC-embedded) |
| `ELA007C02` | 30 | 7 kW series B |
| `ELA007C02R` | 28 | 7 kW series B variant |
| `ELA007C03` | 17 | 7 kW series C |
| `ELA011C02` | 4 | 11 kW single-phase |
| `ELA011C01` | 1 | 11 kW single-phase |
| `E01AS007K10JP0002` | 1 | 7 kW series E (Japan region) |

**All 492 chargers are single-phase AC.** 11 kW variants (5 chargers) are a negligible minority; the field defaults assume 7 kW (31.2 A @ 220 V).

**Excluded model families**: `E01DS*` (DC 100–200 kW), `E01DM*` (DC 50 kW), `FC-SS-*` (DC fast), `WEV-D*` (DC). Raw counts totalled ~140 DC chargers; none reached the AC pipeline.

---

## 3. Export pipeline

### 3.1 Design decisions

| Decision | Rationale |
|---|---|
| Keep 3 actions: `StartTransaction`, `StopTransaction`, `MeterValues` | Minimal set for session timing, energy, and current profiles. Heartbeat / StatusNotification omitted as high-volume noise. |
| One file per charger (`{chargerId}.json`) | Enables per-charger checkpoint/resume, keeps individual files pandas-friendly (median 5.74 MB). |
| Charger list locked to the 492 AC allowlist | Prevents accidental DC ingestion on re-runs. |
| BSON extended JSON in **relaxed** mode | `{"$date": "2025-11-08T..."}` ISO strings parse directly in `pandas.Timestamp`; compatible with the existing `tinkaton.loader.load_ocpp_logs`. |

### 3.2 RECV / SEND pair structure

Each OCPP transaction in this collection appears **twice** — once as `serverRecvType = RECV` (charger → server request) and once as `serverRecvType = SEND` (server → charger response). Ignoring this duplicates sample counts.

Linking rule:

- `StartTransaction` **RECV** payload: `connectorId`, `idTag`, `meterStart` (Wh), `timestamp`, `reservationId`. **Does not carry `transactionId`** — it is assigned by the server in the response.
- `StartTransaction` **SEND** payload: `idTagInfo.status`, **`transactionId`**.
- Both messages share `meta.messageId` → join to recover `transactionId` for the submission record.

This is implemented in `tinkaton.loader.transaction_events_to_dataframe`.

`StopTransaction` RECV already carries `transactionId`, `meterStop`, `timestamp`, `reason` and does not require pairing.

`MeterValues` SEND has an empty payload (`CALLRESULT` ACK) and must be filtered out to avoid zero-duplicate rows.

### 3.3 Export run (single full execution)

Run command (2026-04-22):

```bash
uv run python scripts/export_phase_b.py
```

Output:

| Metric | Value |
|---|---|
| AC chargers queried | 492 |
| Non-empty charger files written | 472 |
| Empty / no-data skipped | 20 |
| Total documents | 5,691,016 |
| Total payload | 3,637.7 MB |
| Runtime | 257.2 min |
| Failures | 0 (`_errors.log` absent) |
| Largest single charger | 226.8 MB, 338 K docs |

Checkpoint-resume via `data/raw/phase_b/_checkpoint.json`; audit manifest at `data/raw/phase_b/_charger_manifest.csv`.

---

## 4. Session aggregation

### 4.1 Keying and authority

Sessions are keyed by `(charger_id, transaction_id)`. For each key the pipeline fuses two evidence sources:

| Field | Authority |
|---|---|
| `arrival_ts` | `StartTransaction.payload.timestamp` (RECV) — falls back to first MeterValue only if Start event missing |
| `plug_out_ts` | `StopTransaction.payload.timestamp` — falls back to last MeterValue |
| `energy_delivered_wh` | `meterStop − meterStart` (authoritative accumulator delta); falls back to positive-diff sum over `Energy.Active.Import.Register` samples |
| `stop_reason` | `StopTransaction.payload.reason` |
| `mean_current_a`, `mean_voltage_v`, `start_soc_pct`, `end_soc_pct`, `capacity_bound_flag`, `initial_*` | Derived from MeterValues wide form; NaN if no MV samples |
| `has_meter_values` | Boolean flag for ML filtering (see §6) |

A session with only Start/Stop but no MeterValues is retained as a **label-only row** (departure known, current profile absent). This is essential: 13 % of raw sessions are in this state, and discarding them would bias the arrival/departure distribution.

### 4.2 Aggregation output (raw)

Run command:

```bash
uv run python scripts/run_phase_b_full_eda.py
```

Streaming aggregator processes each of the 472 files independently and concatenates session rows only (session rows ≪ raw documents, so memory stays bounded).

| Metric | Value |
|---|---|
| Raw sessions | 69,224 |
| MV-present sessions | 54,075 (78.1 %) |
| Label-only sessions (no MV) | 15,149 (21.9 %) |
| Unique chargers | 472 |
| Artifact | `data/phase_b/session_dataset_raw.parquet` (6.0 MB) |

---

## 5. Quality filter (`SessionCleanConfig`)

Raw aggregates contained physically impossible rows (negative duration, meter regression, multi-month "sessions"). A rule-based filter, auditable row-by-row, was introduced in `tinkaton.cleaner.clean_sessions`.

### 5.1 Rule set (defaults)

| Rule | Threshold | Rationale |
|---|---|---|
| `non_positive_duration` | `duration_min ≤ 0` or missing | `stop_ts` earlier than `arrival_ts` — clock skew or payload corruption |
| `duration_below_min` | `< 0.5 min` | 30-second plug-test events; no meaningful current trajectory |
| `duration_above_max` | `> 48 h` | Missed StopTx or hardware fault |
| `non_positive_energy` | `meter_stop < meter_start` | Meter reset or reporting glitch |
| `energy_rate_above_physical_max` | `kWh / h > 7.5` | 7 kW + 10 % margin — exceeds single-phase AC physical limit |
| `orphan_no_stop_reason` | `stop_reason` missing | StopTransaction was never observed; session cannot be anchored |

Configurable via `SessionCleanConfig`; defaults target 7 kW single-phase AC and are conservative.

### 5.2 Rejection outcome

```bash
uv run python scripts/clean_phase_b_sessions.py
```

| Rule | Rejected rows | Share of raw |
|---|---|---|
| `orphan_no_stop_reason` | 8,130 | 11.7 % |
| `duration_below_min` | 1,887 | 2.7 % |
| `non_positive_duration` | 821 | 1.2 % |
| `energy_rate_above_physical_max` | 114 | 0.2 % |
| `duration_above_max` | 94 | 0.1 % |
| `non_positive_energy` | 3 | < 0.1 % |
| **Total rejected** | **11,049** | **16.0 %** |
| **Kept (clean)** | **58,175** | **84.0 %** |

Rejected rows preserved with `rejection_reason` in `data/phase_b/session_dataset_rejected.csv` for audit.

### 5.3 Clean dataset statistics

| Metric | Value |
|---|---|
| Sessions | 58,175 |
| Unique chargers | ≥ 470 |
| Duration: min / median / max | 0.5 min / 177 min / 2,579 min (43 h) |
| Energy: min / median / max | 0.0 kWh / 16.1 kWh / 96.4 kWh |
| `has_meter_values = True` share | 78.0 % |
| `capacity_bound_flag = True` share | 42.9 % (at 20-min, 0.95-ratio threshold) |

### 5.4 Clean `stop_reason` distribution

| Reason | Sessions | Share |
|---|---|---|
| `Other` | 31,096 | 53.5 % |
| `EVDisconnected` | 13,907 | 23.9 % |
| `Local` | 6,970 | 12.0 % |
| `Remote` | 5,295 | 9.1 % |
| `EmergencyStop` | 538 | 0.9 % |
| `PowerLoss` | 359 | 0.6 % |
| `HardReset` | 10 | < 0.1 % |

`EVDisconnected` and short-tail reasons represent the 24 % of sessions where the driver aborted before a natural termination — a non-trivial class for the ML departure predictor.

---

## 6. η structural claim — iterative refinement

This section documents the evolution of the η assumption under empirical evidence. The narrative itself is a defensible Ch.3 "Field validation" element because it shows the assumption was not accepted uncritically.

### 6.1 Stage 1 — initial Phase A claim

Two Phase A test sessions (`260417-ST-22-98.json`, `260420-ST-93-100.json`), both at manually configured PWM 52 % (I_cap = 31.2 A):

- Session 260417: 1,139 MV samples / 9 h 32 min / SoC 22 → 98 % / `binding_ratio = 0.9832 ± 0.0010` over the full window.
- Session 260420: 95 MV samples / 53 min / SoC 93 → 99 % / `binding_ratio = 0.978 → 0.982` flat, single step-down at SoC 99 %.

From these, the initial thesis claim (`HANDOFF_ModelPipeline_v2.md` §3.1):

> η ≈ 0.98 is a structural property of AC Level-2 charging in the Korean fleet because OBC demand exceeds I_cap × 0.6 for all SoC < SoC_taper ≈ 99 %.

LP linearization `P = 220 · η · I_PWM` followed directly.

### 6.2 Stage 2 — naive global I_cap in Phase B

The clean dataset was first evaluated with a fleet-wide `I_cap = 31.2 A`:

```python
binding_ratio_global = mean_current_a / 31.2
```

Filtering to long, normally-completed sessions (MV present, `stop_reason ∈ {Other, Local, Remote}`, `duration ≥ 60 min`, n = 28,172):

| Statistic | Value |
|---|---|
| mean | 0.801 |
| median | 0.863 |
| std | 0.151 |
| P10 / P90 | 0.477 / 0.959 |

**Per-charger median of `binding_ratio_global`** (314 chargers with ≥ 20 long completed sessions):

| Band | Chargers | Share |
|---|---|---|
| 0.95 – 1.00 | 11 | 3.5 % |
| 0.85 – 0.95 | 192 | 61.1 % |
| 0.70 – 0.85 | 87 | 27.7 % |
| 0.50 – 0.70 | 18 | 5.7 % |
| < 0.50 | 6 | 1.9 % |

This appeared bimodal and prompted a working hypothesis: *operators run heterogeneous PWM ceilings per charger, so a single I_cap cannot normalize the fleet*.

### 6.3 Stage 3 — per-charger I_cap estimation

The hypothesis was tested by estimating each charger's I_cap from its own data: the 99-th percentile of `mean_current_a` over its long completed sessions (`tinkaton.transform.estimate_icap_per_charger`).

Run:

```bash
uv run python scripts/normalize_per_charger_icap.py
```

Result (383 chargers with ≥ 1 qualifying session):

| Statistic | `i_cap_observed_a` |
|---|---|
| min | 0.00 (8 outliers) |
| 25 % | 29.90 |
| **median** | **30.48** |
| 75 % | 31.05 |
| max | 35.76 |

Bucket distribution:

| Band (A) | Chargers | Share |
|---|---|---|
| 0 – 10 | 1 | 0.3 % |
| 20 – 25 | 7 | 1.8 % |
| 25 – 30 | 100 | 26.1 % |
| 30 – 33 | 269 | 70.2 % |
| 33 – 40 | 5 | 1.3 % |

**96.3 % of the fleet's observed I_cap lies in the tight band 25 – 33 A, centred on 30.5 A.** The heterogeneity hypothesis is not supported by the hardware distribution.

Rerunning binding ratios with per-charger I_cap:

| Statistic | global (31.2 A) | per-charger |
|---|---|---|
| n | 28,172 | 28,172 |
| mean | 0.8011 | 0.8230 |
| median | 0.8625 | 0.8925 |
| P90 | 0.959 | 0.978 |

Per-charger normalization moves the distribution ≈ 3 % toward Phase A's η = 0.98 — a real but modest shift. It is **not** the dominant source of below-0.98 spread.

### 6.4 Stage 4 — revised interpretation (adopted)

Because the fleet I_cap is effectively constant (~30 A) but session-mean binding ratios spread well below 0.98, the spread must be **session-level physics**, not hardware configuration:

| Duration bucket | Session-mean `binding_ratio_global` |
|---|---|
| < 5 min | 0.22 |
| 5 – 15 min | 0.82 |
| 15 – 60 min | 0.81 |
| 1 – 3 h | 0.80 |
| 3 – 6 h | 0.83 |
| 6 – 12 h | 0.83 |
| > 12 h | 0.58 |

- Very short sessions average low because of OBC soft-start / ramp-up.
- Very long sessions average low because the end-of-charge taper (≥ SoC 99 %) pulls the session mean down.
- Mid-duration sessions run closest to steady state — consistent with Phase A.

**Adopted claim** (replaces naive η = 0.98 constant for session-means):

> AC Level-2 chargers in this fleet operate at a near-uniform hardware ceiling (I_cap ≈ 30.5 A). Steady-state η ≈ 0.98 is preserved (Phase A evidence, P90 of long-session binding ratio = 0.978). Session-mean binding ratios fall below 0.98 because of physical transients — OBC ramp-up and end-of-charge taper — not because of PWM configuration heterogeneity. The LP linearization holds with a single global I_cap ≈ 30 A, and the remaining session-level variance is the quantity the ML departure-time model captures implicitly.

### 6.5 Implications for `HANDOFF_ModelPipeline_v2.md` §3.1

The structural statement is refined, not replaced:

- Before: "η ≈ 0.98 invariant across the AC fleet."
- After: "η_steady ≈ 0.98 is validated by Phase A and by the P90 of long-session field binding ratios. Fleet I_cap is tightly clustered at 30.5 A. LP uses this as the linearization constant. Ramp-up / taper deviations are physical and absorbed into the departure-time ML rather than into η."

---

## 7. Artifact inventory

### 7.1 Data

| Path | Description |
|---|---|
| `data/raw/phase_b/_charger_manifest.csv` | 492 AC chargers with vendor / model |
| `data/raw/phase_b/_checkpoint.json` | Completed chargers (export resume) |
| `data/raw/phase_b/{chargerId}.json` × 472 | Per-charger 3-action OCPP export, BSON extended JSON (relaxed) |
| `data/phase_b/session_dataset_raw.parquet` | 69,224 raw sessions (pre-filter) |
| `data/phase_b/session_dataset.parquet` | Raw, ML-ready column subset |
| `data/phase_b/session_dataset_clean.parquet` | **58,175 clean sessions** |
| `data/phase_b/session_dataset_rejected.csv` | 11,049 rejected rows with `rejection_reason` |
| `data/phase_b/session_dataset_clean_v2.parquet` | Clean + `i_cap_observed_a`, `binding_ratio_self`, `binding_ratio_global` |
| `data/phase_b/charger_icap_manifest.csv` | 383 chargers with `i_cap_observed_a`, `n_sessions_used` |

### 7.2 Reports

| Path | Description |
|---|---|
| `reports/phase_b_full_eda.md` | Raw-dataset narrative + figures |
| `reports/phase_b_clean_summary.md` | Rejection audit + clean-vs-raw comparison |
| `reports/phase_b_icap_per_charger.md` | Per-charger I_cap estimation and renormalization |
| `docs/data_methodology.md` | This document |

### 7.3 Figures (`outputs/phase_b_full_eda/`)

- `duration_hist.png`, `energy_hist.png` — clean-dataset distributions
- `binding_ratio_hist.png` — raw binding ratio spread
- `binding_ratio_raw_vs_clean.png` — effect of quality filter
- `binding_ratio_global_vs_self.png` — per-charger vs global I_cap normalization
- `per_charger_icap.png` — fleet I_cap distribution
- `capacity_bound_by_model.png` — bound-share per charger model
- `stop_reason.png` — termination reason distribution
- `sessions_per_charger.png` — per-charger activity distribution
- `arrival_heatmap.png` — day-of-week × hour arrival heatmap

### 7.4 Code (module interfaces)

| Symbol | Location | Role |
|---|---|---|
| `load_ocpp_logs` | `tinkaton.loader` | Truncated-array-tolerant JSON reader |
| `meter_values_to_dataframe` | `tinkaton.loader` | RECV-only MV wide form |
| `transaction_events_to_dataframe` | `tinkaton.loader` | RECV StartTx / StopTx + messageId pairing |
| `aggregate_sessions` | `tinkaton.transform` | Session row builder (MV + Tx events) |
| `compute_capacity_bound_flag` | `tinkaton.transform` | ≥ 0.95 ratio × 20 min heuristic |
| `estimate_icap_per_charger` | `tinkaton.transform` | Per-charger I_cap from long sessions |
| `clean_sessions` | `tinkaton.cleaner` | Rule-based quality filter |
| `is_ac_model` | `tinkaton.mongo` | `^ELA` / `^E01AS` prefix matcher |
| `list_ac_chargers` | `tinkaton.mongo` | BootNotification aggregate → AC-only fleet |

### 7.5 Scripts (in execution order)

| Script | Purpose |
|---|---|
| `scripts/export_phase_b.py` | 492-charger 3-action export with checkpoint / resume |
| `scripts/run_phase_b_full_eda.py` | Aggregate to session parquet + first-pass EDA |
| `scripts/clean_phase_b_sessions.py` | Apply `SessionCleanConfig` filters |
| `scripts/normalize_per_charger_icap.py` | Per-charger I_cap estimation + renormalization |

### 7.6 Tests

`tests/test_loader.py` (10), `tests/test_transform.py` (20), `tests/test_cleaner.py` (10). Total 40 tests, all synthetic fixtures — no production data required for CI.

---

## 8. Known limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| OBC demand profile not observable on AC side | Cannot separate "vehicle could draw more" from "vehicle chose to draw less" | Structural argument via Phase A + P90 binding ratio; flagged in §6.4 |
| 8 chargers have anomalous `i_cap_observed_a` (< 25 A) | May indicate non-default PWM or sparse sample bias | Kept in dataset; separate flag available via `i_cap_observed_a` column |
| Vehicle identity is `idTag`, not VIN | Cannot directly model per-vehicle OBC behavior | `idTag` retained as categorical; per-vehicle analysis deferred |
| 11 kW chargers (5 units) mixed with 7 kW (487 units) | Potential minor I_cap outlier contribution | Negligible fleet share; not filtered out |
| Single 5.5-month window | No year-over-year concept-drift check | Window covers peak season variability; longer window is future work |
| Orphan (`stop_reason` missing) sessions dropped | 13 % of raw sessions lose their departure label | Conservative choice; documented rejection rate |

---

## 9. Ready-to-use fragments for thesis writing

These blocks are paraphrased from the analyses above and can be inserted into thesis Ch.3 / Ch.8 with minimal editing.

### 9.1 Ch.3 Data section — scope paragraph

> "The dataset is drawn from the MongoDB `logs.PLATFORM-OCPP` collection of a Korean EV-charger aggregator, covering 118.9 million OCPP messages between 2025-11-08 and 2026-04-22. We restrict attention to the 492 AC Level-2 chargers identified through `chargePointModel` prefixes `ELA*` and `E01AS*`, which together represent the operator's single-phase 7 kW (and a small 11 kW fleet) deployment. DC fast chargers, whose session durations and current envelopes differ by orders of magnitude, are structurally outside the scope of this study and are excluded at the classification step."

### 9.2 Ch.3 Pipeline paragraph

> "Per charger, we retain `StartTransaction`, `StopTransaction`, and `MeterValues` records and fuse them into session-level rows keyed by `(chargerId, transactionId)`. Because OCPP `StartTransaction` CALL does not carry `transactionId`, we pair the request with its server-side CALLRESULT using `meta.messageId`; `StopTransaction` CALL already includes `transactionId` and the stop reason. Start and Stop events provide authoritative arrival and departure timestamps and `meterStart` / `meterStop` energy accumulators, which override MeterValue-derived approximations. Sessions without any MeterValues are retained with a `has_meter_values` flag so that departure labels are not biased toward charging-active sessions only."

### 9.3 Ch.3 Data-quality paragraph

> "A rule-based filter (`SessionCleanConfig`) removes sessions with non-positive or > 48 h duration, negative delivered energy, energy delivery rates exceeding the physical maximum of a 7 kW single-phase system (7.5 kWh/h margin), sub-30-second duration, or missing `stop_reason`. From 69,224 raw sessions, 11,049 are rejected — 73.6 % of them because the server never observed a StopTransaction. The clean dataset contains 58,175 sessions, of which 78.0 % carry MeterValue features and 42.9 % satisfy the capacity-bound criterion of `I_actual ≥ 0.95 · I_cap` for at least 20 continuous minutes."

### 9.4 Ch.3 η-validation paragraph

> "Two Phase A pilot sessions collected at a manually configured PWM ceiling of 52 % (I_cap = 31.2 A) yield a binding ratio of 0.983 ± 0.001 over the 22 – 98 % SoC range. Extending this check to the 492-charger field fleet, the per-charger 99-th-percentile of sustained session-mean current concentrates at 30.5 A (IQR 29.9 – 31.1 A, 96.3 % within 25 – 33 A), confirming that operators do not impose heterogeneous PWM ceilings. The 90-th percentile of session-mean binding ratios in long completed sessions is 0.978 under per-charger normalization, matching Phase A. Sub-steady-state session-mean binding ratios (median 0.89) originate from OBC soft-start at session beginning and end-of-charge taper at SoC ≥ 99 %, not from hardware heterogeneity, and are absorbed into the departure-time ML rather than into η."

### 9.5 Ch.8 Limitations paragraph

> "The dataset is a five-and-a-half-month snapshot of a single aggregator's Korean AC fleet. Year-over-year drift, non-Korean OBC populations, and extreme thermal regimes (> 35 °C ambient or < 5 °C battery) are outside the validation scope. Orphan sessions (13 % of raw) were discarded rather than imputed, which biases the retained distribution toward terminated-normally sessions. Per-charger I_cap estimates assume that the 99-th-percentile session-mean current approximates the hardware ceiling, which holds under the observed fleet uniformity but would require explicit `Current.Offered` telemetry to verify on future, possibly more heterogeneous, fleets."

---

## 10. Reproduction

Prerequisites: Python ≥ 3.11, `uv sync`, `configs/db_config.yaml` populated (user enters password interactively at runtime via `getpass`).

```bash
# 1. Classify + export the 492 AC chargers (≈ 4 h)
uv run python scripts/export_phase_b.py

# 2. Aggregate sessions + first-pass EDA (≈ 5 min)
uv run python scripts/run_phase_b_full_eda.py

# 3. Apply quality filter (≈ 30 s)
uv run python scripts/clean_phase_b_sessions.py

# 4. Per-charger I_cap normalization (≈ 10 s)
uv run python scripts/normalize_per_charger_icap.py

# 5. Full test suite (< 2 s)
uv run pytest tests/
```

Outputs land under `data/phase_b/`, `reports/`, and `outputs/phase_b_full_eda/`.

---

## 11. Document history

| Date | Change |
|---|---|
| 2026-04-22 | v1.0 — first consolidated version capturing 492-AC scope, 69 K raw → 58 K clean pipeline, η iterative refinement through per-charger I_cap stage, thesis-ready fragments. |
