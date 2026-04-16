"""Analyze charging session JSON files for thesis data quality assessment."""

import json
import pathlib
from datetime import datetime, timedelta

DATA_DIR = pathlib.Path("data/processed")


def parse_timestamp(ts: dict | str) -> datetime:
    """Parse MongoDB-style timestamp or ISO string."""
    if isinstance(ts, dict) and "$date" in ts:
        ts = ts["$date"]
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def analyze_file(path: pathlib.Path) -> dict:
    with open(path) as f:
        messages = json.load(f)

    n = len(messages)
    if n == 0:
        return {"file": path.name, "count": 0}

    # charger ID
    charger_id = messages[0].get("meta", {}).get("chargerId", "?")

    # Timestamps from meterValue (the charger-side timestamp), fallback to top-level
    timestamps = []
    soc_values = []
    energy_values = []
    power_values = []

    for msg in messages:
        # Top-level timestamp as fallback
        top_ts = msg.get("timestamp")

        meter_values = msg.get("meta", {}).get("payload", {}).get("meterValue", [])
        for mv in meter_values:
            mv_ts = mv.get("timestamp", top_ts)
            if mv_ts:
                try:
                    timestamps.append(parse_timestamp(mv_ts))
                except Exception:
                    pass

            for sv in mv.get("sampledValue", []):
                measurand = sv.get("measurand", "")
                try:
                    val = float(sv.get("value", ""))
                except (ValueError, TypeError):
                    continue

                if measurand == "SoC":
                    soc_values.append(val)
                elif measurand == "Energy.Active.Import.Register":
                    energy_values.append(val)
                elif measurand == "Power.Active.Import":
                    power_values.append(val)

    # Duration
    if len(timestamps) >= 2:
        timestamps.sort()
        duration = timestamps[-1] - timestamps[0]
    else:
        duration = timedelta(0)

    duration_min = duration.total_seconds() / 60

    # SoC
    has_soc = len(soc_values) > 0
    soc_min = min(soc_values) if has_soc else None
    soc_max = max(soc_values) if has_soc else None
    soc_range = (soc_max - soc_min) if has_soc else 0

    # Energy & Power
    max_energy = max(energy_values) if energy_values else None
    max_power = max(power_values) if power_values else None

    return {
        "file": path.name,
        "count": n,
        "charger_id": charger_id,
        "duration_min": duration_min,
        "has_soc": has_soc,
        "soc_min": soc_min,
        "soc_max": soc_max,
        "soc_range": soc_range,
        "max_energy_wh": max_energy,
        "max_power_w": max_power,
    }


def categorize(r: dict) -> str:
    if (
        r["count"] >= 20
        and r["duration_min"] >= 10
        and r["has_soc"]
        and r["soc_range"] >= 10
    ):
        return "Valuable"
    elif r["count"] >= 10 and r["duration_min"] >= 5:
        return "Moderate"
    else:
        return "Too short"


def fmt_duration(minutes: float) -> str:
    h = int(minutes // 60)
    m = int(minutes % 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m"


def fmt_energy(wh: float | None) -> str:
    if wh is None:
        return "-"
    if wh >= 1000:
        return f"{wh / 1000:.2f} kWh"
    return f"{wh:.0f} Wh"


def fmt_power(w: float | None) -> str:
    if w is None:
        return "-"
    if w >= 1000:
        return f"{w / 1000:.1f} kW"
    return f"{w:.0f} W"


def main():
    files = sorted(DATA_DIR.glob("*.json"))
    if not files:
        print("No JSON files found in", DATA_DIR)
        return

    results = [analyze_file(f) for f in files]
    results.sort(key=lambda r: r["count"], reverse=True)

    # Assign categories
    for r in results:
        r["category"] = categorize(r)

    # Print table
    header = f"{'File':<30} {'Msgs':>5} {'Charger':<18} {'Duration':>9} {'SoC':>10} {'Energy':>12} {'Power':>10} {'Category':<10}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in results:
        soc_str = (
            f"{r['soc_min']:.0f}-{r['soc_max']:.0f}%"
            if r["has_soc"]
            else "-"
        )
        print(
            f"{r['file']:<30} {r['count']:>5} {r.get('charger_id', '?'):<18} "
            f"{fmt_duration(r['duration_min']):>9} {soc_str:>10} "
            f"{fmt_energy(r.get('max_energy_wh')):>12} {fmt_power(r.get('max_power_w')):>10} "
            f"{r['category']:<10}"
        )

    print(sep)
    print(f"Total files: {len(results)}")
    print()

    # Category summary
    categories = {"Valuable": [], "Moderate": [], "Too short": []}
    for r in results:
        categories[r["category"]].append(r["file"])

    for cat in ["Valuable", "Moderate", "Too short"]:
        items = categories[cat]
        print(f"\n=== {cat} ({len(items)} files) ===")
        for name in items:
            print(f"  {name}")


if __name__ == "__main__":
    main()
