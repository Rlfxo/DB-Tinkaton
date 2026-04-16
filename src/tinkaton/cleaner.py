"""Data cleaning and filtering utilities."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def split_by_transaction(input_path: str | Path, output_dir: str | Path | None = None) -> dict:
    """MeterValues JSON을 transactionId + connectorId 기준으로 분류하여 개별 파일로 저장.

    파일명 형식: T{transactionId}C{connectorId}-{YYYYMMDD}.json
    날짜는 해당 트랜잭션의 첫 번째 메시지 timestamp 기준.

    Returns:
        dict with keys 'saved' (list of saved file paths) and 'skipped' (count of messages without transactionId).
    """
    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent.parent / "processed"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)

    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    skipped = 0

    for record in records:
        payload = record.get("meta", {}).get("payload", {})
        txn_id = payload.get("transactionId")
        conn_id = payload.get("connectorId")

        if txn_id is None:
            skipped += 1
            continue

        groups[(txn_id, conn_id or 0)].append(record)

    saved = []
    for (txn_id, conn_id), messages in sorted(groups.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        messages.sort(key=lambda r: r.get("timestamp", {}).get("$date", ""))

        first_ts = messages[0].get("timestamp", {}).get("$date", "")
        try:
            date_str = datetime.fromisoformat(first_ts.replace("Z", "+00:00")).strftime("%Y%m%d")
        except (ValueError, AttributeError):
            date_str = "unknown"

        filename = f"T{txn_id}C{conn_id}-{date_str}.json"
        out_path = output_dir / filename

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

        saved.append(str(out_path))

    return {"saved": saved, "skipped": skipped}
