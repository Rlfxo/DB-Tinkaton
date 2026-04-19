"""Tests for tinkaton.loader."""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from tinkaton.loader import (
    load_ocpp_logs,
    logs_to_dataframe,
    meter_values_to_dataframe,
    meter_values_to_long_dataframe,
)

SAMPLE_METER_VALUES = {
    "_id": {"$oid": "69e21665bd6d0f0007583b06"},
    "timestamp": {"$date": "2026-04-17T11:15:49.762Z"},
    "level": "info",
    "message": "OCPP Recv [CALL] - MeterValues",
    "meta": {
        "chargerId": "TEST001",
        "serverRecvType": "RECV",
        "messageType": "CALL",
        "action": "MeterValues",
        "messageId": "test-msg-1",
        "payload": {
            "connectorId": 1,
            "meterValue": [
                {
                    "timestamp": "2026-04-17T20:15:48+09:00",
                    "sampledValue": [
                        {
                            "value": "72001416",
                            "measurand": "Energy.Active.Import.Register",
                            "unit": "Wh",
                        },
                        {"value": "232.53", "measurand": "Voltage", "unit": "V"},
                        {"value": "30.61", "measurand": "Current.Import", "unit": "A"},
                        {
                            "value": "45",
                            "measurand": "Temperature",
                            "location": "Body",
                            "unit": "Celsius",
                        },
                        {"value": "98", "measurand": "SoC", "unit": "Percent"},
                    ],
                }
            ],
        },
        "sessionInfo": {"sessionId": "test-session-1"},
    },
}

SAMPLE_HEARTBEAT = {
    "_id": {"$oid": "69e21665bd6d0f0007583b07"},
    "timestamp": {"$date": "2026-04-17T11:16:00.000Z"},
    "level": "info",
    "message": "OCPP Recv [CALL] - Heartbeat",
    "meta": {
        "chargerId": "TEST001",
        "serverRecvType": "RECV",
        "messageType": "CALL",
        "action": "Heartbeat",
        "messageId": "test-msg-2",
        "payload": {},
    },
}


def test_load_ocpp_logs_repairs_truncated_array(tmp_path):
    path = tmp_path / "truncated.json"
    content = json.dumps([SAMPLE_METER_VALUES])
    path.write_text(content[:-1])

    logs = load_ocpp_logs(path)

    assert len(logs) == 1
    assert logs[0]["meta"]["chargerId"] == "TEST001"


def test_load_ocpp_logs_valid_array(tmp_path):
    path = tmp_path / "valid.json"
    path.write_text(json.dumps([SAMPLE_METER_VALUES, SAMPLE_HEARTBEAT]))

    logs = load_ocpp_logs(path)

    assert len(logs) == 2


def test_load_ocpp_logs_rejects_non_array(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(SAMPLE_METER_VALUES))

    with pytest.raises(ValueError):
        load_ocpp_logs(path)


def test_meter_values_to_dataframe_basic():
    df = meter_values_to_dataframe([SAMPLE_METER_VALUES, SAMPLE_HEARTBEAT])

    assert len(df) == 1
    row = df.iloc[0]
    assert row["soc_pct"] == 98.0
    assert row["voltage_v"] == pytest.approx(232.53)
    assert row["current_a"] == pytest.approx(30.61)
    assert row["energy_wh"] == 72001416.0
    assert row["temperature_body_c"] == 45.0
    assert row["charger_id"] == "TEST001"
    assert row["session_id"] == "test-session-1"
    assert row["connector_id"] == 1
    assert isinstance(row["timestamp"], pd.Timestamp)
    assert row["timestamp"].tz is not None


def test_meter_values_to_dataframe_column_ordering():
    df = meter_values_to_dataframe([SAMPLE_METER_VALUES])
    expected_head = [
        "timestamp",
        "server_timestamp",
        "charger_id",
        "session_id",
        "connector_id",
        "transaction_id",
        "message_id",
    ]
    assert list(df.columns[: len(expected_head)]) == expected_head


def test_meter_values_to_dataframe_multi_phase():
    log = copy.deepcopy(SAMPLE_METER_VALUES)
    log["meta"]["payload"]["meterValue"][0]["sampledValue"] = [
        {"value": "230.1", "measurand": "Voltage", "phase": "L1", "unit": "V"},
        {"value": "231.4", "measurand": "Voltage", "phase": "L2", "unit": "V"},
        {"value": "229.8", "measurand": "Voltage", "phase": "L3", "unit": "V"},
    ]

    df = meter_values_to_dataframe([log])

    assert df.iloc[0]["voltage_l1_v"] == pytest.approx(230.1)
    assert df.iloc[0]["voltage_l2_v"] == pytest.approx(231.4)
    assert df.iloc[0]["voltage_l3_v"] == pytest.approx(229.8)


def test_meter_values_to_long_dataframe():
    df = meter_values_to_long_dataframe([SAMPLE_METER_VALUES, SAMPLE_HEARTBEAT])

    assert len(df) == 5
    assert set(df["measurand"]) == {
        "Energy.Active.Import.Register",
        "Voltage",
        "Current.Import",
        "Temperature",
        "SoC",
    }
    assert df.loc[df["measurand"] == "SoC", "value"].iloc[0] == 98.0


def test_meter_values_sorted_and_utc():
    earlier = copy.deepcopy(SAMPLE_METER_VALUES)
    earlier["meta"]["messageId"] = "earlier"
    earlier["meta"]["payload"]["meterValue"][0]["timestamp"] = "2026-04-17T20:15:18+09:00"

    later = SAMPLE_METER_VALUES

    df = meter_values_to_dataframe([later, earlier])

    assert df.iloc[0]["message_id"] == "earlier"
    assert df.iloc[1]["message_id"] == "test-msg-1"
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_logs_to_dataframe_preserves_all_actions():
    df = logs_to_dataframe([SAMPLE_METER_VALUES, SAMPLE_HEARTBEAT])

    assert len(df) == 2
    assert set(df["action"]) == {"MeterValues", "Heartbeat"}
    assert df["server_timestamp"].is_monotonic_increasing


def test_empty_input_returns_empty_frame():
    assert meter_values_to_dataframe([]).empty
    assert meter_values_to_long_dataframe([]).empty
    assert logs_to_dataframe([]).empty
