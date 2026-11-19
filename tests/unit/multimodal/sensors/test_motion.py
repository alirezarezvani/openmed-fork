"""Contract tests for tri-axial motion samples (#4671); synthetic values only."""

from __future__ import annotations

import datetime
import json
import math
import random
from pathlib import Path

import pytest

from openmed.multimodal.sensors.motion import (
    MAX_SAMPLES,
    MOTION_SCHEMA_VERSION,
    NON_DIAGNOSTIC_NOTICE,
    MotionChannel,
    MotionChannelKind,
    MotionContractError,
)

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "parity"
    / "sensor_records_v1.json"
)
UPRIGHT = ((0.0, 1.0, 0.0), (0.01, 0.99, -0.02), None, (0.0, 1.0, 0.0))
SENTINEL = 7.123456789


def _channel(**overrides):
    values = {
        "kind": "accelerometer",
        "unit": "g",
        "sample_rate_hz": 50,
        "samples": UPRIGHT,
    }
    values.update(overrides)
    return MotionChannel(**values)


def _rejects(code, **overrides):
    with pytest.raises(MotionContractError, match=f"^{code}$") as caught:
        _channel(**overrides)
    assert caught.value.__cause__ is None
    return caught.value


def test_valid_channel_normalizes_without_changing_samples():
    channel = _channel(start_offset_seconds=2)
    assert channel.kind is MotionChannelKind.ACCELEROMETER
    assert channel.sample_rate_hz == 50.0
    assert channel.start_offset_seconds == 2.0
    assert channel.samples == UPRIGHT
    assert channel.valid_mask == (True, True, False, True)
    assert channel.notice == NON_DIAGNOSTIC_NOTICE
    list_input = _channel(samples=[list(s) if s else None for s in UPRIGHT])
    assert list_input == _channel()


@pytest.mark.parametrize(
    ("kind", "unit"),
    [("accelerometer", "g"), ("accelerometer", "m/s^2"), ("gyroscope", "rad/s")],
)
def test_declared_units_are_accepted_exactly(kind, unit):
    assert _channel(kind=kind, unit=unit).to_report()["unit"] == unit


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"kind": "magnetometer"}, "unknown_kind"),
        ({"kind": None}, "unknown_kind"),
        ({"unit": "G"}, "invalid_unit"),
        ({"unit": "m/s2"}, "invalid_unit"),
        ({"kind": "gyroscope", "unit": "g"}, "invalid_unit"),
        ({"kind": "gyroscope", "unit": "deg/s"}, "invalid_unit"),
        ({"unit": b"g"}, "invalid_unit"),
        ({"sample_rate_hz": 0}, "invalid_sample_rate"),
        ({"sample_rate_hz": 0.99}, "invalid_sample_rate"),
        ({"sample_rate_hz": 1000.5}, "invalid_sample_rate"),
        ({"sample_rate_hz": -50}, "invalid_sample_rate"),
        ({"sample_rate_hz": math.nan}, "invalid_sample_rate"),
        ({"sample_rate_hz": math.inf}, "invalid_sample_rate"),
        ({"sample_rate_hz": True}, "invalid_sample_rate"),
        ({"sample_rate_hz": "50"}, "invalid_sample_rate"),
        ({"sample_rate_hz": 10**400}, "invalid_sample_rate"),
        ({"samples": ((0.0, 1.0, 0.0),)}, "invalid_sample_count"),
        ({"samples": ()}, "invalid_sample_count"),
        ({"samples": None}, "invalid_sample_count"),
        ({"samples": iter(UPRIGHT)}, "invalid_sample_count"),
        ({"samples": ((0.0, 1.0, 0.0), (0.0, 1.0))}, "axis_mismatch"),
        ({"samples": ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 0.0))}, "axis_mismatch"),
        ({"samples": ((0.0, 1.0, 0.0), 1.0)}, "axis_mismatch"),
        ({"samples": ((0.0, 1.0, 0.0), "xyz")}, "axis_mismatch"),
        ({"samples": ((0.0, 1.0, 0.0), (0.0, math.nan, 0.0))}, "invalid_sample"),
        ({"samples": ((0.0, 1.0, 0.0), (math.inf, 1.0, 0.0))}, "invalid_sample"),
        ({"samples": ((0.0, 1.0, 0.0), (0.0, None, 0.0))}, "invalid_sample"),
        ({"samples": ((0.0, 1.0, 0.0), (0.0, True, 0.0))}, "invalid_sample"),
        ({"samples": ((0.0, 1.0, 0.0), (0.0, 10**400, 0.0))}, "invalid_sample"),
        ({"start_offset_seconds": -0.01}, "invalid_start_offset"),
        ({"start_offset_seconds": math.nan}, "invalid_start_offset"),
        ({"start_offset_seconds": True}, "invalid_start_offset"),
        ({"start_offset_seconds": None}, "invalid_start_offset"),
        ({"start_offset_seconds": 1_763_510_400.0}, "absolute_timestamp"),
        ({"start_offset_seconds": 1_763_510_400_000}, "absolute_timestamp"),
        ({"start_offset_seconds": "2026-11-19T08:00:00Z"}, "absolute_timestamp"),
        (
            {"start_offset_seconds": datetime.datetime(2026, 11, 19, 8)},
            "absolute_timestamp",
        ),
        ({"start_offset_seconds": datetime.date(2026, 11, 19)}, "absolute_timestamp"),
        ({"start_offset_seconds": datetime.time(8, 0)}, "absolute_timestamp"),
    ],
)
def test_invalid_input_fails_with_stable_value_free_codes(overrides, code):
    _rejects(code, **overrides)


def test_relative_offset_of_one_day_is_the_limit():
    assert _channel(start_offset_seconds=86_400).start_offset_seconds == 86_400.0
    _rejects("absolute_timestamp", start_offset_seconds=86_400.001)


def test_sample_count_is_bounded():
    _rejects("invalid_sample_count", samples=(None,) * (MAX_SAMPLES + 1))


def test_wholly_missing_channel_is_representable_and_counted():
    report = _channel(samples=(None, None, None)).to_report()
    assert report["sample_count"] == 3
    assert report["missing_count"] == 3


def test_channel_has_no_device_or_wall_clock_field():
    names = {name for name in MotionChannel.__dataclass_fields__}
    assert names == {
        "kind",
        "unit",
        "sample_rate_hz",
        "samples",
        "start_offset_seconds",
    }
    with pytest.raises(TypeError):
        _channel(device_id="synthetic-device")


def test_report_carries_kind_unit_rate_and_counts_only():
    channel = _channel(kind="gyroscope", unit="rad/s", sample_rate_hz=200)
    assert channel.to_report() == {
        "schema_version": MOTION_SCHEMA_VERSION,
        "notice": NON_DIAGNOSTIC_NOTICE,
        "review_required": True,
        "kind": "gyroscope",
        "unit": "rad/s",
        "sample_rate_hz": 200.0,
        "sample_count": 4,
        "missing_count": 1,
    }
    assert json.loads(channel.to_json()) == channel.to_report()
    assert (
        channel.to_json()
        == _channel(kind="gyroscope", unit="rad/s", sample_rate_hz=200.0).to_json()
    )


def test_sentinel_samples_never_appear_in_repr_reports_or_errors():
    text = repr(SENTINEL)
    channel = _channel(
        samples=((SENTINEL, SENTINEL, SENTINEL), (0.0, 1.0, 0.0)),
        start_offset_seconds=SENTINEL,
    )
    surfaces = [
        repr(channel),
        str(channel),
        channel.to_json(),
        str(channel.to_report()),
    ]
    for overrides in (
        {"samples": ((SENTINEL, SENTINEL), (0.0, 1.0, 0.0))},
        {"samples": ((SENTINEL, math.nan, SENTINEL), (0.0, 1.0, 0.0))},
        {"sample_rate_hz": SENTINEL * 1000},
        {"start_offset_seconds": SENTINEL * 1e9},
        {"unit": f"g{SENTINEL}"},
    ):
        with pytest.raises(MotionContractError) as caught:
            _channel(**overrides)
        surfaces += [str(caught.value), repr(caught.value)]
    with pytest.raises(MotionContractError) as caught:
        channel.require_reviewer_confirmation()
    surfaces.append(str(caught.value))
    assert all(text not in surface and "7.12" not in surface for surface in surfaces)


def test_reviewer_confirmation_fails_closed():
    channel = _channel()
    for value in (False, None, 1, "yes"):
        with pytest.raises(MotionContractError, match="^review_required$"):
            channel.require_reviewer_confirmation(confirmed=value)
    channel.require_reviewer_confirmation(confirmed=True)
    assert channel.to_report()["review_required"] is True


def test_channel_is_immutable():
    channel = _channel()
    with pytest.raises(AttributeError):
        channel.samples = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "record",
    [
        None,
        [],
        {"kind": "accelerometer"},
        {
            "kind": "accelerometer",
            "unit": "g",
            "sample_rate_hz": 50.0,
            "start_offset_seconds": 0.0,
            "samples": [[0.0, 1.0, 0.0]] * 2,
            "device_id": "synthetic-device",
        },
    ],
)
def test_records_with_unknown_or_missing_keys_fail_closed(record):
    with pytest.raises(MotionContractError, match="^invalid_record$"):
        MotionChannel.from_record(record)


def test_parity_fixture_round_trips_byte_identically():
    text = FIXTURE.read_text(encoding="utf-8")
    document = json.loads(text)
    assert document["synthetic"] is True
    assert document["schema_version"] == MOTION_SCHEMA_VERSION
    rebuilt = []
    for case in document["cases"]:
        if "error" in case:
            with pytest.raises(MotionContractError) as caught:
                MotionChannel.from_record(case["record"])
            assert str(caught.value) == case["error"], case["name"]
            rebuilt.append(case)
            continue
        channel = MotionChannel.from_record(case["record"])
        assert channel.to_json() == case["report_json"], case["name"]
        rebuilt.append(
            {
                "name": case["name"],
                "record": channel.to_record(),
                "report_json": channel.to_json(),
            }
        )
    assert {case["name"] for case in rebuilt} == {
        case["name"] for case in document["cases"]
    }
    again = {**document, "cases": rebuilt}
    encoded = json.dumps(again, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    assert encoded == text
    codes = {case["error"] for case in document["cases"] if "error" in case}
    assert {"unknown_kind", "invalid_unit", "invalid_sample_rate"} <= codes
    assert {"axis_mismatch", "absolute_timestamp"} <= codes


def test_random_inputs_fail_with_a_code_or_keep_every_invariant():
    rng = random.Random(4671)
    values = (0.0, -1.0, 1.0, 9.80665, 1e308, math.nan, math.inf, True, None, "1")
    realized = 0
    for _ in range(2000):
        samples = []
        for _ in range(rng.randrange(0, 8)):
            roll = rng.random()
            if roll < 0.2:
                samples.append(None)
            elif roll < 0.3:
                samples.append(tuple(rng.choice(values) for _ in range(2)))
            else:
                samples.append(
                    tuple(
                        rng.choice(values) if rng.random() < 0.1 else rng.gauss(0, 2)
                        for _ in range(3)
                    )
                )
        record = {
            "kind": rng.choice(("accelerometer", "gyroscope", "magnetometer")),
            "unit": rng.choice(("g", "m/s^2", "rad/s", "deg/s")),
            "sample_rate_hz": rng.choice((0.5, 1, 25.0, 100, 1000.0, 1001, True)),
            "start_offset_seconds": rng.choice((0, 3.5, 86_400, 1.7e9, -1.0)),
            "samples": samples,
        }
        try:
            channel = MotionChannel.from_record(record)
        except MotionContractError as error:
            assert str(error).isidentifier()
            assert error.__cause__ is None
            continue
        realized += 1
        report = channel.to_report()
        assert report["sample_count"] == len(samples) >= 2
        assert report["missing_count"] == samples.count(None)
        assert MotionChannel.from_record(channel.to_record()) == channel
        assert all(
            s is None or all(math.isfinite(v) for v in s) for s in channel.samples
        )
    assert realized >= 40
