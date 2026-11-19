# Tri-axial motion sample contract

`openmed.multimodal.sensors.motion` validates caller-supplied accelerometer and
gyroscope samples locally and deterministically. It reads no files, makes no
network calls and detects nothing. Fall detection, cadence, platform adapters
and cross-modal alignment are separate components that build on it. The schema
is `openmed.multimodal.sensors.motion.v1`.

Every report carries this boundary:

> Motion samples only; not fall detection, activity or clinical assessment.
> Consequential use requires explicit reviewer confirmation.

## Canonical representation

- `MotionChannel.kind` is `accelerometer` or `gyroscope`. Magnetometer,
  fused attitude and unknown kinds are refused.
- `unit` is declared exactly and never converted: `g` or `m/s^2` for an
  accelerometer, `rad/s` for a gyroscope. Other spellings (`G`, `m/s2`,
  `deg/s`) are refused rather than guessed.
- `sample_rate_hz` is a finite number from 1 to 1000 Hz.
- `samples` holds 2 to 1,000,000 entries. Each is an `(x, y, z)` triple of
  finite numbers or `None` for a missing reading; `valid_mask` derives from the
  `None` entries. A missing reading keeps its slot, so sample `i` is at
  `start_offset_seconds + i / sample_rate_hz`. A triple with two or four values
  is refused as `axis_mismatch`.
- `start_offset_seconds` is relative seconds within the caller's recording,
  from 0 to 86,400. Dates, times, ISO strings and values above one day (Unix
  seconds or milliseconds) are refused as `absolute_timestamp`.
- There is no field for a device identifier, model name, wall-clock time or
  source path, so none can leak through the record.

The axis direction and sign are the caller's. Platforms differ: Android's raw
accelerometer reads about +9.81 m/s^2 on the upward axis at rest, while Core
Motion's raw accelerometer reads about -1 g there. Adapters must state and
apply their conversion before building a channel.

## Synthetic example

```python
from openmed.multimodal.sensors.motion import MotionChannel

channel = MotionChannel(
    kind="accelerometer",
    unit="g",
    sample_rate_hz=50,
    samples=((0.0, 1.0, 0.0), None, (0.01, 0.99, -0.02)),
)
assert channel.valid_mask == (True, False, True)
assert channel.to_report()["missing_count"] == 1
```

## Privacy and review boundary

Samples are sensitive in memory; validation is not de-identification. `repr`,
`to_report()` and `to_json()` carry only the schema version, notice,
`review_required: true`, kind, unit, rate, sample count and missing count.
Errors are `MotionContractError` with a stable code only, such as
`invalid_unit`, `invalid_sample_rate`, `axis_mismatch`, `invalid_sample` or
`absolute_timestamp`, and never echo a supplied value.

`to_record()` and `from_record()` carry the raw samples. They exist for
caller-owned storage and for the shared parity fixture
`tests/fixtures/parity/sensor_records_v1.json`, which later Swift and Kotlin
ports load; never log or transmit a record.

`require_reviewer_confirmation()` fails closed unless the caller passes
`confirmed=True` after explicit review. Confirmation records a caller decision;
it does not qualify the sensor, the samples or anything derived from them.

All tests and fixtures use synthetic values. No dataset, model or dependency is
bundled.
