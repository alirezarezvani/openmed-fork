"""Tri-axial accelerometer and gyroscope samples on a relative clock.

The contract validates caller-supplied inertial samples locally. It reads no
files, makes no network calls and detects nothing: fall detection, cadence and
platform adapters build on it. Samples are sensitive and stay out of ``repr``,
reports and errors; device identifiers and wall-clock times have no field.
"""

from __future__ import annotations

import datetime as _datetime
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

__all__ = [
    "MAX_SAMPLES",
    "MAX_START_OFFSET_SECONDS",
    "MOTION_RATE_BOUNDS_HZ",
    "MOTION_SCHEMA_VERSION",
    "MOTION_UNITS",
    "NON_DIAGNOSTIC_NOTICE",
    "MotionChannel",
    "MotionChannelKind",
    "MotionContractError",
]

MOTION_SCHEMA_VERSION: Final = "openmed.multimodal.sensors.motion.v1"
NON_DIAGNOSTIC_NOTICE: Final = (
    "Motion samples only; not fall detection, activity or clinical assessment. "
    "Consequential use requires explicit reviewer confirmation."
)
MAX_SAMPLES: Final = 1_000_000
MAX_START_OFFSET_SECONDS: Final = 86_400.0
MOTION_RATE_BOUNDS_HZ: Final = (1.0, 1000.0)


class MotionChannelKind(str, Enum):
    """Inertial channel semantics; magnetometer and fused attitude are refused."""

    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"


MOTION_UNITS = MappingProxyType(
    {
        MotionChannelKind.ACCELEROMETER: frozenset({"g", "m/s^2"}),
        MotionChannelKind.GYROSCOPE: frozenset({"rad/s"}),
    }
)
_RECORD_KEYS = frozenset(
    {"kind", "unit", "sample_rate_hz", "start_offset_seconds", "samples"}
)


class MotionContractError(ValueError):
    """Stable, value-free rejection code; never echoes a sample or offset."""


def _reject(code: str) -> None:
    raise MotionContractError(code)


def _finite(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)  # type: ignore[arg-type]
    except OverflowError:
        return False


def _relative_seconds(value: object) -> float:
    if isinstance(value, _datetime.date | _datetime.time | str):
        _reject("absolute_timestamp")
    if not _finite(value):
        _reject("invalid_start_offset")
    offset = float(value)  # type: ignore[arg-type]
    if offset < 0:
        _reject("invalid_start_offset")
    if offset > MAX_START_OFFSET_SECONDS:
        # Unix seconds or milliseconds, not an offset within one recording.
        _reject("absolute_timestamp")
    return offset


@dataclass(frozen=True, slots=True, repr=False)
class MotionChannel:
    """Immutable tri-axial samples from one inertial sensor, at a regular rate.

    Args:
        kind: ``accelerometer`` or ``gyroscope``.
        unit: ``g`` or ``m/s^2`` for an accelerometer, ``rad/s`` for a
            gyroscope. Spelling is exact; nothing is converted.
        sample_rate_hz: Nominal rate within ``MOTION_RATE_BOUNDS_HZ``.
        samples: Two to ``MAX_SAMPLES`` samples, each an ``(x, y, z)`` triple of
            finite numbers or ``None`` for a missing sample. A missing reading
            is never dropped or interpolated, so sample ``i`` is at
            ``start_offset_seconds + i / sample_rate_hz``.
        start_offset_seconds: Relative seconds from the start of the caller's
            recording, at most one day. Wall-clock values are refused.

    The axis convention and sign are the caller's, recorded only by ``unit``.
    Samples are sensitive data in memory; this is not de-identification.
    """

    kind: MotionChannelKind
    unit: str
    sample_rate_hz: float
    samples: tuple[tuple[float, float, float] | None, ...] = field(repr=False)
    start_offset_seconds: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        try:
            kind = MotionChannelKind(self.kind)
        except (TypeError, ValueError):
            raise MotionContractError("unknown_kind") from None
        if type(self.unit) is not str or self.unit not in MOTION_UNITS[kind]:
            _reject("invalid_unit")
        low, high = MOTION_RATE_BOUNDS_HZ
        if not _finite(self.sample_rate_hz) or not (low <= self.sample_rate_hz <= high):
            _reject("invalid_sample_rate")
        start = _relative_seconds(self.start_offset_seconds)
        if (
            type(self.samples) not in (tuple, list)
            or not 2 <= len(self.samples) <= MAX_SAMPLES
        ):
            _reject("invalid_sample_count")
        samples: list[tuple[float, float, float] | None] = []
        for sample in self.samples:
            if sample is None:
                samples.append(None)
                continue
            if type(sample) not in (tuple, list) or len(sample) != 3:
                _reject("axis_mismatch")
            if not all(_finite(value) for value in sample):
                _reject("invalid_sample")
            samples.append((float(sample[0]), float(sample[1]), float(sample[2])))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "sample_rate_hz", float(self.sample_rate_hz))
        object.__setattr__(self, "start_offset_seconds", start)
        object.__setattr__(self, "samples", tuple(samples))

    @property
    def notice(self) -> str:
        """Return the mandatory samples-only boundary."""
        return NON_DIAGNOSTIC_NOTICE

    @property
    def valid_mask(self) -> tuple[bool, ...]:
        """Return ``True`` for each present sample and ``False`` for each gap."""
        return tuple(sample is not None for sample in self.samples)

    def __repr__(self) -> str:
        return (
            f"MotionChannel(kind={self.kind.value}, sample_count={len(self.samples)})"
        )

    def require_reviewer_confirmation(self, *, confirmed: bool = False) -> None:
        """Fail closed unless the caller explicitly confirms consequential use.

        Confirmation records a caller decision; it does not qualify the sensor,
        the samples or any result derived from them.
        """
        if confirmed is not True:
            _reject("review_required")

    def to_report(self) -> dict[str, object]:
        """Return kind, unit, rate and counts only; never samples or offsets."""
        missing = sum(sample is None for sample in self.samples)
        return {
            "schema_version": MOTION_SCHEMA_VERSION,
            "notice": NON_DIAGNOSTIC_NOTICE,
            "review_required": True,
            "kind": self.kind.value,
            "unit": self.unit,
            "sample_rate_hz": self.sample_rate_hz,
            "sample_count": len(self.samples),
            "missing_count": missing,
        }

    def to_json(self) -> str:
        """Serialize the content-free report deterministically."""
        return json.dumps(
            self.to_report(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @classmethod
    def from_record(cls, record: Any) -> MotionChannel:
        """Build a channel from a ``to_record`` mapping, such as a parity case.

        Raises:
            MotionContractError: For unknown or missing keys, or invalid values.
        """
        if type(record) is not dict or set(record) != _RECORD_KEYS:
            _reject("invalid_record")
        samples = record["samples"]
        if type(samples) is list:
            samples = tuple(
                tuple(sample) if type(sample) is list else sample for sample in samples
            )
        return cls(
            kind=record["kind"],
            unit=record["unit"],
            sample_rate_hz=record["sample_rate_hz"],
            samples=samples,
            start_offset_seconds=record["start_offset_seconds"],
        )

    def to_record(self) -> dict[str, object]:
        """Return the full channel, raw samples included, for local storage.

        Use it for synthetic parity fixtures and caller-owned files only; it is
        not a report and must never be logged or sent anywhere.
        """
        return {
            "kind": self.kind.value,
            "unit": self.unit,
            "sample_rate_hz": self.sample_rate_hz,
            "start_offset_seconds": self.start_offset_seconds,
            "samples": [None if s is None else list(s) for s in self.samples],
        }
