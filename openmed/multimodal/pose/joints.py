"""Versioned mappings from estimator joint vocabularies onto canonical joints.

MediaPipe BlazePose returns 33 indexed landmarks; Apple Vision returns 17 named
3D joints or 19 named 2D joints. Each table here maps every source joint to
exactly one canonical joint from ``CANONICAL_JOINTS`` or to ``None``
(unmapped), and lists the canonical joints the source cannot produce as
declared-missing instead of synthesizing them. A metric-requirements table
states which canonical joints each pose metric needs, so a later gate can
abstain when a source cannot supply them.

The module holds data only: it runs no estimator, reads no coordinates and
computes no metric. Errors carry a stable code and never echo the rejected
value.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, NoReturn

from .sequence import CANONICAL_JOINTS, PoseContractError

__all__ = [
    "JOINT_TABLES",
    "METRIC_REQUIREMENTS",
    "POSE_JOINTS_SCHEMA_VERSION",
    "JointMapping",
    "JointMappingTable",
    "JointSource",
    "MetricJointRequirement",
    "PoseMetric",
    "canonical_joint",
    "joint_mapping_record",
    "missing_joint_groups",
    "resolve_metric_joints",
]

POSE_JOINTS_SCHEMA_VERSION: Final = "openmed.multimodal.pose.joints.v1"
_SOURCE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_CANONICAL = frozenset(CANONICAL_JOINTS)


def _reject(code: str) -> NoReturn:
    raise PoseContractError(code)


class JointSource(str, Enum):
    """Joint vocabularies with a mapping table."""

    BLAZEPOSE_33 = "blazepose_33"
    APPLE_VISION_3D = "apple_vision_3d"
    APPLE_VISION_2D = "apple_vision_2d"


class PoseMetric(str, Enum):
    """Pose metric families that declare joint requirements."""

    GAIT_TIMING = "gait_timing"
    SIT_TO_STAND = "sit_to_stand"
    POSTURE = "posture"


def _source(value: object) -> JointSource | None:
    """Return the ``JointSource`` member or ``None``, never raising.

    The caller raises outside any handler, so no exception that carries the
    rejected value survives as ``__context__``.
    """
    if isinstance(value, JointSource):
        return value
    if type(value) is not str:
        return None
    for member in JointSource:
        if member.value == value:
            return member
    return None


def _metric(value: object) -> PoseMetric | None:
    if isinstance(value, PoseMetric):
        return value
    if type(value) is not str:
        return None
    for member in PoseMetric:
        if member.value == value:
            return member
    return None


def _version(value: object) -> bool:
    return type(value) is int and value >= 1


def _digest(record: Mapping[str, object]) -> str:
    text = json.dumps(record, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class JointMapping:
    """One source joint and the canonical joint it maps to.

    Args:
        source_joint: A non-negative landmark index for indexed vocabularies
            (BlazePose) or the joint name for named vocabularies (Apple
            Vision, using the Swift ``JointName`` property names).
        canonical: A name from ``CANONICAL_JOINTS``, or ``None`` when the
            source joint has no canonical counterpart and is dropped.
    """

    source_joint: int | str
    canonical: str | None

    def __post_init__(self) -> None:
        key = self.source_joint
        if type(key) is int:
            if key < 0:
                _reject("mapping_table_invalid")
        elif type(key) is not str or not _SOURCE_NAME.fullmatch(key):
            _reject("mapping_table_invalid")
        if self.canonical is not None and (
            type(self.canonical) is not str or self.canonical not in _CANONICAL
        ):
            _reject("mapping_table_invalid")


@dataclass(frozen=True, slots=True)
class JointMappingTable:
    """A versioned, complete mapping for one joint source.

    Args:
        source: The vocabulary the table maps from.
        version: Positive integer; bump it whenever any mapping changes.
        mappings: Every joint the source can return, each exactly once. All
            keys share one type: indices for ``blazepose_33``, names for the
            Apple Vision sources. A canonical joint is the target of at most
            one source joint.

    Canonical joints that no mapping targets are exposed as ``missing``:
    the source cannot produce them, and callers must leave them out of
    frames rather than derive placeholder coordinates.
    """

    source: JointSource
    version: int
    mappings: tuple[JointMapping, ...]

    def __post_init__(self) -> None:
        source = _source(self.source)
        if source is None:
            _reject("joint_source_invalid")
        if not _version(self.version):
            _reject("mapping_table_invalid")
        if type(self.mappings) not in (tuple, list) or not self.mappings:
            _reject("mapping_table_invalid")
        if any(type(entry) is not JointMapping for entry in self.mappings):
            _reject("mapping_table_invalid")
        key_type = int if source is JointSource.BLAZEPOSE_33 else str
        keys = [entry.source_joint for entry in self.mappings]
        if any(type(key) is not key_type for key in keys) or len(set(keys)) != len(
            keys
        ):
            _reject("mapping_table_invalid")
        if key_type is int and sorted(keys) != list(range(len(keys))):  # type: ignore[type-var]
            _reject("mapping_table_invalid")
        targets = [entry.canonical for entry in self.mappings if entry.canonical]
        if len(set(targets)) != len(targets):
            _reject("mapping_table_invalid")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "mappings", tuple(self.mappings))

    @property
    def missing(self) -> tuple[str, ...]:
        """Return canonical joints this source cannot produce, in order."""
        produced = {entry.canonical for entry in self.mappings}
        return tuple(name for name in CANONICAL_JOINTS if name not in produced)

    @property
    def produced(self) -> frozenset[str]:
        """Return the canonical joints this source can produce."""
        return frozenset(entry.canonical for entry in self.mappings if entry.canonical)

    @property
    def digest(self) -> str:
        """Return the SHA-256 of the table's canonical JSON form."""
        return _digest(self._content())

    def canonical_joint(self, source_joint: object) -> str | None:
        """Map one source joint, or return ``None`` when it is unmapped.

        Raises:
            PoseContractError: ``source_joint_unknown`` for a name or index the
                source does not define, including a wrong key type.
        """
        for entry in self.mappings:
            if (
                type(source_joint) is type(entry.source_joint)
                and source_joint == entry.source_joint
            ):
                return entry.canonical
        _reject("source_joint_unknown")

    def _content(self) -> dict[str, object]:
        return {
            "source": self.source.value,
            "version": self.version,
            "mappings": [
                {"source_joint": entry.source_joint, "canonical": entry.canonical}
                for entry in self.mappings
            ],
            "missing": list(self.missing),
        }

    def to_record(self) -> dict[str, object]:
        """Return the table with its digest, as written to the parity fixture."""
        return {**self._content(), "digest": self.digest}


@dataclass(frozen=True, slots=True)
class MetricJointRequirement:
    """The canonical joints a metric family needs from a source.

    Args:
        metric: The metric family.
        version: Positive integer; bump it whenever the groups change.
        requires: Groups of canonical joints. A source satisfies the
            requirement when it produces at least one joint of every group;
            a single-joint group is a hard requirement. Joints within a group
            are in order of preference, so a later joint is a fallback.

    This checks vocabulary capability only. Per-frame visibility and view
    checks belong to the metric itself.
    """

    metric: PoseMetric
    version: int
    requires: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        metric = _metric(self.metric)
        if metric is None:
            _reject("metric_invalid")
        if not _version(self.version):
            _reject("metric_requirement_invalid")
        groups = self.requires
        if type(groups) not in (tuple, list) or not groups:
            _reject("metric_requirement_invalid")
        seen: set[str] = set()
        normalized = []
        for group in groups:
            if type(group) not in (tuple, list) or not group:
                _reject("metric_requirement_invalid")
            for joint in group:
                if type(joint) is not str or joint not in _CANONICAL or joint in seen:
                    _reject("metric_requirement_invalid")
                seen.add(joint)
            normalized.append(tuple(group))
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "requires", tuple(normalized))

    @property
    def digest(self) -> str:
        """Return the SHA-256 of the requirement's canonical JSON form."""
        return _digest(self._content())

    def _content(self) -> dict[str, object]:
        return {
            "metric": self.metric.value,
            "version": self.version,
            "requires": [list(group) for group in self.requires],
        }

    def to_record(self) -> dict[str, object]:
        """Return the requirement with its digest, as in the parity fixture."""
        return {**self._content(), "digest": self.digest}


def _named(pairs: tuple[tuple[str, str | None], ...]) -> tuple[JointMapping, ...]:
    return tuple(JointMapping(name, canonical) for name, canonical in pairs)


# MediaPipe PoseLandmark indices 0-32 (mediapipe/python/solutions/pose.py).
_BLAZEPOSE_33: Final = JointMappingTable(
    source=JointSource.BLAZEPOSE_33,
    version=1,
    mappings=tuple(
        JointMapping(index, canonical)
        for index, canonical in enumerate(
            (
                "nose",  # 0 nose
                None,  # 1 left_eye_inner
                None,  # 2 left_eye
                None,  # 3 left_eye_outer
                None,  # 4 right_eye_inner
                None,  # 5 right_eye
                None,  # 6 right_eye_outer
                None,  # 7 left_ear
                None,  # 8 right_ear
                None,  # 9 mouth_left
                None,  # 10 mouth_right
                "left_shoulder",  # 11
                "right_shoulder",  # 12
                "left_elbow",  # 13
                "right_elbow",  # 14
                "left_wrist",  # 15
                "right_wrist",  # 16
                None,  # 17 left_pinky
                None,  # 18 right_pinky
                None,  # 19 left_index
                None,  # 20 right_index
                None,  # 21 left_thumb
                None,  # 22 right_thumb
                "left_hip",  # 23
                "right_hip",  # 24
                "left_knee",  # 25
                "right_knee",  # 26
                "left_ankle",  # 27
                "right_ankle",  # 28
                "left_heel",  # 29
                "right_heel",  # 30
                "left_foot_index",  # 31
                "right_foot_index",  # 32
            )
        )
    ),
)

# HumanBodyPose3DObservation.JointName (VNDetectHumanBodyPose3DRequest).
_APPLE_VISION_3D: Final = JointMappingTable(
    source=JointSource.APPLE_VISION_3D,
    version=1,
    mappings=_named(
        (
            ("root", "pelvis"),
            ("spine", "spine"),
            ("centerShoulder", "neck"),
            ("centerHead", "head"),
            ("topHead", None),
            ("leftShoulder", "left_shoulder"),
            ("rightShoulder", "right_shoulder"),
            ("leftElbow", "left_elbow"),
            ("rightElbow", "right_elbow"),
            ("leftWrist", "left_wrist"),
            ("rightWrist", "right_wrist"),
            ("leftHip", "left_hip"),
            ("rightHip", "right_hip"),
            ("leftKnee", "left_knee"),
            ("rightKnee", "right_knee"),
            ("leftAnkle", "left_ankle"),
            ("rightAnkle", "right_ankle"),
        )
    ),
)

# HumanBodyPoseObservation.JointName (VNDetectHumanBodyPoseRequest, 2D).
_APPLE_VISION_2D: Final = JointMappingTable(
    source=JointSource.APPLE_VISION_2D,
    version=1,
    mappings=_named(
        (
            ("nose", "nose"),
            ("leftEye", None),
            ("rightEye", None),
            ("leftEar", None),
            ("rightEar", None),
            ("neck", "neck"),
            ("root", "pelvis"),
            ("leftShoulder", "left_shoulder"),
            ("rightShoulder", "right_shoulder"),
            ("leftElbow", "left_elbow"),
            ("rightElbow", "right_elbow"),
            ("leftWrist", "left_wrist"),
            ("rightWrist", "right_wrist"),
            ("leftHip", "left_hip"),
            ("rightHip", "right_hip"),
            ("leftKnee", "left_knee"),
            ("rightKnee", "right_knee"),
            ("leftAnkle", "left_ankle"),
            ("rightAnkle", "right_ankle"),
        )
    ),
)

JOINT_TABLES: Final[Mapping[JointSource, JointMappingTable]] = MappingProxyType(
    {
        table.source: table
        for table in (_BLAZEPOSE_33, _APPLE_VISION_3D, _APPLE_VISION_2D)
    }
)


def _pairs(*names: str) -> tuple[tuple[str, ...], ...]:
    return tuple((f"left_{name}",) for name in names) + tuple(
        (f"right_{name}",) for name in names
    )


METRIC_REQUIREMENTS: Final[Mapping[PoseMetric, MetricJointRequirement]] = (
    MappingProxyType(
        {
            # Heel-strike and toe-off candidates from foot trajectories,
            # referenced to the hips (#4682). The heel is preferred; sources
            # without heels (Apple Vision) fall back to the ankle.
            PoseMetric.GAIT_TIMING: MetricJointRequirement(
                PoseMetric.GAIT_TIMING,
                1,
                _pairs("hip")
                + (("left_heel", "left_ankle"), ("right_heel", "right_ankle")),
            ),
            # Hip (shoulder-hip-knee) and knee (hip-knee-ankle) angles (#4683).
            PoseMetric.SIT_TO_STAND: MetricJointRequirement(
                PoseMetric.SIT_TO_STAND,
                1,
                _pairs("shoulder", "hip", "knee", "ankle"),
            ),
            # Trunk lean, shoulder and pelvic tilt, and head-forward offset
            # from either a head or a nose point (#4684).
            PoseMetric.POSTURE: MetricJointRequirement(
                PoseMetric.POSTURE,
                1,
                _pairs("shoulder", "hip") + (("head", "nose"),),
            ),
        }
    )
)


def _table(source: object) -> JointMappingTable:
    member = _source(source)
    if member is None:
        _reject("joint_source_invalid")
    return JOINT_TABLES[member]


def canonical_joint(source: JointSource | str, source_joint: object) -> str | None:
    """Map one source joint to its canonical name, or ``None`` if unmapped.

    Args:
        source: A ``JointSource`` or its string value.
        source_joint: A BlazePose landmark index (``int``) or an Apple Vision
            joint name (``str``).

    Raises:
        PoseContractError: ``joint_source_invalid`` for an unknown source and
            ``source_joint_unknown`` for an index or name the source does not
            define. Neither echoes the rejected value.
    """
    return _table(source).canonical_joint(source_joint)


def missing_joint_groups(
    metric: PoseMetric | str, source: JointSource | str
) -> tuple[tuple[str, ...], ...]:
    """Return the requirement groups ``source`` cannot satisfy for ``metric``.

    An empty tuple means the source's vocabulary covers the metric. A
    non-empty result is the reason a gate should abstain.

    Raises:
        PoseContractError: ``metric_invalid`` or ``joint_source_invalid``.
    """
    member = _metric(metric)
    if member is None:
        _reject("metric_invalid")
    produced = _table(source).produced
    return tuple(
        group
        for group in METRIC_REQUIREMENTS[member].requires
        if not produced.intersection(group)
    )


def resolve_metric_joints(
    metric: PoseMetric | str, source: JointSource | str
) -> tuple[str, ...]:
    """Return the joint ``source`` should use for each group of ``metric``.

    Each group resolves to its first joint the source produces, so the result
    records whether a fallback is in use: for ``gait_timing`` BlazePose
    resolves to the heels and Apple Vision to the ankles.

    Raises:
        PoseContractError: ``metric_invalid``, ``joint_source_invalid``, or
            ``metric_joints_missing`` when any group is unmet (see
            ``missing_joint_groups``).
    """
    member = _metric(metric)
    if member is None:
        _reject("metric_invalid")
    produced = _table(source).produced
    resolved = []
    for group in METRIC_REQUIREMENTS[member].requires:
        usable = [joint for joint in group if joint in produced]
        if not usable:
            _reject("metric_joints_missing")
        resolved.append(usable[0])
    return tuple(resolved)


def joint_mapping_record() -> dict[str, object]:
    """Return every table and metric requirement for the shared parity fixture.

    Swift and Kotlin ports load ``tests/fixtures/parity/pose_joints_v1.json``,
    which is this record plus rejection vectors.
    """
    return {
        "schema_version": POSE_JOINTS_SCHEMA_VERSION,
        "canonical_joints": list(CANONICAL_JOINTS),
        "tables": [table.to_record() for table in JOINT_TABLES.values()],
        "metric_requirements": [
            requirement.to_record() for requirement in METRIC_REQUIREMENTS.values()
        ],
    }
