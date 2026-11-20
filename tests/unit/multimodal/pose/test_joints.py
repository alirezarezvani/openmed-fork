"""Tests for pose joint-vocabulary mapping tables (#4676); no coordinates."""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

import pytest

from openmed.multimodal.pose.joints import (
    JOINT_TABLES,
    METRIC_REQUIREMENTS,
    POSE_JOINTS_SCHEMA_VERSION,
    JointMapping,
    JointMappingTable,
    JointSource,
    MetricJointRequirement,
    PoseMetric,
    canonical_joint,
    joint_mapping_record,
    missing_joint_groups,
)
from openmed.multimodal.pose.sequence import CANONICAL_JOINTS, PoseContractError

FIXTURE = (
    Path(__file__).resolve().parents[3] / "fixtures" / "parity" / "pose_joints_v1.json"
)

# Source vocabularies as published: MediaPipe PoseLandmark (indices 0-32),
# Apple HumanBodyPose3DObservation.JointName (17) and
# HumanBodyPoseObservation.JointName (19).
BLAZEPOSE_INDICES = tuple(range(33))
APPLE_3D_NAMES = (
    "centerHead",
    "centerShoulder",
    "leftAnkle",
    "leftElbow",
    "leftHip",
    "leftKnee",
    "leftShoulder",
    "leftWrist",
    "rightAnkle",
    "rightElbow",
    "rightHip",
    "rightKnee",
    "rightShoulder",
    "rightWrist",
    "root",
    "spine",
    "topHead",
)
APPLE_2D_NAMES = (
    "leftAnkle",
    "leftEar",
    "leftElbow",
    "leftEye",
    "leftHip",
    "leftKnee",
    "leftShoulder",
    "leftWrist",
    "neck",
    "nose",
    "rightAnkle",
    "rightEar",
    "rightElbow",
    "rightEye",
    "rightHip",
    "rightKnee",
    "rightShoulder",
    "rightWrist",
    "root",
)
SOURCE_JOINTS = {
    JointSource.BLAZEPOSE_33: BLAZEPOSE_INDICES,
    JointSource.APPLE_VISION_3D: APPLE_3D_NAMES,
    JointSource.APPLE_VISION_2D: APPLE_2D_NAMES,
}

# Never edit a pinned digest. Changing a table or requirement means bumping
# its version and pinning the new (version, digest) pair here and in the
# regenerated parity fixture.
PINNED_TABLE_DIGESTS = {
    "blazepose_33": (
        1,
        "d1c5871c56168ebdb8254263eabce4fe9837b75f89ca392d6f1b94246c769f8e",
    ),
    "apple_vision_3d": (
        1,
        "64e3f23d3256f542d1ed73415982724ce11314741a4cedbc6fb6ffd89f13d5f4",
    ),
    "apple_vision_2d": (
        1,
        "2e79a4232086f2a0e1f328350a96989687da36b8366c5d1bc16917e38baefb51",
    ),
}
PINNED_REQUIREMENT_DIGESTS = {
    "gait_timing": (
        1,
        "a824c9edc859974ab3e3b2964cf5749029a2b443ceb09392538a194eec5336e0",
    ),
    "sit_to_stand": (
        1,
        "09de7e85dab8d04236660ad3c39ff1d7ee3f4bb3f956d4ec9c00cb4d13fb2308",
    ),
    "posture": (
        1,
        "25b31c00f94ce5c51bae1dd8dde52dc8caa254c4b37883b3219bcc12b94ae1e9",
    ),
}

REJECTIONS = (
    {"source": "blazepose_33", "source_joint": 33, "error": "source_joint_unknown"},
    {"source": "blazepose_33", "source_joint": -1, "error": "source_joint_unknown"},
    {"source": "blazepose_33", "source_joint": "0", "error": "source_joint_unknown"},
    {
        "source": "blazepose_33",
        "source_joint": "left_heel",
        "error": "source_joint_unknown",
    },
    {
        "source": "apple_vision_3d",
        "source_joint": "nose",
        "error": "source_joint_unknown",
    },
    {
        "source": "apple_vision_3d",
        "source_joint": "leftheel",
        "error": "source_joint_unknown",
    },
    {
        "source": "apple_vision_2d",
        "source_joint": "centerHead",
        "error": "source_joint_unknown",
    },
    {"source": "apple_vision_2d", "source_joint": 0, "error": "source_joint_unknown"},
    {"source": "openpose_25", "source_joint": 0, "error": "joint_source_invalid"},
)


def _code(code):
    return pytest.raises(PoseContractError, match=f"^{code}$")


def _fixture_text():
    document = {**joint_mapping_record(), "rejections": list(REJECTIONS)}
    return json.dumps(document, indent=2) + "\n"


@pytest.mark.parametrize("source", list(JointSource))
def test_every_published_source_joint_maps_exactly_once(source):
    table = JOINT_TABLES[source]
    keys = [entry.source_joint for entry in table.mappings]
    assert sorted(keys, key=str) == sorted(SOURCE_JOINTS[source], key=str)
    assert len(keys) == len(set(keys))
    for key in SOURCE_JOINTS[source]:
        mapped = canonical_joint(source, key)
        assert mapped is None or mapped in CANONICAL_JOINTS
        assert canonical_joint(source.value, key) == mapped


@pytest.mark.parametrize("source", list(JointSource))
def test_canonical_targets_are_unique_and_missing_is_the_complement(source):
    table = JOINT_TABLES[source]
    targets = [entry.canonical for entry in table.mappings if entry.canonical]
    assert len(targets) == len(set(targets))
    assert set(targets) == table.produced
    assert set(table.missing) == set(CANONICAL_JOINTS) - table.produced
    assert list(table.missing) == [j for j in CANONICAL_JOINTS if j in table.missing]


@pytest.mark.parametrize(
    ("source", "key", "expected"),
    [
        ("blazepose_33", 0, "nose"),
        ("blazepose_33", 7, None),
        ("blazepose_33", 17, None),
        ("blazepose_33", 23, "left_hip"),
        ("blazepose_33", 29, "left_heel"),
        ("blazepose_33", 32, "right_foot_index"),
        ("apple_vision_3d", "root", "pelvis"),
        ("apple_vision_3d", "centerShoulder", "neck"),
        ("apple_vision_3d", "centerHead", "head"),
        ("apple_vision_3d", "topHead", None),
        ("apple_vision_3d", "spine", "spine"),
        ("apple_vision_2d", "root", "pelvis"),
        ("apple_vision_2d", "neck", "neck"),
        ("apple_vision_2d", "leftEar", None),
        ("apple_vision_2d", "rightAnkle", "right_ankle"),
    ],
)
def test_known_mappings(source, key, expected):
    assert canonical_joint(source, key) == expected


def test_sides_are_never_swapped():
    for source, table in JOINT_TABLES.items():
        for entry in table.mappings:
            if entry.canonical is None:
                continue
            name = str(entry.source_joint).lower()
            if entry.canonical.startswith("left_"):
                assert source is JointSource.BLAZEPOSE_33 or name.startswith("left")
            if entry.canonical.startswith("right_"):
                assert source is JointSource.BLAZEPOSE_33 or name.startswith("right")
    blaze = JOINT_TABLES[JointSource.BLAZEPOSE_33]
    # MediaPipe numbers left before right in every pair from 11 to 32.
    for left in range(11, 33, 2):
        a, b = blaze.canonical_joint(left), blaze.canonical_joint(left + 1)
        if a is not None:
            assert a.startswith("left_") and b == "right_" + a[len("left_") :]


def test_declared_missing_joints_match_the_issue():
    assert JOINT_TABLES[JointSource.BLAZEPOSE_33].missing == (
        "head",
        "neck",
        "spine",
        "pelvis",
    )
    for source in (JointSource.APPLE_VISION_3D, JointSource.APPLE_VISION_2D):
        missing = JOINT_TABLES[source].missing
        for joint in ("left_heel", "right_heel", "left_foot_index", "right_foot_index"):
            assert joint in missing


@pytest.mark.parametrize("case", REJECTIONS, ids=lambda case: case["error"])
def test_unknown_source_joints_fail_closed(case):
    with _code(case["error"]):
        canonical_joint(case["source"], case["source_joint"])


@pytest.mark.parametrize(
    "key", [True, False, 1.0, 0.0, None, b"nose", ("root",), ["root"], {"x": 1}]
)
@pytest.mark.parametrize("source", list(JointSource))
def test_wrong_key_types_never_alias_a_valid_joint(source, key):
    with _code("source_joint_unknown"):
        canonical_joint(source, key)


@pytest.mark.parametrize("source", [None, 0, "BLAZEPOSE_33", "blazepose", b"x"])
def test_unknown_source_fails_closed(source):
    with _code("joint_source_invalid"):
        canonical_joint(source, 0)  # type: ignore[arg-type]


def test_rejections_do_not_echo_or_chain_the_rejected_value():
    secret = "SECRET-joint-7731"
    for call in (
        lambda: canonical_joint("apple_vision_3d", secret),
        lambda: canonical_joint(secret, 0),
        lambda: missing_joint_groups(secret, "blazepose_33"),
        lambda: missing_joint_groups("posture", secret),
    ):
        with pytest.raises(PoseContractError) as raised:
            call()
        assert secret not in str(raised.value)
        assert secret not in repr(raised.value)
        assert raised.value.__context__ is None
        assert raised.value.__cause__ is None


def test_seeded_random_lookups_never_raise_anything_but_the_contract_error():
    rng = random.Random(4676)
    for _ in range(2000):
        source = rng.choice([*JointSource, "openpose_25"])
        key = rng.choice(
            [
                rng.randint(-50, 80),
                "".join(rng.choice("leftRightAnkleroot_0") for _ in range(8)),
                rng.random(),
                None,
            ]
        )
        try:
            result = canonical_joint(source, key)
        except PoseContractError as error:
            assert str(error) in {"source_joint_unknown", "joint_source_invalid"}
        else:
            assert result is None or result in CANONICAL_JOINTS
            assert key in SOURCE_JOINTS[JointSource(source)]


def test_metric_requirements_cover_every_metric_with_canonical_joints():
    assert set(METRIC_REQUIREMENTS) == set(PoseMetric)
    for metric, requirement in METRIC_REQUIREMENTS.items():
        assert requirement.metric is metric
        joints = [joint for group in requirement.requires for joint in group]
        assert set(joints) <= set(CANONICAL_JOINTS)
        assert len(joints) == len(set(joints))


@pytest.mark.parametrize(
    ("metric", "source", "expected"),
    [
        ("gait_timing", "blazepose_33", ()),
        ("gait_timing", "apple_vision_3d", (("left_heel",), ("right_heel",))),
        ("gait_timing", "apple_vision_2d", (("left_heel",), ("right_heel",))),
        ("sit_to_stand", "blazepose_33", ()),
        ("sit_to_stand", "apple_vision_3d", ()),
        ("sit_to_stand", "apple_vision_2d", ()),
        ("posture", "blazepose_33", ()),
        ("posture", "apple_vision_3d", ()),
        ("posture", "apple_vision_2d", ()),
    ],
)
def test_missing_joint_groups_drive_abstention(metric, source, expected):
    assert missing_joint_groups(metric, source) == expected
    assert missing_joint_groups(PoseMetric(metric), JointSource(source)) == expected


def test_alternative_group_needs_only_one_joint():
    groups = METRIC_REQUIREMENTS[PoseMetric.POSTURE].requires
    assert ("head", "nose") in groups
    # BlazePose has a nose but no head; Apple 3D has a head but no nose.
    assert "head" in JOINT_TABLES[JointSource.BLAZEPOSE_33].missing
    assert "nose" in JOINT_TABLES[JointSource.APPLE_VISION_3D].missing


@pytest.mark.parametrize("metric", [None, "fall_risk", "GAIT_TIMING", 1])
def test_unknown_metric_fails_closed(metric):
    with _code("metric_invalid"):
        missing_joint_groups(metric, "blazepose_33")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source_joint", "canonical"),
    [
        (-1, "nose"),
        (True, "nose"),
        (1.0, "nose"),
        ("", "nose"),
        ("1abc", "nose"),
        ("left heel", "nose"),
        ("a" * 65, "nose"),
        (0, "left_pinky"),
        (0, ""),
        (0, 1),
    ],
)
def test_joint_mapping_validates_its_fields(source_joint, canonical):
    with _code("mapping_table_invalid"):
        JointMapping(source_joint, canonical)


def _blaze_mappings():
    return list(JOINT_TABLES[JointSource.BLAZEPOSE_33].mappings)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m[:-1] + [JointMapping(40, None)],  # non-contiguous indices
        lambda m: m + [JointMapping(32, None)],  # duplicate index
        lambda m: m + [JointMapping("leftHeel", None)],  # mixed key types
        lambda m: m[:1] + [JointMapping(1, "nose")] + m[2:],  # duplicate target
        lambda m: m + ["not-a-mapping"],
        lambda m: [],
    ],
)
def test_mapping_table_rejects_inconsistent_tables(mutate):
    with _code("mapping_table_invalid"):
        JointMappingTable(JointSource.BLAZEPOSE_33, 1, tuple(mutate(_blaze_mappings())))


@pytest.mark.parametrize("version", [0, -1, True, 1.0, "1", None])
def test_versions_must_be_positive_integers(version):
    with _code("mapping_table_invalid"):
        JointMappingTable(JointSource.BLAZEPOSE_33, version, tuple(_blaze_mappings()))
    with _code("metric_requirement_invalid"):
        MetricJointRequirement(PoseMetric.POSTURE, version, (("nose",),))


@pytest.mark.parametrize(
    "requires",
    [
        (),
        ((),),
        (("pelvis",), ("pelvis",)),
        (("left_pinky",),),
        (("nose", 1),),
        ("nose",),
        None,
    ],
)
def test_metric_requirement_validates_groups(requires):
    with _code("metric_requirement_invalid"):
        MetricJointRequirement(PoseMetric.POSTURE, 1, requires)


def test_tables_and_requirements_are_immutable():
    table = JOINT_TABLES[JointSource.BLAZEPOSE_33]
    with pytest.raises(dataclasses.FrozenInstanceError):
        table.version = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        JOINT_TABLES[JointSource.BLAZEPOSE_33] = table  # type: ignore[index]
    with pytest.raises(TypeError):
        METRIC_REQUIREMENTS[PoseMetric.POSTURE] = None  # type: ignore[index]


@pytest.mark.parametrize("source", list(JointSource))
def test_table_digest_is_pinned_to_its_version(source):
    table = JOINT_TABLES[source]
    assert (table.version, table.digest) == PINNED_TABLE_DIGESTS[source.value], (
        "Mapping table changed: bump its version and pin the new digest."
    )


@pytest.mark.parametrize("metric", list(PoseMetric))
def test_requirement_digest_is_pinned_to_its_version(metric):
    requirement = METRIC_REQUIREMENTS[metric]
    assert (requirement.version, requirement.digest) == (
        PINNED_REQUIREMENT_DIGESTS[metric.value]
    ), "Metric requirement changed: bump its version and pin the new digest."


def test_changing_a_mapping_without_a_version_bump_changes_the_digest():
    table = JOINT_TABLES[JointSource.APPLE_VISION_3D]
    edited = [
        JointMapping("topHead", "head") if entry.source_joint == "topHead" else entry
        for entry in table.mappings
        if entry.source_joint != "centerHead"
    ] + [JointMapping("centerHead", None)]
    changed = JointMappingTable(table.source, table.version, tuple(edited))
    assert changed.digest != PINNED_TABLE_DIGESTS[table.source.value][1]
    reordered = JointMappingTable(
        table.source, table.version, tuple(reversed(table.mappings))
    )
    assert reordered.digest != table.digest


def test_parity_fixture_matches_the_tables_byte_for_byte():
    assert FIXTURE.read_text(encoding="utf-8") == _fixture_text(), (
        "Regenerate tests/fixtures/parity/pose_joints_v1.json from "
        "joint_mapping_record() plus REJECTIONS."
    )
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert document["schema_version"] == POSE_JOINTS_SCHEMA_VERSION
    assert document["canonical_joints"] == list(CANONICAL_JOINTS)
    for record in document["tables"]:
        table = JOINT_TABLES[JointSource(record["source"])]
        assert record["digest"] == table.digest
        rebuilt = JointMappingTable(
            JointSource(record["source"]),
            record["version"],
            tuple(
                JointMapping(entry["source_joint"], entry["canonical"])
                for entry in record["mappings"]
            ),
        )
        assert rebuilt == table
        assert list(rebuilt.missing) == record["missing"]


def test_fixture_contains_no_coordinates_or_free_text():
    text = FIXTURE.read_text(encoding="utf-8")
    for forbidden in ('"x"', '"y"', '"z"', "visibility", "presence", "/", "\\"):
        assert forbidden not in text
