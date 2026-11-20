# Pose keypoint sequence contract

`openmed.multimodal.pose.sequence` validates body landmarks that a caller's
pose estimator produced for sampled video frames. It reads no files, runs no
estimator, makes no network calls and computes no metric. Estimators, gait and
posture metrics and biometric session handling are separate components that
build on it; mapping estimator joint vocabularies onto the canonical joints is
covered [below](#joint-vocabulary-mapping). The schema is
`openmed.multimodal.pose.sequence.v1`.

Every report carries this boundary:

> Pose landmarks only; not gait, balance, fall-risk or clinical assessment.
> Consequential use requires explicit reviewer confirmation.

## Canonical representation

- `CANONICAL_JOINTS` is a closed vocabulary of 21 body joints: `nose`,
  `head`, `neck`, `spine`, `pelvis`, and left and right `shoulder`, `elbow`,
  `wrist`, `hip`, `knee`, `ankle`, `heel` and `foot_index`. Estimator-specific
  names (BlazePose landmark indices, Apple Vision joint names) must be mapped
  before admission; unknown names fail as `unknown_joint`.
- `PoseJointObservation` holds one joint: normalized image `x` (0 at the left
  edge, 1 at the right) and `y` (0 at the top, 1 at the bottom), `visibility`
  and `presence` from 0 to 1, and optional depth `z`. Out-of-range or
  non-finite values fail as `coordinate_invalid`, `confidence_invalid` or
  `depth_invalid`; nothing is clipped.
- `PoseFrame` is keyed by an integer tick of a `FrameSamplingManifest`. A joint
  the estimator did not return is left out, never filled with placeholder
  coordinates; a second observation of the same joint fails as
  `duplicate_joint`. Joints are stored in canonical order.
- `PoseSequence` is bound to one validated manifest and keeps its
  `coverage_digest`. Every frame tick must be one of the manifest's selected
  ticks (`tick_not_in_manifest`), in strictly increasing order. A selected
  frame without a detected person may be omitted or given an empty frame.
- `depth_kind` declares what `z` means: `none` forbids depth, `relative`
  allows estimator-scaled depth up to 10 in magnitude, and `metric` allows
  metres from the estimator's origin, up to 100 in magnitude.
- `PoseEstimatorProvenance` is digest-only: a lowercase estimator slug, the
  SHA-256 of the model or library revision and the SHA-256 of the licence or
  terms accepted for it. Paths, device names and free text are refused.
- A sequence holds at most 100,000 frames and 1,000,000 joint observations.

## Synthetic example

```python
import hashlib

from openmed.multimodal.frame_sampling_manifest import (
    FrameSamplingManifest,
    SamplingMode,
    canonical_coverage_digest,
)
from openmed.multimodal.pose.sequence import (
    PoseEstimatorProvenance,
    PoseFrame,
    PoseJointObservation,
    PoseSequence,
)

ticks = (0, 10)
manifest = FrameSamplingManifest(
    duration_ticks=30,
    time_base_num=1,
    time_base_den=30,
    sampling_mode=SamplingMode.UNIFORM,
    timestamps=ticks,
    step_ticks=10,
    coverage_digest=canonical_coverage_digest(SamplingMode.UNIFORM, 1, 30, ticks),
)
digest = hashlib.sha256(b"synthetic").hexdigest()
sequence = PoseSequence(
    manifest=manifest,
    frames=(
        PoseFrame(0, (PoseJointObservation("left_hip", 0.31, 0.52, 0.98, 1.0),)),
        PoseFrame(10, ()),
    ),
    depth_kind="none",
    provenance=PoseEstimatorProvenance("synthetic-walker", digest, digest),
)
assert sequence.to_report()["missing_joint_counts"]["left_hip"] == 1
```

## Joint vocabulary mapping

`openmed.multimodal.pose.joints` holds versioned tables (schema
`openmed.multimodal.pose.joints.v1`) that map each estimator's joints onto
`CANONICAL_JOINTS`. It is data only: it runs no estimator and never sees a
coordinate.

| Source | Source joints | Canonical joints it cannot produce |
| --- | --- | --- |
| `blazepose_33`: MediaPipe BlazePose landmark indices 0 to 32 | 33 | `head`, `neck`, `spine`, `pelvis` |
| `apple_vision_3d`: Apple Vision `HumanBodyPose3DObservation.JointName` | 17 | `nose`, both heels, both foot indices |
| `apple_vision_2d`: Apple Vision `HumanBodyPoseObservation.JointName` | 19 | `head`, `spine`, both heels, both foot indices |

- Every source joint maps to exactly one canonical joint or to `None`
  (unmapped). BlazePose face and hand landmarks, Apple's `topHead`, eyes and
  ears are unmapped. Apple's `root` maps to `pelvis`, `centerShoulder` and the
  2D `neck` to `neck`, and `centerHead` to `head`.
- A canonical joint the source cannot produce is declared missing in the
  table's `missing` list. Leave it out of the frame; do not derive a
  placeholder such as a hip midpoint for `pelvis`.
- `canonical_joint(source, source_joint)` takes a BlazePose index (`int`) or
  an Apple joint name (`str`, the Swift property name such as `leftAnkle`). An
  unknown source fails as `joint_source_invalid`; an out-of-range index, an
  unknown name or a wrong key type (for example `True`, `1.0` or `"0"`) fails
  as `source_joint_unknown`. Errors never echo the rejected value.
- A mapping identifies the same joint by name. Estimators still place joints
  at different anatomical positions, so a metric must not mix sources within
  one sequence; the sequence's estimator provenance records which one ran.
- Each table and requirement has an integer `version` and a SHA-256 `digest`
  of its canonical JSON. Tests pin each version to its digest, so any change
  needs a version bump.

`METRIC_REQUIREMENTS` declares the canonical joints each metric family needs,
as groups where at least one joint of every group must be producible:

| Metric | Requires |
| --- | --- |
| `gait_timing` | both hips, ankles and heels |
| `sit_to_stand` | both shoulders, hips, knees and ankles |
| `posture` | both shoulders and hips, and `head` or `nose` |

`missing_joint_groups(metric, source)` returns the unmet groups, empty when the
source's vocabulary covers the metric. Both Apple Vision sources lack heels, so
`gait_timing` abstains for them. This checks vocabulary only; per-frame
visibility and camera view remain the metric's job.

The tables, requirements, digests and rejection vectors are exported as
`tests/fixtures/parity/pose_joints_v1.json` for the Swift and Kotlin ports.

```python
from openmed.multimodal.pose.joints import canonical_joint, missing_joint_groups

assert canonical_joint("blazepose_33", 29) == "left_heel"
assert canonical_joint("apple_vision_3d", "root") == "pelvis"
assert canonical_joint("apple_vision_3d", "topHead") is None
assert missing_joint_groups("gait_timing", "apple_vision_2d") == (
    ("left_heel",),
    ("right_heel",),
)
```

## Privacy and review boundary

Landmark sequences are biometric data; validation is not de-identification.
`repr`, `to_report()` and `to_json()` carry only the schema version, notice,
frame count, the number of distinct joints observed, per-joint missing counts
and `review_required: true`. Errors are `PoseContractError` with a stable code
only and never echo a coordinate. An invalid embedded manifest raises the
manifest's own value-free `FrameSamplingManifestError`.

`to_record()` and `from_record()` carry the coordinates. They exist for
caller-owned storage and for the shared parity fixture
`tests/fixtures/parity/pose_sequence_v1.json`, which later Swift and Kotlin
ports load; never log or transmit a record.

`require_reviewer_confirmation()` fails closed unless the caller passes
`confirmed=True` after explicit review. Confirmation records a caller decision;
it does not qualify the estimator, the landmarks or any metric derived from
them.

All tests and fixtures use synthetic values. No dataset, model or dependency is
bundled.
