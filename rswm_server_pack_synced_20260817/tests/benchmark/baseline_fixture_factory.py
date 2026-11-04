from __future__ import annotations

from benchmark.core.identity import (
    ArchitectureIdentity,
    CameraConditionIdentity,
    CheckpointIdentity,
    ComponentIdentity,
    ExecutionIdentity,
    ModelIdentity,
    canonical_sha256,
)
from benchmark.core.types import CapabilityPartition, TaskId, TrackId
from baselines import (
    AdaptationType,
    AdmissionState,
    BaselineCapabilities,
    BaselineCardV2,
    BaselineIdentity,
    CameraSupport,
    DisclosureStatus,
    ExternalDataDisclosure,
    InputType,
    LegacyCardReference,
    LicenseDisclosure,
    RuntimeDisclosure,
    StateEvidence,
    TrajectorySupport,
)


def digest(label: str) -> str:
    return canonical_sha256({"m5_fixture": label})


def baseline_identity(
    partition: CapabilityPartition = CapabilityPartition.NATIVE,
) -> BaselineIdentity:
    adapter_name = f"FixtureAdapter:{partition.value}"
    checkpoint = CheckpointIdentity(
        repository=f"fixture/{partition.value}",
        revision="fixture-revision",
        sha256_manifest=digest(f"{partition.value}:checkpoint-manifest"),
        config_hash=digest(f"{partition.value}:checkpoint-config"),
        status="verified",
    )
    execution = ExecutionIdentity(
        route=f"fixture-{partition.value}-route",
        entrypoint="tests/benchmark/baseline_fixture_factory.py",
    )
    model = ModelIdentity.create(
        model_family="FixtureVideoModel",
        release_identity=f"FixtureVideoModel-{partition.value}-v1",
        adapter_identity=adapter_name,
        checkpoint_identity=checkpoint,
        architecture_identity=ArchitectureIdentity(
            transformer_type="FixtureTransformer",
            vae="FixtureVAE",
            text_encoder="FixtureTextEncoder",
            camera_conditioning_type="fixture-camera",
        ),
        camera_condition_identity=CameraConditionIdentity(
            encoding="plucker_ray_v1",
            pose_injection="fixture-injection",
            injector_type="fixture-injector",
        ),
        execution_identity=execution,
        status="verified",
    )
    adapter = ComponentIdentity(
        kind="baseline_adapter",
        artifact_id=adapter_name,
        schema="rswm.baseline_adapter/v1",
        version="fixture-v1",
        sha256=digest(f"{partition.value}:adapter"),
        status="verified",
    )
    return BaselineIdentity.create(
        baseline_id=f"fixture_{partition.value}",
        model_identity=model,
        adapter_identity=adapter,
        checkpoint_identity=checkpoint,
        execution_identity=execution,
    )


def capabilities(
    partition: CapabilityPartition = CapabilityPartition.NATIVE,
) -> BaselineCapabilities:
    identity = baseline_identity(partition)
    if partition is CapabilityPartition.MULTIVIEW:
        input_type = InputType.MULTIVIEW
        adaptation = AdaptationType.FROZEN
        tracks = (TrackId.C_MULTIVIEW,)
        condition_views = 3
        camera = CameraSupport.EXTRINSICS
    elif partition is CapabilityPartition.ADAPTED:
        input_type = InputType.SINGLE_IMAGE
        adaptation = AdaptationType.PROMPT_BRIDGE
        tracks = (TrackId.B_CAMERA,)
        condition_views = 1
        camera = CameraSupport.EXTRINSICS
    else:
        input_type = InputType.SINGLE_IMAGE
        adaptation = AdaptationType.FROZEN
        tracks = (TrackId.B_CAMERA,)
        condition_views = 1
        camera = CameraSupport.PLUCKER
    return BaselineCapabilities(
        input_type=input_type,
        trajectory_support=TrajectorySupport.FULL_PATH,
        camera_support=camera,
        adaptation_type=adaptation,
        checkpoint_identity=identity.checkpoint_identity,
        license=LicenseDisclosure(
            code="Apache-2.0",
            weights="fixture-permissive",
            data="fixture-permissive",
        ),
        tasks=(TaskId.T1_SPATIAL_ROLLOUT, TaskId.T2_ARBITRARY_DIRECTION),
        tracks=tracks,
        external_data=ExternalDataDisclosure(
            status=DisclosureStatus.DECLARED,
            description="fixture data declaration",
            references=("fixture://data-card",),
        ),
        runtime=RuntimeDisclosure(
            framework="fixture-framework",
            framework_version="1.0",
            container_identity=digest(f"{partition.value}:container"),
            hardware="cpu-fixture",
            precision="float32",
        ),
        condition_views=condition_views,
        uses_future_gt=False,
        output_format="THWC_uint8",
    )


def declared_card(
    partition: CapabilityPartition = CapabilityPartition.NATIVE,
) -> BaselineCardV2:
    identity = baseline_identity(partition)
    caps = capabilities(partition)
    return BaselineCardV2.create_declared(
        baseline_identity=identity,
        capabilities=caps,
        legacy_card=LegacyCardReference(
            name=identity.baseline_id,
            sha256=digest(f"{partition.value}:legacy-card"),
            relative_path=f"fixtures/{identity.baseline_id}.json",
        ),
        capability_partition=partition,
    )


def evidence(state: AdmissionState) -> StateEvidence:
    kinds = {
        AdmissionState.ADAPTER_AVAILABLE: "adapter",
        AdmissionState.CHECKPOINT_VERIFIED: "checkpoint",
        AdmissionState.RUNNABLE: "runtime_probe",
        AdmissionState.EVALUATED: "result",
        AdmissionState.ADMITTED: "admission_decision",
    }
    return StateEvidence(
        target_state=state,
        kind=kinds[state],
        artifact_id=f"fixture:{state.value.lower()}",
        sha256=digest(f"evidence:{state.value}"),
    )


def advance_to_admitted(card: BaselineCardV2) -> BaselineCardV2:
    for state in tuple(AdmissionState)[1:]:
        card = card.advance(evidence(state))
    return card
