"""Actual NumPy conditions and payload isolation; no real backend claims."""
from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from rswm_inria.adapters.camera_conditioning import CameraConfig, build_camera_batch, backend_payload
from rswm_inria.geometry import digest


def queries(scale=1):
    return np.arange(1, 17)[:, None] * [12 * scale, 0]


@pytest.mark.parametrize("channel", ["drop", "identity"])
def test_ablation_has_no_query_trace_and_backend_observes_same_conditions(channel):
    cfg = CameraConfig(camera_channel=channel)
    a = build_camera_batch(queries(), config=cfg)
    b = build_camera_batch(queries(2), config=cfg)
    assert a.native_queries_xy is None and b.native_queries_xy is None
    observed = []
    def backend_spy(**conditions):
        observed.append(conditions)
    for batch in (a, b):
        backend_spy(**backend_payload(batch))
    assert set(observed[0]) == {"camera_poses", "intrinsics_fx_fy_cx_cy", "selected_event_indices",
                                "camera_manifest", "camera_manifest_sha256"}
    for field in ("camera_poses", "intrinsics_fx_fy_cx_cy", "selected_event_indices"):
        if observed[0][field] is None:
            assert observed[1][field] is None
        else:
            np.testing.assert_array_equal(observed[0][field], observed[1][field])
    assert observed[0]["camera_manifest"] == observed[1]["camera_manifest"]


def test_native_payload_whitelist_excludes_trace_and_binds_shared_manifest():
    cfg = CameraConfig(framewise=True, temporal_stride=2)
    batch = build_camera_batch(queries(), config=cfg)
    payload = backend_payload(batch, expected_config_sha256=digest(cfg.manifest()))
    assert "native_queries_xy" not in payload
    assert payload["camera_manifest_sha256"] == digest(payload["camera_manifest"])
    assert payload["camera_manifest"]["temporal_alignment"] == "anchor_then_current_call_queries_sample_before_delta"
    assert payload["camera_manifest"]["canonical_to_native"] == "Pillow_BILINEAR_2x"
    assert payload["camera_manifest"]["backend_validation"] == "NOT_RUN"
    with pytest.raises(ValueError, match="hash mismatch"):
        backend_payload(batch, expected_config_sha256="0" * 64)


def test_batch_conditions_and_config_cannot_change_after_hash_binding():
    q = queries()
    batch = build_camera_batch(q)
    q[:] = 0
    assert np.any(batch.native_queries_xy)
    with pytest.raises(FrozenInstanceError):
        batch.config = CameraConfig(camera_channel="drop")
    for a in (batch.native_queries_xy, batch.camera_poses, batch.selected_event_indices,
              batch.intrinsics_fx_fy_cx_cy):
        with pytest.raises(ValueError):
            a.setflags(write=True)
        with pytest.raises(ValueError):
            a.flat[0] = 10


def test_stride4_short_loop_alias_is_preserved_and_stride1_exposes_motion():
    loop = np.tile([[16, 0], [16, 16], [0, 16], [0, 0]], (4, 1))
    still = np.zeros((16, 2), dtype=int)
    for framewise in (False, True):
        cfg = CameraConfig(framewise=framewise)
        a = backend_payload(build_camera_batch(loop, config=cfg))
        b = backend_payload(build_camera_batch(still, config=cfg))
        np.testing.assert_array_equal(a["camera_poses"], b["camera_poses"])
        fine = CameraConfig(framewise=framewise, temporal_stride=1)
        assert not np.array_equal(build_camera_batch(loop, config=fine).camera_poses,
                                  build_camera_batch(still, config=fine).camera_poses)
