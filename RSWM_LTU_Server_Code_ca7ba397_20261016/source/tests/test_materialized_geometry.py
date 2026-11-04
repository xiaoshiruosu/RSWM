"""A second geometry algorithm checks persisted labels pixel for pixel."""
from copy import deepcopy

import numpy as np
import pytest

from rswm_inria import geometry, tasks
from rswm_inria.materialized_geometry import audit_task_geometry, independent_target_labels


@pytest.mark.parametrize("origin_index", range(4))
@pytest.mark.parametrize("world_id", ["audit-random-8675309", "audit-random-314159"])
def test_all_families_and_all_target_pixels_agree(world_id, origin_index):
    for task, private in tasks.packages(world_id, "train", origin_index):
        computed = audit_task_geometry(task, private)
        assert set(computed) == set(private)
        for stream, labels in private.items():
            for field, expected in labels.items():
                assert computed[stream][field].shape == (128, 128)
                np.testing.assert_array_equal(computed[stream][field], expected)
            assert computed[stream]["first_event"].min() >= 1
            assert computed[stream]["invisible_query_gap"].min() >= 32
            assert computed[stream]["reentry_event"].max() < 97


@pytest.fixture(scope="module")
def persistence_tasks():
    return {task["family"]: (task, private)
            for task, private in tasks.packages("audit-corruption-271828", "formal", 2)
            if private}


@pytest.mark.parametrize("family", ["F5", "F7"])
@pytest.mark.parametrize("field", ["first_event", "last_prefix_visible_event", "reentry_event",
                                    "invisible_query_gap", "fully_absent_calls", "x", "y"])
def test_one_corrupted_pixel_is_rejected(persistence_tasks, family, field):
    task, private = persistence_tasks[family]
    bad = deepcopy(private)
    stream = next(iter(bad))
    bad[stream][field][127, 0] += 1
    with pytest.raises(ValueError, match="stored labels disagree"):
        audit_task_geometry(task, bad)


def test_does_not_delegate_to_generator_or_frozen_validator(persistence_tasks, monkeypatch):
    task, private = persistence_tasks["F7"]

    def forbidden(*args, **kwargs):
        raise AssertionError("The independent audit delegated to the generator.")

    monkeypatch.setattr(tasks, "target_labels", forbidden)
    monkeypatch.setattr(tasks, "validate_task", forbidden)
    monkeypatch.setattr(geometry, "coordinates", forbidden)
    monkeypatch.setattr(geometry, "validate_path", forbidden)
    assert set(audit_task_geometry(task, private)) == {"a", "b"}


@pytest.mark.parametrize("kind", ["missing_stream", "missing_field", "wrong_shape", "float_dtype"])
def test_stored_label_contract_is_complete(persistence_tasks, kind):
    task, private = persistence_tasks["F7"]
    bad = deepcopy(private)
    if kind == "missing_stream":
        del bad["b"]
    elif kind == "missing_field":
        del bad["a"]["fully_absent_calls"]
    elif kind == "wrong_shape":
        bad["a"]["x"] = bad["a"]["x"][:-1]
    else:
        bad["a"]["x"] = bad["a"]["x"].astype(float)
    with pytest.raises(ValueError, match="stored"):
        audit_task_geometry(task, bad)


@pytest.mark.parametrize("kind", ["q0", "float", "bounds", "overlap", "prefix", "target", "origin"])
def test_invalid_f7_geometry_is_rejected(persistence_tasks, kind):
    task, private = persistence_tasks["F7"]
    bad = deepcopy(task)
    if kind == "q0":
        bad["programs"]["a"][0] = [1, 0]
    elif kind == "float":
        bad["programs"]["a"][10][0] = float(bad["programs"]["a"][10][0])
    elif kind == "bounds":
        bad["programs"]["a"][96][0] = 705
    elif kind == "overlap":
        bad["programs"]["a"][95] = [704, 704]
    elif kind == "prefix":
        bad["programs"]["a"][5][0] += 1
    elif kind == "target":
        bad["metadata"]["target_center"][0] += 1
    else:
        bad["origin"][0] += 1
    with pytest.raises(ValueError):
        audit_task_geometry(bad, private)


def test_suffix_without_reentry_and_initial_exposure_are_rejected(persistence_tasks):
    task, _ = persistence_tasks["F5"]
    path = np.asarray(task["programs"]["main"])
    no_return = path.copy()
    no_return[33:] = path[32]
    with pytest.raises(ValueError, match="suffix reentry"):
        independent_target_labels(no_return, task["metadata"]["target_center"])
    with pytest.raises(ValueError, match="visible in I0"):
        independent_target_labels(path, [0, 0])


def test_gap_cannot_be_relaxed(persistence_tasks):
    task, _ = persistence_tasks["F5"]
    with pytest.raises(ValueError, match="cannot weaken"):
        independent_target_labels(task["programs"]["main"], task["metadata"]["target_center"], min_gap=31)


def test_actual_short_invisibility_gap_is_rejected(persistence_tasks):
    task, _ = persistence_tasks["F5"]
    path = np.asarray(task["programs"]["main"], dtype=np.int64).copy()
    start = path[32].copy()
    target = np.asarray(task["metadata"]["target_center"], dtype=np.int64)
    path[33:49] = np.rint(start + (target - start) * np.arange(1, 17)[:, None] / 16).astype(np.int64)
    path[49:] = target
    with pytest.raises(ValueError, match="invisibility gap is too short"):
        independent_target_labels(path, target)


def test_f3_pair_requires_exact_integer_double():
    task, private = tasks.packages("audit-f3-123456", "motion_calibrate", 0)[1]
    bad = deepcopy(task)
    bad["programs"]["large"][20][0] += 1
    with pytest.raises(ValueError, match="exact integer 2x"):
        audit_task_geometry(bad, private)


def test_unsigned_out_of_range_cannot_wrap(persistence_tasks):
    task, _ = persistence_tasks["F5"]
    path = np.asarray(task["programs"]["main"], dtype=np.int64).astype(np.uint64)
    path[1, 0] = np.iinfo(np.uint64).max
    with pytest.raises(ValueError, match="limits"):
        independent_target_labels(path, task["metadata"]["target_center"])
