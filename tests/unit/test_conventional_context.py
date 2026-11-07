import copy
import pickle
from dataclasses import FrozenInstanceError, replace
from itertools import combinations
from math import comb

import numpy as np
import pytest

from atprl.baselines.round_robin import RoundRobinScheduler
from atprl.envs.leo_sat import LeoSatEnv
from atprl.envs.actuator import physical_actuator_state
from atprl.solvers.optimization_context import snapshot_from_env, candidate_terms
from atprl.solvers.mrt_initialization import initialize_grouped_mrt


def environment(n=4, k=2):
    env = LeoSatEnv(
        dict(
            num_ground_stations=n,
            num_gs_per_rb=k,
            num_resource_blocks=1,
            random_seed=90009,
            use_fixed_positions=True,
            topology_profile=f"paper_n{n}",
            ris_num_elem_x=16,
            ris_num_elem_y=16,
            ris_power_control_mode="power_budget",
            ris_reflection_power_limit_dBm=-5,
            csi_error_variance=0.01,
            max_steps_per_episode=8,
        )
    )
    env.reset(seed=90009)
    return env


@pytest.mark.parametrize("n,k", [(4, 2), (5, 2), (20, 2), (20, 5), (13, 4), (4, 1), (4, 4)])
def test_rr_balance_and_resume(n, k):
    rr = RoundRobinScheduler(n, k)
    counts = np.zeros(n, dtype=int)
    assert rr.cycle_length == comb(n, k)
    for cycle in range(3):
        seen = set()
        for _ in range(rr.cycle_length):
            state = rr.state_dict()
            users = rr.peek()
            assert rr.state_dict() == state
            assert len(set(users)) == k and tuple(sorted(users)) == users
            assert users not in seen
            seen.add(users)
            counts[list(users)] += 1
            rr.commit()
        assert seen == set(combinations(range(n), k))
        assert np.all(counts == (cycle + 1) * comb(n - 1, k - 1))
    clone = RoundRobinScheduler(n, k)
    clone.load_state_dict(rr.state_dict())
    assert clone.peek() == rr.peek()
    with pytest.raises(ValueError):
        clone.load_state_dict(dict(rr.state_dict(), pointer=rr.pointer + 1))
    clone.reset()
    assert clone.peek() == tuple(range(k))


def test_n4k2_requested_six_pair_order_and_wrap():
    rr = RoundRobinScheduler(4, 2)
    expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    actual = []
    for _ in range(12):
        actual.append(rr.peek())
        rr.commit()
    assert actual == expected * 2
    from atprl.training.periodic import PERIODIC_PAIR_CYCLE_12_4U

    assert set(expected) == set(PERIODIC_PAIR_CYCLE_12_4U)


@pytest.mark.parametrize("offset", [0, 1, 5])
def test_combination_offset_and_resume_at_every_cycle_position(offset):
    expected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    rr = RoundRobinScheduler(4, 2, offset)
    for decision in range(13):
        clone = RoundRobinScheduler(4, 2, offset)
        clone.load_state_dict(rr.state_dict())
        assert clone.peek() == rr.peek() == expected[(offset + decision) % 6]
        clone.commit()
        rr.commit()
        assert clone.state_dict() == rr.state_dict()
    rr.reset()
    assert rr.peek() == expected[offset] and rr.decisions == 0


def test_old_scheduler_state_and_corruption_rejected_without_mutation():
    rr = RoundRobinScheduler(4, 2)
    before = rr.state_dict()
    old = dict(n=4, k=2, offset=0, pointer=0, decisions=6)
    bad = [
        old,
        dict(before, schedule_kind="adjacent_user_blocks"),
        dict(before, cycle_length=2),
        dict(before, decisions=True),
        dict(before, pointer=True),
        dict(before, decisions=-1),
        dict(before, pointer=1),
    ]
    for state in bad:
        with pytest.raises(ValueError):
            rr.load_state_dict(state)
        assert rr.state_dict() == before


@pytest.mark.parametrize("offset", [-1, 6, True, 1.5])
def test_invalid_combination_offsets(offset):
    with pytest.raises(ValueError):
        RoundRobinScheduler(4, 2, offset)


def test_snapshot_immutable_and_hidden_information_isolated():
    env = environment()
    actuator = physical_actuator_state(env)
    before = pickle.dumps(env.__dict__)
    first = snapshot_from_env(env, (0, 1), actuator)
    assert before == pickle.dumps(env.__dict__)
    with pytest.raises(FrozenInstanceError):
        first.pmax = 3
    with pytest.raises(ValueError):
        first.direct_hat.setflags(write=True)
    for name in (
        "h_sg_true",
        "h_sr_true",
        "h_rg_true",
        "small_scale_coeffs_sat_to_gs",
        "small_scale_sat_to_ris",
        "small_scale_ris_to_gs",
        "small_scale_coeffs_bs_to_gs",
        "_fixed_shadowing_db",
        "large_scale_gain_bs_to_gs",
    ):
        setattr(
            env,
            name,
            np.asarray(getattr(env, name)) * 7 + 2j
            if np.iscomplexobj(getattr(env, name))
            else np.asarray(getattr(env, name)) * 7,
        )
    second = snapshot_from_env(env, (0, 1), actuator)
    for name in ("direct_hat", "cascade_hat", "error_variance", "base_noise"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    np.testing.assert_array_equal(initialize_grouped_mrt(first), initialize_grouped_mrt(second))
    assert not any("true" in name or "perfect" in name for name in first.__dataclass_fields__)


def test_actuator_matches_existing_physics_without_side_effects():
    env = environment()
    before = pickle.dumps(env.__dict__)
    actuator = physical_actuator_state(env)
    assert before == pickle.dumps(env.__dict__)
    env._apply_ris_phase_action(np.zeros(env.ris_num_groups), actuator.group_weights)
    for i, group in enumerate(env.ris_groups):
        np.testing.assert_allclose(
            abs(env.ris_phase_response[group]), actuator.group_amplitudes[i], rtol=1e-14
        )


def test_mrt_zero_cases_and_candidate_order():
    env = environment()
    snap = snapshot_from_env(env, (3, 0), physical_actuator_state(env))
    th = initialize_grouped_mrt(snap)
    assert np.all(np.isfinite(th)) and np.all(th >= -np.pi) and np.all(th < np.pi)
    np.testing.assert_array_equal(th, initialize_grouped_mrt(snap))
    zero = replace(snap, direct_hat=np.zeros(2), cascade_hat=np.zeros((2, 4)))
    np.testing.assert_array_equal(initialize_grouped_mrt(zero), np.zeros(4))
    out = candidate_terms(snap, [snap.pmax, 0, 0], th)
    assert np.isfinite(out["objective"])
    with pytest.raises(ValueError):
        candidate_terms(snap, [np.nan, 0, 0], th)
