"""Headless checks for the keyboard teleop, using synthetic key events.

Run from the project root:  python tests/test_teleop.py   (or: pytest tests)
"""
import sys
import pathlib
from unittest.mock import patch

import numpy as np
import pybullet as p

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from admittance import PointForce  # noqa: E402
from force_input import ForceInput  # noqa: E402
from keyboard_teleop import PLACE_SPOTS, CONTROL_SUBSTEPS, KinovaTeleop, build_scene, parse_keys  # noqa: E402
from pick_and_place import PickAndPlace  # noqa: E402
from rotations import quat_conj, quat_mul, rotvec_from_quat  # noqa: E402

DT = 1.0 / 240.0
HELD = p.KEY_IS_DOWN
TAPPED = p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED


def make():
    if p.isConnected():
        p.disconnect()
    pb, kinova, _ = build_scene(gui=False, dt=DT)
    return kinova, KinovaTeleop(kinova, DT)


def make_with_task():
    """Like the GUI: P-key task wired up, control at 60 Hz over 240 Hz physics."""
    if p.isConnected():
        p.disconnect()
    pb, kinova, cube = build_scene(gui=False, dt=DT)
    teleop = KinovaTeleop(
        kinova, DT * CONTROL_SUBSTEPS, task_factory=lambda t: PickAndPlace(t, cube, PLACE_SPOTS)
    )
    return kinova, teleop, cube


def tick(teleop, keys={}, load=None):
    teleop.apply(parse_keys(keys), load)
    for _ in range(CONTROL_SUBSTEPS):
        if teleop.active_load is not None:
            teleop.active_load.apply(teleop.robot)
        p.stepSimulation()


def press_p_and_wait(teleop, timeout=30.0):
    """Press P, then run until the task and the drive home afterwards are done."""
    tick(teleop, {ord("p"): TAPPED})
    for _ in range(int(timeout / (DT * CONTROL_SUBSTEPS))):
        if teleop.task is None and not teleop.homing:
            return
        tick(teleop)
    raise AssertionError(f"task still running after {timeout} s")


def cube_pose(cube):
    pos, orn = p.getBasePositionAndOrientation(cube)
    return np.array(pos), np.rad2deg(np.abs(p.getEulerFromQuaternion(orn)[:2])).max()


def run(teleop, keys, seconds):
    for _ in range(int(seconds / DT)):
        teleop.apply(parse_keys(keys))
        p.stepSimulation()


def tap(teleop, key):
    run(teleop, {key: TAPPED}, DT)


def ee(teleop):
    return teleop.robot.link_pose(teleop.ee)


def angle_between(q1, q2):
    return np.linalg.norm(rotvec_from_quat(quat_mul(q1, quat_conj(q2))))


def test_holds_home_with_no_input():
    kinova, teleop = make()
    pos0, orn0 = ee(teleop)
    run(teleop, {}, 2.0)
    pos1, orn1 = ee(teleop)
    assert np.max(np.abs(kinova.q() - teleop.home_q)) < 0.01, kinova.q() - teleop.home_q
    assert np.linalg.norm(pos1 - pos0) < 0.005
    assert angle_between(orn0, orn1) < 0.02


def test_cartesian_translation():
    for key, axis, sign in [("w", 0, 1), ("s", 0, -1), ("a", 1, 1), ("d", 1, -1), ("q", 2, 1), ("e", 2, -1)]:
        _, teleop = make()
        pos0, orn0 = ee(teleop)
        run(teleop, {ord(key): HELD}, 1.0)
        run(teleop, {}, 0.5)  # let it settle
        pos1, orn1 = ee(teleop)
        d = pos1 - pos0
        off_axis = np.delete(d, axis)
        print(f"  {key}: moved {d.round(3)}  rot drift {np.rad2deg(angle_between(orn0, orn1)):.1f} deg")
        assert sign * d[axis] > 0.07, (key, d)  # 0.1 m/s for 1 s
        assert np.max(np.abs(off_axis)) < 0.01, (key, d)
        assert angle_between(orn0, orn1) < 0.05, key


def test_cartesian_rotation():
    for key, axis in [("u", 0), ("i", 1), ("j", 2)]:
        _, teleop = make()
        pos0, orn0 = ee(teleop)
        run(teleop, {ord(key): HELD}, 0.5)
        run(teleop, {}, 0.5)
        pos1, orn1 = ee(teleop)
        rv = rotvec_from_quat(quat_mul(orn1, quat_conj(orn0)))  # world-frame rotation
        print(f"  {key}: rotvec {rv.round(3)}  pos drift {np.linalg.norm(pos1 - pos0) * 1000:.1f} mm")
        assert rv[axis] > 0.2, (key, rv)  # 0.6 rad/s for 0.5 s
        assert np.max(np.abs(np.delete(rv, axis))) < 0.05, (key, rv)
        assert np.linalg.norm(pos1 - pos0) < 0.01, key


def test_joint_mode_moves_only_selected_joint():
    kinova, teleop = make()
    tap(teleop, ord("m"))
    assert teleop.mode == "joint"
    tap(teleop, ord("3"))
    q0 = kinova.q()
    run(teleop, {p.B3G_UP_ARROW: HELD}, 1.0)
    run(teleop, {}, 0.5)
    dq = kinova.q() - q0
    print(f"  joint 3 up: dq {dq.round(3)}")
    assert dq[2] > 0.4, dq
    assert np.max(np.abs(np.delete(dq, 2))) < 0.01, dq


def test_joint_limits_respected():
    kinova, teleop = make()
    tap(teleop, ord("m"))
    tap(teleop, ord("4"))
    run(teleop, {ord("s"): HELD}, 4.0)
    assert teleop.q_target[3] >= kinova.lower[3] - 1e-9


def test_home_returns():
    kinova, teleop = make()
    run(teleop, {ord("w"): HELD, ord("q"): HELD, ord("j"): HELD}, 1.0)
    assert np.max(np.abs(kinova.q() - teleop.home_q)) > 0.05
    tap(teleop, ord("h"))
    run(teleop, {}, 4.0)
    assert not teleop.homing
    assert np.max(np.abs(kinova.q() - teleop.home_q)) < 0.01


def test_target_does_not_run_away_out_of_reach():
    _, teleop = make()
    run(teleop, {ord("w"): HELD}, 15.0)  # 1.5 m forward: well past the arm's reach
    pos, _ = ee(teleop)
    assert np.linalg.norm(teleop.pos_target - pos) <= teleop.MAX_POS_LEAD + 1e-6


def test_gripper_toggle_and_hold():
    kinova, teleop = make()
    assert abs(kinova.gripper_closure()) < 0.02
    tap(teleop, ord(" "))
    run(teleop, {}, 1.5)
    print(f"  after SPACE: closure {kinova.gripper_closure():.2f}")
    assert kinova.gripper_closure() > 0.97
    tap(teleop, ord(" "))
    run(teleop, {}, 1.5)
    assert kinova.gripper_closure() < 0.03
    run(teleop, {ord("]"): HELD}, 0.4)  # hold to close part way
    run(teleop, {}, 0.5)
    print(f"  after holding ] 0.4 s: closure {kinova.gripper_closure():.2f}")
    assert 0.3 < kinova.gripper_closure() < 0.5


def pick_and_carry(dx, dy, yaw_deg):
    """Grip the cube from above, lift it with Q and carry it with D. Returns (closure, cube travel)."""
    kinova, teleop = make()
    cube = 2  # floor, table, cube, kinova (in load order)
    cube_pos0 = np.array([0.55 + dx, 0.1 + dy, 0.65])
    p.resetBasePositionAndOrientation(cube, cube_pos0.tolist(), p.getQuaternionFromEuler([0, 0, np.deg2rad(yaw_deg)]))
    top_down = np.array([1.0, 0, 0, 0])  # tool z axis pointing at the table

    # Setup: teleport the arm 12 cm above the cube (IK seeds from the current
    # state, so iterate) instead of swinging it there and maybe knocking the cube.
    pre_grasp = cube_pos0 + [0, 0, 0.12]
    for _ in range(20):
        q = kinova.inverse_kinematics(teleop.ee, pre_grasp, top_down, rest_q=teleop.home_q)
        kinova.reset(q)
    teleop.sync_target()
    run(teleop, {}, 0.5)

    run(teleop, {ord("e"): HELD}, 0.9)  # descend 9 cm: pads around the cube
    run(teleop, {}, 0.3)
    cube_before_grip = np.array(p.getBasePositionAndOrientation(cube)[0])
    assert np.linalg.norm(cube_before_grip - cube_pos0) < 0.002, "approach disturbed the cube"

    tap(teleop, ord(" "))
    run(teleop, {}, 1.5)
    closure = kinova.gripper_closure()
    run(teleop, {ord("q"): HELD}, 1.5)  # lift 15 cm
    run(teleop, {ord("d"): HELD}, 1.0)  # carry 10 cm to the right
    run(teleop, {}, 1.0)
    return closure, np.array(p.getBasePositionAndOrientation(cube)[0]) - cube_pos0


def test_pick_up_cube():
    for dx, dy, yaw in [(0, 0, 0), (0.005, 0, 0), (0, -0.005, 0), (0, 0, 15), (-0.005, 0.005, -15)]:
        closure, moved = pick_and_carry(dx, dy, yaw)
        print(f"  cube offset ({dx:+.3f}, {dy:+.3f}) yaw {yaw:+3d}: closure {closure:.2f}, cube moved {moved.round(3)}")
        assert 0.2 < closure < 0.9, closure  # stopped on the cube, not fully closed
        assert moved[2] > 0.12, moved  # lifted
        assert moved[1] < -0.07, moved  # carried along -y


def test_pick_and_place_key_round_trip():
    kinova, teleop, cube = make_with_task()
    for target in (PLACE_SPOTS[1], PLACE_SPOTS[0]):  # out to the second pad, then back
        press_p_and_wait(teleop)
        pos, tilt = cube_pose(cube)
        err = np.linalg.norm(pos[:2] - target)
        print(f"  {teleop.task_result}: cube at {pos.round(3)}, {err * 1000:.0f} mm from pad, tilt {tilt:.1f} deg")
        assert "placed" in teleop.task_result
        assert err < 0.01, err
        assert abs(pos[2] - 0.65) < 0.003 and tilt < 2  # resting flat on the table
        assert np.max(np.abs(kinova.q() - teleop.home_q)) < 0.02  # back home


def test_pick_and_place_from_other_spots():
    for xy, yaw in [((0.5, 0.0), 30), ((0.62, -0.05), -40), ((0.4, 0.22), 10), ((0.35, -0.1), 45)]:
        _, teleop, cube = make_with_task()
        p.resetBasePositionAndOrientation(cube, [*xy, 0.65], p.getQuaternionFromEuler([0, 0, np.deg2rad(yaw)]))
        target = max(PLACE_SPOTS, key=lambda s: np.linalg.norm(np.array(s) - xy))
        press_p_and_wait(teleop)
        pos, tilt = cube_pose(cube)
        err = np.linalg.norm(pos[:2] - target)
        print(f"  cube from {xy} yaw {yaw:+3d}: {err * 1000:.0f} mm from pad {target}, tilt {tilt:.1f} deg")
        assert err < 0.015, err
        assert tilt < 2


def test_pick_and_place_refuses_out_of_reach():
    _, teleop, cube = make_with_task()
    p.resetBasePositionAndOrientation(cube, [1.0, 0, 0.65], [0, 0, 0, 1])
    press_p_and_wait(teleop)
    assert teleop.task_result == "object is out of reach"
    assert np.linalg.norm(cube_pose(cube)[0] - [1.0, 0, 0.65]) < 0.002


def test_pick_and_place_backs_off_when_grasp_misses():
    _, teleop, cube = make_with_task()
    tick(teleop, {ord("p"): TAPPED})
    while teleop.task.status != "gripping":
        tick(teleop)
    p.resetBasePositionAndOrientation(cube, [0.3, 0.3, 0.65], [0, 0, 0, 1])  # snatch it away
    for _ in range(int(20 / (DT * CONTROL_SUBSTEPS))):
        if teleop.task is None and not teleop.homing:
            break
        tick(teleop)
    assert teleop.task_result == "missed the object", teleop.task_result


def test_any_key_takes_back_control_from_task():
    kinova, teleop, cube = make_with_task()
    tick(teleop, {ord("p"): TAPPED})
    for _ in range(int(2.0 / (DT * CONTROL_SUBSTEPS))):
        tick(teleop)
    assert teleop.task is not None
    z0 = kinova.link_pose(teleop.ee)[0][2]
    for _ in range(int(0.5 / (DT * CONTROL_SUBSTEPS))):
        tick(teleop, {ord("q"): HELD})
    assert teleop.task is None and teleop.task_result == "stopped by the user"
    assert teleop.mode_label().startswith("CARTESIAN")
    # the arm now follows the keyboard (Q = up), not the script
    assert kinova.link_pose(teleop.ee)[0][2] > z0 + 0.02


def test_force_selection_and_point_jacobian():
    kinova, _ = make()
    selector = ForceInput(kinova)
    idle_label = selector.hint()
    assert "0 N" in idle_label and "click" in idle_label.lower(), idle_label
    link = kinova.link_index("forearm_link")
    target = np.mean(p.getAABB(kinova._id, link), axis=0)
    camera = (
        640, 480,
        p.computeViewMatrix(target + [0, 0.35, 0], target, [0, 0, 1]),
        p.computeProjectionMatrixFOV(50, 4 / 3, 0.01, 10),
    )
    with patch.object(p, "getDebugVisualizerCamera", return_value=camera), patch.object(
        p, "getMouseEvents", return_value=[(2, 320, 240, 0, TAPPED)]
    ):
        load = selector.poll({ord("f"): TAPPED}, np.array([1, 1, 0]))
    assert load.link == link
    assert np.isclose(np.linalg.norm(load.force), 6)
    assert "6 N" in selector.hint(load, task_running=True)
    assert "0 N" in selector.hint(task_running=True)
    assert np.allclose(load.world_point(kinova)[[0, 2]], target[[0, 2]], atol=1e-5)

    # Check the whole point Jacobian against finite differences of the same
    # link-local point, including the zero columns downstream of the selection.
    q0 = kinova.q()
    jacobian, _ = kinova.link_jacobians(link, load.local_point)
    h = 0.002
    for j, dq in enumerate(np.eye(kinova.ndof) * h):
        kinova.reset(q0 + dq)
        plus = load.world_point(kinova)
        kinova.reset(q0 - dq)
        minus = load.world_point(kinova)
        assert np.allclose(jacobian[:, j], (plus - minus) / (2 * h), atol=1e-4)
    assert np.allclose(jacobian[:, 4:], 0)


def test_pick_and_place_recovers_after_point_forces():
    kinova, teleop, cube = make_with_task()
    tick(teleop, {ord("p"): TAPPED})
    for _ in range(900):
        if teleop.task.status == "carrying":
            break
        tick(teleop)
    else:
        raise AssertionError("task did not reach carrying")
    for _ in range(30):
        tick(teleop)
    nominal = teleop.q_target.copy()

    pushes = (("forearm_link", [0, -6, 0], 240), ("bracelet_link", [0, 0, 6], 45))
    for name, force, ticks in pushes:
        load = PointForce(kinova.link_index(name), np.array([0.03, 0.02, 0.04]), np.array(force))
        point_before = load.world_point(kinova)
        q_before = kinova.q()
        orn_before = ee(teleop)[1]
        for _ in range(min(60, ticks)):
            tick(teleop, load=load)
        short_travel = (load.world_point(kinova) - point_before) @ load.force / 6
        assert short_travel > 0.015, short_travel
        if ticks > 60:
            for _ in range(ticks - 60):
                tick(teleop, load=load)
            travel = (load.world_point(kinova) - point_before) @ load.force / 6
            assert travel > short_travel + 0.10, travel
            assert np.max(np.abs(kinova.q()[4:] - q_before[4:])) > 0.25
            assert angle_between(ee(teleop)[1], orn_before) < 0.08
        assert np.array_equal(teleop.q_target, nominal)
        assert teleop.admittance.phase == "yielding"

        # Release briefly, then push the second link mid-return.
        tick(teleop)
        assert teleop.active_load is None and teleop.admittance.phase == "returning"

    duration = teleop.admittance.return_duration
    assert duration > teleop.admittance.MIN_RETURN_TIME
    for _ in range(int(np.ceil(duration / teleop.dt)) - 1):
        tick(teleop)
    assert teleop.admittance.phase is None
    assert np.array_equal(teleop.q_target, nominal)
    assert np.max(np.abs(kinova.q() - nominal)) < 0.01

    for _ in range(900):
        tick(teleop)
        if teleop.task is None and not teleop.homing:
            break
    else:
        raise AssertionError("task did not finish after the force was released")
    pos, tilt = cube_pose(cube)
    assert "placed" in teleop.task_result
    assert np.linalg.norm(pos[:2] - PLACE_SPOTS[1]) < 0.01
    assert tilt < 2


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
        print("  ok")
    p.disconnect()
    print(f"\nall {len(tests)} tests passed")
