import time
from dataclasses import dataclass, field

import numpy as np
import pybullet as p

from admittance import JointAdmittance
from force_input import ForceInput
from pick_and_place import PickAndPlace
from pybullet_api import PyBullet, Kinova, VisualBox
from rotations import quat_conj, quat_from_rotvec, quat_mul, rotvec_from_quat

HELP = """
Click the simulation window for keyboard focus.
P starts or stops pick-and-place. F toggles force mode.
In force mode, click a moving link, then hold W/S, A/D, or Q/E.
Manual: M switches modes, H homes, Space toggles the gripper,
[ and ] adjust grip, - and = change speed, X quits.
Cartesian: W/S A/D Q/E translate, U/O I/K J/L rotate.
Joint: 1–7 selects, W/S or the up/down arrows move the joint.
"""

HOME_Q = np.deg2rad([0, 15, 180, -130, 0, 55, 90])
EE_LINK = "tool_frame"
# Green pads on the table; P moves the cube to whichever is farther from it.
# The cube starts on the first one.
PLACE_SPOTS = [(0.55, 0.10), (0.45, -0.20)]

# key -> (axis, sign)
LINEAR_KEYS = {
    ord("w"): (0, +1), ord("s"): (0, -1),
    ord("a"): (1, +1), ord("d"): (1, -1),
    ord("q"): (2, +1), ord("e"): (2, -1),
}
ANGULAR_KEYS = {
    ord("u"): (0, +1), ord("o"): (0, -1),
    ord("i"): (1, +1), ord("k"): (1, -1),
    ord("j"): (2, +1), ord("l"): (2, -1),
}
JOINT_UP_KEYS = (ord("w"), p.B3G_UP_ARROW)
JOINT_DOWN_KEYS = (ord("s"), p.B3G_DOWN_ARROW)
JOINT_SELECT_KEYS = {ord(str(n)): n - 1 for n in range(1, 8)}
GRIPPER_TOGGLE_KEY = ord(" ")
GRIPPER_OPEN_KEY = ord("[")
GRIPPER_CLOSE_KEY = ord("]")


@dataclass
class Command:
    """What the keyboard asked for during one control step."""

    linear: np.ndarray = field(default_factory=lambda: np.zeros(3))  # each in {-1, 0, 1}
    angular: np.ndarray = field(default_factory=lambda: np.zeros(3))  # each in {-1, 0, 1}
    joint_dir: int = 0
    select_joint: int | None = None
    gripper_dir: int = 0  # +1 close, -1 open
    gripper_toggle: bool = False
    toggle_mode: bool = False
    home: bool = False
    pick_place: bool = False
    speed_change: int = 0
    quit: bool = False


def parse_keys(keys):
    """Turn a p.getKeyboardEvents() dict into a Command.

    Held keys (KEY_IS_DOWN) drive continuous motion; single actions such as
    toggling the mode fire once, on the press (KEY_WAS_TRIGGERED).
    """
    cmd = Command()

    def down(k):
        return bool(keys.get(k, 0) & p.KEY_IS_DOWN)

    def pressed(k):
        return bool(keys.get(k, 0) & p.KEY_WAS_TRIGGERED)

    for k, (axis, sign) in LINEAR_KEYS.items():
        if down(k):
            cmd.linear[axis] += sign
    for k, (axis, sign) in ANGULAR_KEYS.items():
        if down(k):
            cmd.angular[axis] += sign
    cmd.joint_dir = int(any(down(k) for k in JOINT_UP_KEYS)) - int(
        any(down(k) for k in JOINT_DOWN_KEYS)
    )
    cmd.gripper_dir = int(down(GRIPPER_CLOSE_KEY)) - int(down(GRIPPER_OPEN_KEY))
    cmd.gripper_toggle = pressed(GRIPPER_TOGGLE_KEY)
    for k, j in JOINT_SELECT_KEYS.items():
        if pressed(k):
            cmd.select_joint = j
    cmd.toggle_mode = pressed(ord("m"))
    cmd.home = pressed(ord("h"))
    cmd.pick_place = pressed(ord("p"))
    cmd.speed_change = int(pressed(ord("="))) - int(pressed(ord("-")))
    cmd.quit = pressed(ord("x"))
    return cmd


class KinovaTeleop:
    LINEAR_SPEED = 0.10  # m/s
    ANGULAR_SPEED = 0.6  # rad/s
    JOINT_SPEED = 0.5  # rad/s
    GRIPPER_SPEED = 1.0  # closure per second (full stroke in 1 s)
    # How far the Cartesian target may run ahead of the real end effector.
    # Without this, holding a key at the edge of the workspace keeps moving the
    # target while the arm stalls, and the arm then jumps when you change direction.
    MAX_POS_LEAD = 0.03  # m
    MAX_ROT_LEAD = 0.25  # rad

    def __init__(self, robot, dt, home_q=HOME_Q, ee_link=EE_LINK, task_factory=None):
        """task_factory(teleop) -> task with .step() and .status, started by the P key."""
        self.robot = robot
        self.dt = dt
        self.home_q = np.asarray(home_q, dtype=float)
        self.ee = robot.link_index(ee_link)
        self.mode = "cartesian"
        self.joint = 0
        self.speed = 1.0
        self.homing = False
        self.task_factory = task_factory
        self.task = None  # automatic task in progress (P key)
        self.task_result = None  # what the last automatic task reported when it ended
        self.admittance = JointAdmittance(robot, dt, self.ee)
        self.active_load = None
        self.gripper_goal = 0.0  # closure the user asked for, 0 = open, 1 = closed
        self.gripper_target = 0.0  # closure being commanded, ramps toward the goal
        robot.reset(self.home_q)
        self.sync_target()

    def sync_target(self):
        """Make the commanded pose/joints match where the arm actually is."""
        self.q_target = self.robot.q()
        self.pos_target, self.orn_target = self.robot.link_pose(self.ee)

    def apply(self, cmd, load=None):
        """Update the targets from one Command and send the joint command."""
        moving = cmd.linear.any() or cmd.angular.any() or cmd.joint_dir != 0
        if self.task is not None:
            manual = (moving or cmd.home or cmd.toggle_mode or cmd.pick_place
                      or cmd.gripper_toggle or cmd.gripper_dir)
            if manual:  # any manual input takes control back from the automatic task
                self.task = None
                self.admittance.reset()
                self.task_result = "stopped by the user"
                self.sync_target()
                self.gripper_goal = self.gripper_target  # hold the fingers where they are
        elif cmd.pick_place and self.task_factory is not None:
            self.homing = False
            self.sync_target()
            self.task = self.task_factory(self)

        if cmd.toggle_mode:
            self.mode = "joint" if self.mode == "cartesian" else "cartesian"
            self.homing = False
            self.sync_target()
        if cmd.select_joint is not None:
            self.joint = cmd.select_joint
        if cmd.speed_change:
            self.speed = float(np.clip(self.speed * 1.5 ** cmd.speed_change, 0.1, 4.0))
        if cmd.home:
            self.homing = True

        self.active_load = None
        if self.task is not None:
            override = self.admittance.update(self.q_target, load)
            if override is not None:
                self.active_load = load
                self.robot.cmd(override)
                return
            if not self.task.step():
                self.task_result = self.task.result
                self.task = None
                self.homing = True  # finish by driving back to home
        elif self.homing:
            if moving:  # any motion key cancels homing
                self.homing = False
                self.sync_target()
            else:
                self.q_target = self.home_q.copy()
                if np.max(np.abs(self.robot.q() - self.home_q)) < 0.01:
                    self.homing = False
                    self.sync_target()

        if self.task is None and not self.homing:
            if self.mode == "cartesian":
                self._step_cartesian(cmd)
            else:
                self._step_joint(cmd)
        self.robot.cmd(self.q_target)
        self._step_gripper(cmd)

    def _step_gripper(self, cmd):
        max_step = self.GRIPPER_SPEED * self.dt
        if cmd.gripper_toggle:
            self.gripper_goal = 0.0 if self.gripper_goal > 0.5 else 1.0
        if cmd.gripper_dir:
            self.gripper_goal = float(np.clip(self.gripper_target + cmd.gripper_dir * max_step, 0.0, 1.0))
        self.gripper_target += np.clip(self.gripper_goal - self.gripper_target, -max_step, max_step)
        self.robot.gripper_cmd(self.gripper_target)

    def _step_cartesian(self, cmd):
        if not (cmd.linear.any() or cmd.angular.any()):
            return  # hold the last joint target; re-solving IK while idle lets it drift
        step = self.speed * self.dt
        pos, orn = self.robot.link_pose(self.ee)

        target = self.pos_target + cmd.linear * self.LINEAR_SPEED * step
        lead = target - pos
        if np.linalg.norm(lead) > self.MAX_POS_LEAD:
            target = pos + lead / np.linalg.norm(lead) * self.MAX_POS_LEAD
        self.pos_target = target

        # rotate about world axes: q_new = dq * q
        dq = quat_from_rotvec(cmd.angular * self.ANGULAR_SPEED * step)
        target_orn = quat_mul(dq, self.orn_target)
        rot_lead = rotvec_from_quat(quat_mul(target_orn, quat_conj(orn)))
        if np.linalg.norm(rot_lead) > self.MAX_ROT_LEAD:
            rot_lead = rot_lead / np.linalg.norm(rot_lead) * self.MAX_ROT_LEAD
            target_orn = quat_mul(quat_from_rotvec(rot_lead), orn)
        self.orn_target = target_orn / np.linalg.norm(target_orn)

        self.q_target = self.robot.inverse_kinematics(
            self.ee, self.pos_target, self.orn_target, rest_q=self.home_q
        )

    def _step_joint(self, cmd):
        if cmd.joint_dir == 0:
            return
        j = self.joint
        self.q_target[j] += cmd.joint_dir * self.JOINT_SPEED * self.speed * self.dt
        self.q_target[j] = np.clip(self.q_target[j], self.robot.lower[j], self.robot.upper[j])

    def mode_label(self):
        if self.task is not None:
            phase = self.admittance.phase
            state = f" | {phase}" if phase else ""
            return f"AUTO: {self.task.status}{state}"
        if self.homing:
            mode = "HOMING"
        elif self.mode == "cartesian":
            mode = "CARTESIAN"
        else:
            mode = f"JOINT {self.joint + 1}"
        return f"{mode}   speed x{self.speed:.2f}"


def build_scene(gui=True, dt=1.0 / 240.0):
    """Floor, table, a cube, two place pads, and the Kinova mounted on the table edge.

    Returns (pybullet world, kinova, cube body id).
    """
    pb = PyBullet(dt, gui=gui)
    if gui:
        # Otherwise pybullet's built-in shortcuts (w = wireframe, s = shadows, ...)
        # fire alongside our controls.
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
    p.loadURDF("table/table.urdf", basePosition=[0.4, 0, 0])
    cube = p.loadURDF("cube_small.urdf", basePosition=[*PLACE_SPOTS[0], 0.65])
    table_top = p.getAABB(cube)[0][2]
    for xy in PLACE_SPOTS:  # visual only, no collision
        VisualBox([*xy, table_top + 0.0005], [0.04, 0.04, 0.0005], rgba_color=[0.3, 0.75, 0.35, 1])
    kinova = Kinova(base_position=[0, 0, 0.6])
    return pb, kinova, cube


PHYSICS_DT = 1.0 / 240.0
# Keys + IK run at 60 Hz, physics at 240 Hz. Solving IK every physics step is
# too slow in the GUI (each pybullet call syncs with the renderer).
CONTROL_SUBSTEPS = 4


def main():
    control_dt = PHYSICS_DT * CONTROL_SUBSTEPS
    pb, kinova, cube = build_scene(gui=True, dt=PHYSICS_DT)
    teleop = KinovaTeleop(
        kinova, control_dt, task_factory=lambda t: PickAndPlace(t, cube, PLACE_SPOTS)
    )
    marker = VisualBox(teleop.pos_target, [0.012] * 3, rgba_color=[1, 0.2, 0.2, 0.6])
    force_input = ForceInput(kinova)
    print(HELP)

    hud_id = -1
    last_label = None
    force_hud_id = -1
    last_force_label = None
    tick = 0
    next_tick = time.perf_counter()
    try:
        while p.isConnected():
            keys = p.getKeyboardEvents()
            cmd = parse_keys(keys)
            if cmd.quit:
                break
            load = force_input.poll(keys, cmd.linear)
            if force_input.enabled:
                cmd.linear.fill(0)
                cmd.joint_dir = 0
            teleop.apply(cmd, load)
            for _ in range(CONTROL_SUBSTEPS):
                if teleop.active_load is not None:
                    teleop.active_load.apply(kinova)
                p.stepSimulation()

            if tick % 2 == 0:  # 30 Hz is plenty for the target marker
                marker.reset(teleop.pos_target.tolist(), teleop.orn_target.tolist())
                force_input.draw(teleop.active_load)
            # Replace debug text only when its contents change.
            label = teleop.mode_label()
            if label != last_label:
                hud_id = p.addUserDebugText(
                    label, [0, 0.45, 1.5], textColorRGB=[0, 0, 0],
                    textSize=1.5, replaceItemUniqueId=hud_id,
                )
                last_label = label
            force_label = force_input.hint(teleop.active_load, task_running=teleop.task is not None)
            if force_label != last_force_label:
                force_hud_id = p.addUserDebugText(
                    force_label, [0, 0.45, 1.45], textColorRGB=[0, 0.28, 0.38],
                    textSize=1.35, replaceItemUniqueId=force_hud_id,
                )
                last_force_label = force_label
            if teleop.task_result is not None:
                print(f"[pick & place] {teleop.task_result}", flush=True)
                teleop.task_result = None
            tick += 1
            # Pace against an absolute schedule so sleep overshoot doesn't
            # accumulate and make the sim run slower than real time.
            next_tick += control_dt
            lag = next_tick - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
            elif lag < -0.1:  # fell far behind (e.g. window dragged); don't try to catch up
                next_tick = time.perf_counter()
    except p.error:
        pass  # window was closed
    finally:
        if p.isConnected():
            pb.close()


if __name__ == "__main__":
    main()
