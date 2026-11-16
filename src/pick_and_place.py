import numpy as np
import pybullet as p

from rotations import quat_conj, quat_from_rotvec, quat_mul, rotvec_from_quat

TOP_DOWN = np.array([1.0, 0.0, 0.0, 0.0])  # tool z axis pointing straight down


class PickAndPlace:
    LINEAR_SPEED = 0.15  # m/s, average along each straight-line move
    ANGULAR_SPEED = 1.0  # rad/s
    TRAVEL_HEIGHT = 0.2  # tool height above the table top while carrying, m
    # tool_frame sits this far above the object's centre when gripping, m. This
    # centres the closed finger pads on the 5 cm cube; gripping higher lets the
    # cube swing about the grip line while carried (~30°) and land off target.
    GRASP_HEIGHT = 0.022
    PLACE_GAP = 0.005  # set the object down from this high, m
    MAX_REACH = 0.8  # horizontal distance from the arm base it will try to reach, m
    POS_TOL = 0.005  # m
    ROT_TOL = 0.03  # rad
    SETTLE_TIME = 1.5  # max time to wait for the arm to reach the end of a move, s
    MISSED_CLOSURE = 0.9  # gripper closing past this means it closed on nothing

    def __init__(self, teleop, obj_id, spots):
        """spots: (x, y) places on the table; the object goes to the one farthest from it."""
        self.teleop = teleop
        self.robot = teleop.robot
        self.obj = obj_id
        self.spots = [np.asarray(s, dtype=float) for s in spots]
        self.status = "starting"
        self.result = None  # set when the task ends: what happened
        self._plan = self._run()

    def step(self):
        """Advance one control tick. Returns False once the task has finished."""
        try:
            next(self._plan)
            return True
        except StopIteration:
            return False

    def _run(self):
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.obj)
        obj_pos = np.array(obj_pos)
        lo, hi = p.getAABB(self.obj)
        table_z = lo[2]  # assumes the object is resting on the table
        half_height = (hi[2] - lo[2]) / 2
        base_xy = np.array(p.getBasePositionAndOrientation(self.robot._id)[0][:2])
        if np.linalg.norm(obj_pos[:2] - base_xy) > self.MAX_REACH:
            self.result = "object is out of reach"
            return
        place_xy = max(self.spots, key=lambda s: np.linalg.norm(s - obj_pos[:2]))

        # Line the fingers up with the object's faces: its yaw, folded into ±45°.
        yaw = p.getEulerFromQuaternion(obj_orn)[2]
        yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        grasp_orn = quat_mul(quat_from_rotvec([0, 0, yaw]), TOP_DOWN)
        travel_z = table_z + self.TRAVEL_HEIGHT

        self.status = "opening gripper"
        yield from self._gripper(0.0)
        self.status = "moving to object"
        yield from self._move([*obj_pos[:2], travel_z], grasp_orn)
        yield from self._move(obj_pos + [0, 0, self.GRASP_HEIGHT], grasp_orn)
        self.status = "gripping"
        yield from self._gripper(1.0)
        if self.robot.gripper_closure() > self.MISSED_CLOSURE:
            self.status = "missed, backing off"
            yield from self._gripper(0.0)
            yield from self._move([*obj_pos[:2], travel_z], grasp_orn)
            self.result = "missed the object"
            return

        self.status = "lifting"
        yield from self._move([*obj_pos[:2], travel_z], grasp_orn)
        # Where the object actually sits in the hand decides how low to go when placing.
        tool_above_obj = self.robot.link_pose(self.teleop.ee)[0][2] - p.getBasePositionAndOrientation(self.obj)[0][2]
        self.status = "carrying"
        yield from self._move([*place_xy, travel_z], grasp_orn)
        place_z = table_z + half_height + self.PLACE_GAP + tool_above_obj
        yield from self._move([*place_xy, place_z], grasp_orn)
        self.status = "releasing"
        yield from self._gripper(0.0)
        yield from self._move([*place_xy, travel_z], grasp_orn)
        self.result = f"placed the object at ({place_xy[0]:.2f}, {place_xy[1]:.2f})"

    def _ticks(self, seconds):
        return max(1, int(round(seconds / self.teleop.dt)))

    def _gripper(self, goal):
        """Open/close and wait for the fingers to finish moving (or stall on the object)."""
        self.teleop.gripper_goal = goal
        while abs(self.teleop.gripper_target - goal) > 1e-9:
            yield
        for _ in range(self._ticks(0.4)):
            yield

    def _move(self, pos, orn):
        """Straight-line move of tool_frame to pos/orn, eased in and out, then settle."""
        t = self.teleop
        pos = np.asarray(pos, dtype=float)
        p0, q0 = t.pos_target.copy(), t.orn_target.copy()
        rot = rotvec_from_quat(quat_mul(orn, quat_conj(q0)))
        duration = max(
            np.linalg.norm(pos - p0) / self.LINEAR_SPEED,
            np.linalg.norm(rot) / self.ANGULAR_SPEED,
        )
        n = self._ticks(duration)
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)  # smoothstep: start and stop gently
            self._track(p0 + (pos - p0) * s, quat_mul(quat_from_rotvec(rot * s), q0))
            yield
        for _ in range(self._ticks(self.SETTLE_TIME)):
            ee_pos, ee_orn = self.robot.link_pose(t.ee)
            rot_err = np.linalg.norm(rotvec_from_quat(quat_mul(orn, quat_conj(ee_orn))))
            if np.linalg.norm(ee_pos - pos) < self.POS_TOL and rot_err < self.ROT_TOL:
                break
            self._track(pos, orn)
            yield

    def _track(self, pos, orn):
        t = self.teleop
        t.pos_target, t.orn_target = np.asarray(pos, dtype=float), orn / np.linalg.norm(orn)
        t.q_target = self.robot.inverse_kinematics(t.ee, t.pos_target, t.orn_target, rest_q=t.home_q)
