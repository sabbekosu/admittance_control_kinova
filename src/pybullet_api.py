from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

KINOVA_URDF = Path(__file__).resolve().parents[1] / "robots/kinova/gen3_7dof_robotiq_85.urdf"


class PyBullet:
    def __init__(self, dt, gui=True):
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(dt)
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(1.5, -70, -30, [0.35, 0, 0.75])

        floor = p.createCollisionShape(p.GEOM_PLANE)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=floor)

    def close(self):
        p.disconnect(self.client_id)


class VisualBox:
    def __init__(
        self,
        base_position,
        half_extents,
        rgba_color,
    ):
        visid = p.createVisualShape(
            p.GEOM_BOX, rgbaColor=rgba_color, halfExtents=half_extents
        )
        self._id = p.createMultiBody(
            baseMass=0.0,
            basePosition=base_position,
            baseVisualShapeIndex=visid,
        )

    def reset(self, base_position, base_orientation=(0, 0, 0, 1)):
        p.resetBasePositionAndOrientation(
            self._id,
            base_position,
            base_orientation,
        )


class Kinova:
    """Kinova Gen3 7-DoF arm with a Robotiq 2F-85 gripper."""

    ARM_JOINTS = [f"joint_{i}" for i in range(1, 8)]
    GRIPPER_DRIVER = "finger_joint"  # 0 = fully open, GRIPPER_CLOSED = fully closed
    # Gear constraints reproduce the gripper linkage in PyBullet.
    GRIPPER_MIMIC = {
        "left_inner_knuckle_joint": 1,
        "left_inner_finger_joint": -1,
        "right_outer_knuckle_joint": 1,
        "right_inner_knuckle_joint": 1,
        "right_inner_finger_joint": -1,
    }
    GRIPPER_CLOSED = 0.8
    GRIPPER_FORCE = 10.0  # max driver torque, Nm (the URDF's 1000 crushes through objects)
    FINGER_LINKS = [
        "left_inner_finger", "left_inner_finger_pad",
        "right_inner_finger", "right_inner_finger_pad",
    ]

    def __init__(self, base_position=(0, 0, 0)):
        self._id = p.loadURDF(str(KINOVA_URDF), basePosition=base_position, useFixedBase=True)
        self._joint_index = {}
        self._link_index = {}
        limits = {}
        for j in range(p.getNumJoints(self._id)):
            info = p.getJointInfo(self._id, j)
            self._joint_index[info[1].decode()] = j
            self._link_index[info[12].decode()] = j
            if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                lower, upper, force = info[8:11]
                if lower > upper:  # continuous joint
                    lower, upper = -2 * np.pi, 2 * np.pi
                limits[j] = lower, upper, force

        self._movable_joints = list(limits)
        self._actuated_joints = [self._joint_index[name] for name in self.ARM_JOINTS]
        # Bullet's kinematics take both arm and finger joints.
        self._ik_slots = [self._movable_joints.index(j) for j in self._actuated_joints]
        self._all_lower, self._all_upper, all_forces = np.array(list(limits.values())).T
        self.lower = self._all_lower[self._ik_slots]
        self.upper = self._all_upper[self._ik_slots]
        self.max_forces = all_forces[self._ik_slots]
        self.ndof = len(self._actuated_joints)
        self._driver = self._joint_index[self.GRIPPER_DRIVER]
        for name, mult in self.GRIPPER_MIMIC.items():
            j = self._joint_index[name]
            p.setJointMotorControl2(self._id, j, p.VELOCITY_CONTROL, force=0)
            c = p.createConstraint(
                self._id, self._driver, self._id, j, p.JOINT_GEAR,
                jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0],
            )
            p.changeConstraint(c, gearRatio=-mult, maxForce=100, erp=1)
        for name in self.FINGER_LINKS:
            p.changeDynamics(self._id, self.link_index(name), lateralFriction=1.5)

    def link_index(self, link_name):
        return self._link_index[link_name]

    def reset(self, q):
        for j, idx in enumerate(self._actuated_joints):
            p.resetJointState(self._id, idx, q[j])

    def cmd(self, q):
        p.setJointMotorControlArray(
            self._id,
            self._actuated_joints,
            p.POSITION_CONTROL,
            targetPositions=np.asarray(q).tolist(),
            forces=self.max_forces.tolist(),
        )

    def _joint_positions(self, joints):
        return np.array([state[0] for state in p.getJointStates(self._id, joints)])

    def q(self):
        return self._joint_positions(self._actuated_joints)

    def link_pose(self, link):
        """World position and [x, y, z, w] orientation of a link (index or name)."""
        if isinstance(link, str):
            link = self.link_index(link)
        state = p.getLinkState(self._id, link, computeForwardKinematics=True)
        return np.array(state[4]), np.array(state[5])

    def link_jacobians(self, link, local_point):
        """World linear and angular Jacobians for the actuated joints."""
        q = self._joint_positions(self._movable_joints).tolist()
        zero = [0.0] * len(q)
        linear, angular = p.calculateJacobian(self._id, link, local_point, q, zero, zero)
        return np.asarray(linear)[:, self._ik_slots], np.asarray(angular)[:, self._ik_slots]

    def inverse_kinematics(self, link, position, orientation, rest_q):
        """Joint angles that put `link` at the given world pose, biased toward rest_q."""
        if isinstance(link, str):
            link = self.link_index(link)
        # pybullet's IK wants limits/rest poses for every movable joint; the
        # non-actuated ones (e.g. gripper fingers) rest where they are.
        rest = self._joint_positions(self._movable_joints)
        rest[self._ik_slots] = rest_q
        q = p.calculateInverseKinematics(
            self._id,
            link,
            targetPosition=np.asarray(position).tolist(),
            targetOrientation=np.asarray(orientation).tolist(),
            lowerLimits=self._all_lower.tolist(),
            upperLimits=self._all_upper.tolist(),
            jointRanges=(self._all_upper - self._all_lower).tolist(),
            restPoses=rest.tolist(),
            maxNumIterations=30,
            residualThreshold=1e-5,
        )
        return np.array(q)[self._ik_slots]

    def gripper_cmd(self, closure):
        """Drive the gripper to a closure in [0, 1]: 0 = fully open, 1 = fully closed."""
        angle = float(np.clip(closure, 0.0, 1.0)) * self.GRIPPER_CLOSED
        p.setJointMotorControl2(
            self._id, self._driver, p.POSITION_CONTROL,
            targetPosition=angle, force=self.GRIPPER_FORCE,
        )

    def gripper_closure(self):
        """Current closure in [0, 1], read from finger_joint."""
        return p.getJointState(self._id, self._driver)[0] / self.GRIPPER_CLOSED
