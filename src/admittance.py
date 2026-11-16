from dataclasses import dataclass

import numpy as np
import pybullet as p

from rotations import quat_conj, quat_mul, rotvec_from_quat


@dataclass
class PointForce:
    link: int
    local_point: np.ndarray
    force: np.ndarray  # world axes, N

    def world_point(self, robot):
        pos, orn = robot.link_pose(self.link)
        rotation = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        return pos + rotation @ self.local_point

    def apply(self, robot):
        # Bullet clears external forces after each physics step.
        p.applyExternalForce(
            robot._id, self.link, self.force, self.world_point(robot), p.WORLD_FRAME
        )


class JointAdmittance:
    INERTIA = 0.30  # kg m²
    DAMPING = 6.0  # Nm s/rad
    MAX_SPEED = 0.75  # rad/s per joint during the push
    MAX_LEAD = 0.12  # rad ahead of the measured joint position
    ORIENTATION_GAIN = 4.0  # 1/s
    ORIENTATION_DAMPING = 0.10  # regularizes the wrist solve near singularities
    MIN_RETURN_TIME = 0.65  # s
    RETURN_AVERAGE_SPEED = 0.4  # rad/s for the joint farthest from its target

    def __init__(self, robot, dt, tool_link):
        self.robot = robot
        self.dt = dt
        self.tool_link = tool_link
        self.offset = np.zeros(robot.ndof)
        self.velocity = np.zeros(robot.ndof)
        self.phase = None
        self._tool_orientation = None
        self._return_start = None
        self._return_elapsed = 0.0
        self.return_duration = 0.0

    def reset(self):
        self.offset.fill(0)
        self.velocity.fill(0)
        self.phase = None

    def update(self, nominal, load):
        """Return an override joint target, or None when the task can advance."""
        if load is None:
            if self.phase is None:
                return None
            if self.phase == "yielding":
                self._start_return(nominal)
            self.offset = self._return_step()
            return nominal + self.offset

        measured_q = self.robot.q()
        _, orientation = self.robot.link_pose(self.tool_link)
        if self.phase is None:
            self._tool_orientation = orientation
        self.phase = "yielding"

        velocity = self._push_velocity(load, orientation)
        target = nominal + self.offset + self.dt * velocity
        target = np.clip(target, measured_q - self.MAX_LEAD, measured_q + self.MAX_LEAD)
        target = np.clip(target, self.robot.lower, self.robot.upper)
        offset = target - nominal
        self.velocity = (offset - self.offset) / self.dt
        self.offset = offset
        return target

    def _push_velocity(self, load, orientation):
        contact, _ = self.robot.link_jacobians(load.link, load.local_point)
        _, angular = self.robot.link_jacobians(self.tool_link, [0, 0, 0])
        _, singular_values, right = np.linalg.svd(contact, full_matrices=False)
        basis = right[singular_values > 1e-5]
        row_space = basis.T @ basis

        # Integrate mass–damper dynamics in the directions that move the contact.
        drive = row_space @ self.velocity
        drive += self.dt * (contact.T @ load.force - self.DAMPING * drive) / self.INERTIA

        # Use the remaining joint motion to stabilize the gripper's orientation.
        angular_null = angular - angular @ row_space
        error = rotvec_from_quat(quat_mul(self._tool_orientation, quat_conj(orientation)))
        rhs = self.ORIENTATION_GAIN * error - angular @ drive
        gram = angular_null @ angular_null.T + self.ORIENTATION_DAMPING**2 * np.eye(3)
        velocity = drive + angular_null.T @ np.linalg.solve(gram, rhs)
        return velocity / max(1.0, np.max(np.abs(velocity)) / self.MAX_SPEED)

    def _start_return(self, nominal):
        self.phase = "returning"
        self._return_elapsed = 0.0
        distance = np.max(np.abs(self.offset))
        self.return_duration = max(self.MIN_RETURN_TIME, distance / self.RETURN_AVERAGE_SPEED)
        # Bound the cubic's interior Bézier point so the whole return respects
        # the joint limits, even when the push ends with outward velocity.
        handle = np.clip(
            self.offset + (self.return_duration / 3) * self.velocity,
            self.robot.lower - nominal, self.robot.upper - nominal,
        )
        self._return_start = self.offset.copy(), (3 / self.return_duration) * (handle - self.offset)

    def _return_step(self):
        """Cubic with the release offset/rate and zero final offset/rate."""
        self._return_elapsed += self.dt
        s = self._return_elapsed / self.return_duration
        if s >= 1 - 1e-12:
            self.reset()
            return self.offset

        offset, velocity = self._return_start
        self.velocity = (6 * s * (s - 1) / self.return_duration) * offset + (
            3 * s**2 - 4 * s + 1
        ) * velocity
        return (1 - s)**2 * ((1 + 2 * s) * offset + s * self.return_duration * velocity)
