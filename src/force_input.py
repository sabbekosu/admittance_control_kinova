import numpy as np
import pybullet as p

from admittance import PointForce
from pybullet_api import VisualBox


def camera_ray(x, y, camera):
    width, height, view, projection = camera[:4]
    clip_x, clip_y = 2 * x / width - 1, 1 - 2 * y / height
    world_from_clip = np.linalg.inv(
        np.asarray(projection).reshape(4, 4, order="F")
        @ np.asarray(view).reshape(4, 4, order="F")
    )
    endpoints = world_from_clip @ np.array([[clip_x, clip_x], [clip_y, clip_y], [-1, 1], [1, 1]])
    return (endpoints[:3] / endpoints[3]).T


class ForceInput:
    NEWTONS = 6.0
    HIDDEN = [0, 0, -1]

    def __init__(self, robot):
        self.robot = robot
        self.enabled = False
        self.point = None
        self.link_name = ""
        self.base_link = robot.link_index("base_link")
        self.marker = VisualBox(self.HIDDEN, [0.009] * 3, rgba_color=[0.2, 0.9, 1, 1])
        self.marker_visible = False
        self.line = -1

    def poll(self, keys, direction):
        if keys.get(ord("f"), 0) & p.KEY_WAS_TRIGGERED:
            self.enabled = not self.enabled

        for event, x, y, button, flags in p.getMouseEvents():
            if self.enabled and event == 2 and button == 0 and flags & p.KEY_WAS_TRIGGERED:
                self._pick(x, y)

        if self.enabled and self.point is not None and direction.any():
            link, local_point = self.point
            return PointForce(link, local_point, self.NEWTONS * direction / np.linalg.norm(direction))
        return None

    def _pick(self, x, y):
        camera = p.getDebugVisualizerCamera()
        if min(camera[:2]) <= 0:
            return
        body, link, _, hit, _ = p.rayTest(*camera_ray(x, y, camera))[0]
        self.point = None
        if body == self.robot._id and link >= 0 and link != self.base_link:
            pos, orn = self.robot.link_pose(link)
            rotation = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            self.point = link, rotation.T @ (np.asarray(hit) - pos)
            self.link_name = p.getJointInfo(self.robot._id, link)[12].decode()

    def draw(self, active_load):
        if self.enabled and self.point is not None:
            selected = active_load or PointForce(*self.point, np.zeros(3))
            origin = selected.world_point(self.robot)
            self.marker.reset(origin)
            self.marker_visible = True
            if active_load is not None:
                tip = origin + 0.10 * active_load.force / self.NEWTONS
                self.line = p.addUserDebugLine(
                    origin, tip, [0, 0.65, 0.9], lineWidth=5, replaceItemUniqueId=self.line
                )
                return
        elif self.marker_visible:
            self.marker.reset(self.HIDDEN)
            self.marker_visible = False
        if self.line >= 0:
            p.removeUserDebugItem(self.line)
            self.line = -1

    def hint(self, active_load=None, task_running=False):
        if not self.enabled:
            return "Force 0 N  |  F, then click a moving link"
        if self.point is None:
            return "Force 0 N  |  Click a moving link"
        if not task_running:
            return f"Force 0 N  |  {self.link_name}  |  P to start"
        magnitude = np.linalg.norm(active_load.force) if active_load is not None else 0.0
        return f"Force {magnitude:g} N  |  {self.link_name}  |  W/S A/D Q/E"
