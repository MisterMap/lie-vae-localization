import cv2
import numpy as np

from .math import cvt_local2global, cvt_global2local


class ToyDatasetMaker(object):
    def __init__(self, map_height, map_width, image_size, resolution, radius=1, point_count=0):
        self._map_height = map_height
        self._map_width = map_width
        self._resolution = resolution
        self._radius = radius
        self._image_size = image_size
        self._centers, self._colors = self.make_random_circles(point_count)

    def make_random_circles(self, point_count):
        centers = np.random.random((point_count, 2))
        centers[:, 0] = centers[:, 0] * self._map_width
        centers[:, 1] = centers[:, 1] * self._map_height

        colors = np.random.randint(0, 255, (point_count, 3), dtype=np.uint8)
        return centers, colors

    def get_image(self, position):
        mask = np.sum((self._centers - position[:2]) ** 2, axis=1) < (self._image_size + self._radius) ** 2
        image_origin = cvt_local2global(np.array([-self._image_size / 2, -self._image_size / 2, 0]), position)
        centers = cvt_global2local(self._centers[mask], image_origin) / self._resolution
        image = np.ones((int(self._image_size // self._resolution),
                         int(self._image_size // self._resolution), 3),
                        dtype=np.uint8) * 255
        for center, color in zip(centers, self._colors[mask]):
            cv2.circle(image, (int(center[0]), int(center[1])), int(self._radius // self._resolution),
                       (int(color[0]), int(color[1]), int(color[2])), thickness=-1)
        image = cv2.blur(image, (int(self._radius // self._resolution // 2),
                                 int(self._radius // self._resolution // 2)))
        return image

    @staticmethod
    def make_trajectory_part(start, finish, step_length):
        point_count = int(np.linalg.norm(start - finish) / step_length) + 1
        x = np.linspace(start[0], finish[0], point_count)
        y = np.linspace(start[1], finish[1], point_count)
        angles = np.arctan2(finish[1] - start[1], finish[0] - start[0]) * np.ones(point_count)
        return np.array([x, y, angles]).T

    def snail_trajectory(self, step_length, line_count):
        trajectory = np.zeros((0, 3))
        line_step = self._map_width / line_count
        for line in range(line_count):
            if line % 2:
                start = np.array([line * line_step, self._map_height])
                finish = np.array([line * line_step, 0])
            else:
                start = np.array([line * line_step, 0])
                finish = np.array([line * line_step, self._map_height])
            trajectory_part = self.make_trajectory_part(start, finish, step_length)
            trajectory = np.concatenate((trajectory, trajectory_part), axis=0)
            if line % 2:
                start = np.array([line * line_step, 0])
                finish = np.array([line * line_step + line_step, 0])
            else:
                start = np.array([line * line_step, self._map_height])
                finish = np.array([line * line_step + line_step, self._map_height])
            trajectory_part = self.make_trajectory_part(start, finish, step_length)
            trajectory = np.concatenate((trajectory, trajectory_part), axis=0)
        return trajectory

    def make_random_trajectory(self, point_count):
        positions = np.random.random((point_count, 3))
        positions[:, 0] *= self._map_width
        positions[:, 1] *= self._map_height
        positions[:, 2] *= 2 * np.pi
        return positions

    def save_dataset(self, path, step_length, line_count, trajectory_type="snail"):
        if trajectory_type == "snail":
            trajectory = self.snail_trajectory(step_length, line_count)
        elif trajectory_type == "random":
            trajectory = self.make_random_trajectory(line_count)
        else:
            trajectory = None
        images = []
        for pose in trajectory:
            images.append(self.get_image(pose))

        data = {
            "images": images,
            "trajectory": trajectory,
            "point_centers": self._centers,
            "point_colors": self._colors
        }
        np.savez(path, data)
