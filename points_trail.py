import cv2
import numpy as np


TRAIL_COLOR = (0, 0, 0.8)
TRAIL_FADEOUT_START = 100
TRAIL_FADEOUT_END = 200


class PointsTrail:

  def __init__(self):
    self.points = []

  def add_point(self, pt: tuple[int, int]) -> None:
    self.points.append(pt)
    if len(self.points) > TRAIL_FADEOUT_END:
      del self.points[:-TRAIL_FADEOUT_END]
  
  def draw_on_img(self, img: np.ndarray) -> None:
    for i, pt in enumerate(self.points[::-1]):
      color = np.asarray(TRAIL_COLOR, dtype=np.float32)
      if i > TRAIL_FADEOUT_START:
        color *= 1 - (i - TRAIL_FADEOUT_START)/(TRAIL_FADEOUT_END - TRAIL_FADEOUT_START)
      cv2.circle(img, pt, 5, tuple(color.tolist()), -1)
