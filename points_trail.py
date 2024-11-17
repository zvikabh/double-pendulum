import cv2
import numpy as np

import opencv_utils


TRAIL_COLOR = (0, 0, 0.8)
TRAIL_FADEOUT_START = 200
TRAIL_FADEOUT_END = 400


class PointsTrail:

  def __init__(self):
    self.points = []

  def add_point(self, pt: tuple[int, int]) -> None:
    self.points.append(pt)
    if len(self.points) > TRAIL_FADEOUT_END:
      del self.points[:-TRAIL_FADEOUT_END]
  
  def draw_on_img(self, img: np.ndarray) -> None:
    trail_color = np.asarray(TRAIL_COLOR, dtype=np.float32)
    background_color = np.asanyarray(opencv_utils.BACKGROUND_COLOR, dtype=np.float32)
    for i, pt in enumerate(self.points):
      pos = len(self.points) - i
      if pos > TRAIL_FADEOUT_START:
        alpha = 1 - (pos - TRAIL_FADEOUT_START)/(TRAIL_FADEOUT_END - TRAIL_FADEOUT_START)
        color = alpha*trail_color + (1-alpha)*background_color
      else:
        color = trail_color
      cv2.circle(img, pt, 5, tuple(color.tolist()), -1)
