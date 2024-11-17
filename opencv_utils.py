import cv2
import numpy as np


Point = tuple[float, float]
Color = tuple[int, int, int]


BACKGROUND_COLOR = (1, 1, 1)
ROD_FILL_COLOR = (1, 0.5, 0)
HINGE_FILL_COLOR = BACKGROUND_COLOR
ROD_BORDER_COLOR = (.8, .2, 0)
ROD_BORDER_THICKNESS = 3
ROD_WIDTH = 100  # Must be even
ROD_HWIDTH = ROD_WIDTH // 2
HINGE_RADIUS = 20


def ellipse_with_fill(img, center, axes, startAngle, endAngle, fillColor, borderColor, borderThickness):
  cv2.ellipse(
    img, center, axes,
    angle=0, 
    startAngle=startAngle, 
    endAngle=endAngle, 
    color=fillColor,
    thickness=-1,
    lineType=cv2.LINE_AA
  )
  cv2.ellipse(
    img, center, axes,
    angle=0, 
    startAngle=startAngle, 
    endAngle=endAngle, 
    color=borderColor,
    thickness=borderThickness,
    lineType=cv2.LINE_AA
  )


def draw_hinge(img: np.ndarray, hinge_center: Point, angle: float):
  ellipse_with_fill(
    img, hinge_center,
    axes=(ROD_HWIDTH, ROD_HWIDTH), 
    startAngle=angle,
    endAngle=angle+180,
    fillColor=ROD_FILL_COLOR,
    borderColor=ROD_BORDER_COLOR,
    borderThickness=ROD_BORDER_THICKNESS,
  )
  ellipse_with_fill(
    img, hinge_center,
    axes=(HINGE_RADIUS, HINGE_RADIUS), 
    startAngle=0,
    endAngle=360,
    fillColor=HINGE_FILL_COLOR,
    borderColor=ROD_BORDER_COLOR,
    borderThickness=ROD_BORDER_THICKNESS,
  )


def draw_rod(img: np.ndarray, hinge1: Point, hinge2: Point):
  hinge1 = np.asarray(hinge1)
  hinge2 = np.asarray(hinge2)
  dx, dy = hinge2 - hinge1
  angle = np.arctan2(dy, dx)
  angle_deg = np.rad2deg(angle)
  unit_vec_perp = np.asarray([np.sin(angle), -np.cos(angle)], dtype=np.float32)
  poly = np.asarray(
      [[
          hinge1 + ROD_HWIDTH*unit_vec_perp,
          hinge1 - ROD_HWIDTH*unit_vec_perp,
          hinge2 - ROD_HWIDTH*unit_vec_perp,
          hinge2 + ROD_HWIDTH*unit_vec_perp,
      ]],
      dtype=np.int32
  )
  cv2.fillPoly(img, poly, ROD_FILL_COLOR)
  cv2.line(img, poly[0, 1], poly[0, 2], ROD_BORDER_COLOR, ROD_BORDER_THICKNESS, cv2.LINE_AA)
  cv2.line(img, poly[0, 0], poly[0, 3], ROD_BORDER_COLOR, ROD_BORDER_THICKNESS, cv2.LINE_AA)
  draw_hinge(img, hinge1, angle_deg + 90)
  draw_hinge(img, hinge2, angle_deg + 270)


def downsample(img: np.ndarray, downsample_factor: float) -> np.ndarray:
  img = cv2.resize(
      img,
      dsize=None,
      fx=1/downsample_factor,
      fy=1/downsample_factor,
      interpolation=cv2.INTER_AREA
  )
  return img
