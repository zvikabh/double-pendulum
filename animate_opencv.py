"""Animation of a double pendulum, saved to an MP4 file.

Note that because of OpenCV compression setting limitations, the output file
will be quite large. You can easily reduce it to a much smaller MP4, while
essentially preserving the same quality, using:

ffmpeg -i animation.mp4 -vcodec libx264 -crf 30 -preset slow animation_small.mp4
"""
import cv2
import numpy as np
import tqdm

import dbl_pendulum_solver
import opencv_utils
import points_trail


L1 = 250
L2 = 250
M1 = 200
M2 = 200
PIXELS_PER_METER = 1.5
DURATION = 400
INITIAL_STATE = dbl_pendulum_solver.State(
    theta1=np.deg2rad(90),
    theta1_dot=0,
    theta2=np.deg2rad(90),
    theta2_dot=0,
)
SPEEDUP_RATIO = 10  # Ratio between simulation time and video time
FRAMES_PER_SEC = 25
SIMULATION_TIME_STEP = 1 / FRAMES_PER_SEC * SPEEDUP_RATIO

IMAGE_RESOLUTION = (2000, 2000)
DOWNSAMPLE_FACTOR = 2
OUTPUT_RESOLUTION = (IMAGE_RESOLUTION[0]//DOWNSAMPLE_FACTOR, IMAGE_RESOLUTION[1]//DOWNSAMPLE_FACTOR)

FIXED_HINGE = (IMAGE_RESOLUTION[0]//2, IMAGE_RESOLUTION[1]//2 - 200)

def main():
  solution = dbl_pendulum_solver.solve(
      DURATION, INITIAL_STATE, L1, L2, M1, M2,
      num_eval_points=int(DURATION/SIMULATION_TIME_STEP))

  bgimg = np.zeros(IMAGE_RESOLUTION + (3,), dtype=np.float32)
  for j in range(3):
    bgimg[:,:,j].fill(opencv_utils.BACKGROUND_COLOR[j])

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  video = cv2.VideoWriter('animation_wp.mp4', fourcc, FRAMES_PER_SEC, OUTPUT_RESOLUTION)
  trail = points_trail.PointsTrail()
  for i in tqdm.tqdm(range(len(solution.t))):
    img = np.copy(bgimg)
    pos1 = L1 * np.asarray([np.sin(solution.theta1[i]), np.cos(solution.theta1[i])])
    pos2 = pos1 + L2 * np.asarray([np.sin(solution.theta2[i]), np.cos(solution.theta2[i])])
    hinge1 = FIXED_HINGE + pos1 * PIXELS_PER_METER
    hinge2 = FIXED_HINGE + pos2 * PIXELS_PER_METER
    hinge1 = hinge1.astype(np.int32)
    hinge2 = hinge2.astype(np.int32)
    trail.add_point(hinge2)
    trail.draw_on_img(img)
    opencv_utils.draw_rod(img, FIXED_HINGE, hinge1)
    opencv_utils.draw_rod(img, hinge1, hinge2)
    img = opencv_utils.downsample(img, DOWNSAMPLE_FACTOR)
    img = (img * 255).astype(np.uint8)
    video.write(img)
  
  video.release()


if __name__ == '__main__':
  main()
