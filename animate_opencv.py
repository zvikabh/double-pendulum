"""Animation of a double pendulum, saved to an MP4 file.

Note that because of OpenCV compression setting limitations, the output file
will be quite large. You can easily reduce it to a smaller MP4 using:

ffmpeg -i animation.mp4 -vcodec libx264 -crf 30 -preset slow animation_small.mp4
"""
import cv2
import numpy as np
import tqdm

import dbl_pendulum_solver
import opencv_utils


L1 = 200
L2 = 300
M1 = 100
M2 = 300
PIXELS_PER_METER = 1.5
DURATION = 450
INITIAL_STATE = dbl_pendulum_solver.State(
    theta1=np.deg2rad(45),
    theta1_dot=0,
    theta2=np.deg2rad(140),
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

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  video = cv2.VideoWriter('animation.mp4', fourcc, FRAMES_PER_SEC, OUTPUT_RESOLUTION)
  for i in tqdm.tqdm(range(len(solution.t))):
    img = np.zeros(IMAGE_RESOLUTION + (3,), dtype=np.float32)
    pos1 = L1 * np.asarray([np.sin(solution.theta1[i]), np.cos(solution.theta1[i])])
    pos2 = pos1 + L2 * np.asarray([np.sin(solution.theta2[i]), np.cos(solution.theta2[i])])
    hinge1 = FIXED_HINGE + pos1 * PIXELS_PER_METER
    hinge2 = FIXED_HINGE + pos2 * PIXELS_PER_METER
    hinge1 = hinge1.astype(np.int32)
    hinge2 = hinge2.astype(np.int32)
    opencv_utils.draw_rod(img, FIXED_HINGE, hinge1)
    opencv_utils.draw_rod(img, hinge1, hinge2)
    img = opencv_utils.downsample(img, DOWNSAMPLE_FACTOR)
    img = (img * 255).astype(np.uint8)
    video.write(img)
  
  video.release()


if __name__ == '__main__':
  main()
