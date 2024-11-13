import time

import cv2
import numpy as np
import tqdm

import dbl_pendulum_solver
import opencv_utils


L1 = 200
L2 = 300
M1 = 100
M2 = 300
DURATION = 600
INITIAL_STATE = dbl_pendulum_solver.State(
    theta1=np.deg2rad(45),
    theta1_dot=0,
    theta2=np.deg2rad(140),
    theta2_dot=0,
)
SPEEDUP_RATIO = 10  # Ratio between simulation time and video time
FRAMES_PER_SEC = 25
SIMULATION_TIME_STEP = 1 / FRAMES_PER_SEC * SPEEDUP_RATIO

IMAGE_RESOLUTION = (1000, 1000)
FIXED_HINGE = (IMAGE_RESOLUTION[0]//2, 300)


def main():
  solution = dbl_pendulum_solver.solve(
      DURATION, INITIAL_STATE, L1, L2, M1, M2,
      num_eval_points=int(DURATION/SIMULATION_TIME_STEP))

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  video = cv2.VideoWriter('animation.mp4', fourcc, FRAMES_PER_SEC, IMAGE_RESOLUTION)
  for i in tqdm.tqdm(range(len(solution.t))):
    img = np.zeros(IMAGE_RESOLUTION + (3,), dtype=np.uint8)
    hinge1 = (FIXED_HINGE[0] + L1*np.sin(solution.theta1[i]), FIXED_HINGE[1] + L1*np.cos(solution.theta1[i]))
    hinge2 = (hinge1[0] + L2*np.sin(solution.theta2[i]), hinge1[1] + L2*np.cos(solution.theta2[i]))
    hinge1 = np.asarray(hinge1, dtype=np.int32)
    hinge2 = np.asarray(hinge2, dtype=np.int32)
    opencv_utils.draw_rod(img, FIXED_HINGE, hinge1)
    opencv_utils.draw_rod(img, hinge1, hinge2)
    video.write(img)
  
  video.release()


if __name__ == '__main__':
  main()
