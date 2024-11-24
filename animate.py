"""Animation of a double pendulum, saved to an MP4 file.

Note that because of OpenCV compression setting limitations, the output file
will be quite large. You can easily reduce it to a much smaller MP4, while
essentially preserving the same quality, using:

ffmpeg -i animation.mp4 -vcodec libx264 -crf 30 -preset slow animation_small.mp4

To convert to a gif, the following options are recommended:

ffmpeg -i animation.mp4 -vf "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" animation.gif
"""
import cv2
import matplotlib.pyplot as plt
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

# You may comment out all but the first State objects to get an animation with just one pendulum.
INITIAL_STATES = [
  dbl_pendulum_solver.State(
    theta1=np.deg2rad(90),
    theta1_dot=0,
    theta2=np.deg2rad(90),
    theta2_dot=0,
  ),
  dbl_pendulum_solver.State(
    theta1=np.deg2rad(90),
    theta1_dot=0,
    theta2=np.deg2rad(89),
    theta2_dot=0,
  ),
  dbl_pendulum_solver.State(
    theta1=np.deg2rad(90),
    theta1_dot=0,
    theta2=np.deg2rad(88),
    theta2_dot=0,
  ),
]

TRAIL_COLORS = [
  (.0, .8, .8),
  (.0, .5, .8),
  (.0, .0, .8),
]

SPEEDUP_RATIO = 10  # Ratio between simulation time and video time
FRAMES_PER_SEC = 25
SIMULATION_TIME_STEP = 1 / FRAMES_PER_SEC * SPEEDUP_RATIO

IMAGE_RESOLUTION = (2000, 2000)
DOWNSAMPLE_FACTOR = 2
OUTPUT_RESOLUTION = (IMAGE_RESOLUTION[0]//DOWNSAMPLE_FACTOR, IMAGE_RESOLUTION[1]//DOWNSAMPLE_FACTOR)

FIXED_HINGE = (IMAGE_RESOLUTION[0]//2, IMAGE_RESOLUTION[1]//2 - 200)

def main():
  solutions = []
  for initial_state in INITIAL_STATES:
    solutions.append(
        dbl_pendulum_solver.solve(
            DURATION, initial_state, L1, L2, M1, M2,
            num_eval_points=int(DURATION/SIMULATION_TIME_STEP)
        )
    )

  bgimg = np.zeros(IMAGE_RESOLUTION + (3,), dtype=np.float32)
  for j in range(3):
    bgimg[:,:,j].fill(opencv_utils.BACKGROUND_COLOR[j])

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  video = cv2.VideoWriter('animation_wp.mp4', fourcc, FRAMES_PER_SEC, OUTPUT_RESOLUTION)
  trails = [points_trail.PointsTrail(TRAIL_COLORS[i]) for i in range(len(solutions))]
  all_pos_2 = [[], [], []]
  for i in tqdm.tqdm(range(len(solutions[0].t))):
    img = np.copy(bgimg)
    pos1, pos2 = [], []
    hinge1, hinge2 = [], []
    for n, solution in enumerate(solutions):
      pos1.append(L1 * np.asarray([np.sin(solution.theta1[i]), np.cos(solution.theta1[i])]))
      pos2.append(pos1[-1] + L2 * np.asarray([np.sin(solution.theta2[i]), np.cos(solution.theta2[i])]))
      hinge1.append((FIXED_HINGE + pos1[-1] * PIXELS_PER_METER).astype(np.int32))
      hinge2.append((FIXED_HINGE + pos2[-1] * PIXELS_PER_METER).astype(np.int32))
      trails[n].add_point(hinge2[-1])
      trails[n].draw_on_img(img)
      all_pos_2[n].append(pos2[-1])
    for h1, h2 in zip(hinge1, hinge2):
      opencv_utils.draw_rod(img, FIXED_HINGE, h1)
      opencv_utils.draw_rod(img, h1, h2)

    img = opencv_utils.downsample(img, DOWNSAMPLE_FACTOR)
    img = (img * 255).astype(np.uint8)
    # video.write(img)
  
  video.release()
 
  all_pos_2 = np.asarray(all_pos_2)
  print(all_pos_2.shape)
  distances = [[], [], []]
  indexes = [(0,1), (0,2), (1,2)]
  for n, (i, j) in enumerate(indexes):
    distances[n] = np.sqrt(np.sum(np.square(all_pos_2[i,:,:] - all_pos_2[j,:,:]), axis=1))
  distances = np.asarray(distances)
  print(f'{distances.shape=}')
  plt.figure()
  for i in range(len(distances)):
    plt.plot(distances[i], label=f'{indexes[i]}')
  plt.legend()
  plt.show()


if __name__ == '__main__':
  main()
