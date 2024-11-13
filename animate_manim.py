"""Render an animation of a double pendulum."""

from manim import *

import dbl_pendulum_solver


# Solver params
L1 = 2.5
L2 = 2.5
M1 = 1
M2 = 3
DURATION = 30
INITIAL_STATE = dbl_pendulum_solver.State(
    theta1=np.deg2rad(45),
    theta1_dot=0,
    theta2=np.deg2rad(140),
    theta2_dot=0,
)
NUM_EVAL_POINTS = 2000  # There might be cumulative numeric errors if this is too high.
SPEEDUP_RATIO = 0.5  # Ratio between simulation time and video time

# Display params
ROD_WIDTH = 0.5


class Hinge(Circle):
    def __init__(self):
        super().__init__(radius=ROD_WIDTH*0.2, color=WHITE)
        self.set_fill(BLACK, opacity=1)


class Rod(VGroup):
    def __init__(self, length: float):
        super().__init__()
        self.rect = Rectangle(width=ROD_WIDTH, height=length)
        self.rect.round_corners(radius=ROD_WIDTH*0.5)
        self.rect.set_fill(PINK, opacity=0.5)
        self.add(self.rect)
        self.top_hinge = Hinge()
        self.top_hinge.move_to((0, length/2 - ROD_WIDTH*0.5, 0))
        self.add(self.top_hinge)
        self.bottom_hinge = Hinge()
        self.bottom_hinge.move_to((0, -length/2 + ROD_WIDTH*0.5, 0))
        self.add(self.bottom_hinge)
    
    def move_to(self, new_pos, **kwargs):
        new_pos = np.asarray(new_pos, dtype=np.float64)
        new_pos += self.get_center() - self.top_hinge.get_center()
        super().move_to(new_pos, **kwargs)


class DoublePendulum(Scene):
    def __init__(self):
        super().__init__()
        self.solution = dbl_pendulum_solver.solve(
            DURATION, INITIAL_STATE, L1, L2, M1, M2, NUM_EVAL_POINTS)

    def construct(self):
        config.max_files_cached = 2000
        rod1 = Rod(length=L1)
        rod1.rotate(INITIAL_STATE.theta1)
        rod1.move_to((0, 1, 0))
        self.add(rod1)
        rod2 = Rod(length=L2)
        rod2.rotate(INITIAL_STATE.theta2)
        rod2.move_to(rod1.bottom_hinge.get_center())
        self.add(rod2)
        for i in range(1, len(self.solution.t)):
            self.wait((self.solution.t[i] - self.solution.t[i-1]) / SPEEDUP_RATIO)
            
            rod1.rotate(
                angle=self.solution.theta1[i] - self.solution.theta1[i-1],
                about_point=rod1.top_hinge.get_center(),
            )
            rod2.rotate(
                angle=self.solution.theta2[i] - self.solution.theta2[i-1],
                about_point=rod2.top_hinge.get_center(),
            )
            rod2.move_to(rod1.bottom_hinge.get_center())
