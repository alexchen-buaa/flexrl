'''
Apple-Key-Door-Treasure Maze env with flattened pixel input (13*13*3)
'''

import numpy as np
import cv2
from gym import Env, spaces


class AKDTMazeFlat(Env):

    def __init__(self):
        super(AKDTMazeFlat, self).__init__()
        # Env specs
        self.wall_pos = [(x, 12) for x in range(13)] +\
            [(x, 0) for x in range(13)] +\
            [(0, y) for y in range(13)] +\
            [(12, y) for y in range(13)] +\
            [(x, 6) for x in [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]] +\
            [(6, y) for y in [1, 3, 4, 5, 6, 7, 8, 9, 10]]
        self.key_pos = (11, 11)
        self.door_pos = (6, 11)
        self.trea_pos = (1, 7)
        self.apple_pos = [(4, 2), (2, 4)]
        self.init_pos = (8, 2)
        # Receives rgb pixel input between 0 and 1
        self.canvas_shape = (13, 13, 3)
        self.observation_shape = (507,)  # 13*13*3
        low = np.zeros(self.observation_shape)
        high = np.ones(self.observation_shape)
        self.observation_space = spaces.Box(low, high, dtype=np.float64)
        self.action_space = spaces.Discrete(4,)  # 0 - 3
        # Env specific variables
        self.key_count = None
        self.door_count = None
        self.apple_count = None
        self.agent_pos = None
        self.canvas = None

    def reset(self):
        self.canvas = np.ones(self.canvas_shape)
        # Locate the agent
        self.key_count = 1
        self.door_count = 1
        self.apple_count = [1, 1]
        self.agent_pos = self.init_pos
        # Draw elements onto the canvas
        for pos in self.wall_pos:
            self.canvas[pos] *= 0
        for pos in self.apple_pos:
            self.canvas[pos] = np.array([242, 143, 124], dtype=np.float64) / 255.0
        self.canvas[self.key_pos] = np.array([120, 183, 247], dtype=np.float64) / 255.0
        self.canvas[self.door_pos] = np.array([129, 214, 138], dtype=np.float64) / 255.0
        self.canvas[self.trea_pos] = np.array([255, 253, 95], dtype=np.float64) / 255.0
        self.canvas[self.init_pos] = np.array([110, 110, 110], dtype=np.float64) / 255.0
        return self.canvas.flatten()

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array'], 'Invalid mode'
        if mode == 'human':
            img = cv2.resize(self.canvas, (260, 260),
                             interpolation=cv2.INTER_AREA)
            img = img[..., ::-1]  # bgr to rgb
            cv2.imshow('KDTMazePixel', img)
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            img = cv2.resize(self.canvas, (260, 260),
                             interpolation=cv2.INTER_AREA)
            return img

    def close(self):
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def step(self, action):
        reward = 0  # Sparse reward
        done = False  # Episode termination flag
        assert self.action_space.contains(action), 'Invalid action'

        # Compute next position
        x, y = self.agent_pos
        if action == 0:
            x -= 1  # Up
        elif action == 1:
            x += 1  # Down
        elif action == 2:
            y -= 1  # Left
        elif action == 3:
            y += 1  # Right
        next_pos = (x, y)

        # Moving and capturing
        if not (next_pos in self.wall_pos):
            if not ((next_pos == self.door_pos) and self.key_count):
                # If not wall and locked door:
                # Capturing
                if next_pos in self.apple_pos:
                    idx = self.apple_pos.index(next_pos)
                    if self.apple_count[idx]:
                        self.apple_count[idx] = 0
                        reward = 1
                if (next_pos == self.key_pos) and self.key_count:
                    self.key_count = 0
                    reward = 2
                elif (next_pos == self.door_pos) and self.door_count:
                    self.door_count = 0
                    reward = 2
                elif next_pos == self.trea_pos:
                    reward = 3
                    done = True
                # Moving and switching canvas color
                self.canvas[next_pos] = self.canvas[self.agent_pos]
                self.canvas[self.agent_pos] = np.array([1, 1, 1], dtype=np.float64)
                self.agent_pos = next_pos
        return self.canvas.flatten(), reward, done, {}  # Info has to be dict
