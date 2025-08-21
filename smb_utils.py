import numpy as np
import matplotlib.pyplot as plt

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


class MarioGrid:
    """
    Class for constructing a grid-based view of the Super Mario Bros RAM state.
    Converts the raw RAM values into a simplified 2D representation of the
    visible screen, including background tiles, Mario, and enemies.
    """

    def __init__(self, env):
        self.ram = env.unwrapped.ram

        # Dimensions of the rendered viewport
        self.screen_width = 16
        self.screen_height = 13

        # Marios absolute position in the level (in pixels)
        self.level_x = self.ram[0x6d] * 256 + self.ram[0x86]

        # Marios on-screen pixel position
        self.mario_x = self.ram[0x3ad]
        self.mario_y = self.ram[0x3b8] + 16  # offset to account for Mario’s sprite height

        # Leftmost pixel column of the current viewport
        self.left_edge = self.level_x - self.mario_x

        # Construct screen on init
        self.grid = self._render_screen()


    # Helper methods

    def _tile_to_address(self, x, y):

        page = x // 16
        x_pos = x % 16
        y_pos = page * 13 + y
        return 0x500 + x_pos + y_pos * 16

    def _render_screen(self):

        screen = np.zeros((self.screen_height, self.screen_width))
        screen_start_tile = int(np.rint(self.left_edge / 16))

        # Background tiles
        for i in range(self.screen_width):
            for j in range(self.screen_height):
                tile_x = (screen_start_tile + i) % (self.screen_width * 2)
                tile_y = j
                addr = self._tile_to_address(tile_x, tile_y)

                if self.ram[addr] != 0:
                    screen[j, i] = 1  # mark any non-zero tile as solid

        # Mario
        mario_tile_x = (self.mario_x + 8) // 16
        mario_tile_y = (self.mario_y - 32) // 16  # top two rows are not stored in RAM
        if mario_tile_x < 16 and mario_tile_y < 13:
            screen[mario_tile_y, mario_tile_x] = 2

        # Enemies (up to 5 tracked in RAM)
        for i in range(5):
            if self.ram[0xF + i] == 1:  # enemy active flag
                enemy_x = self.ram[0x6e + i] * 256 + self.ram[0x87 + i] - self.left_edge
                enemy_y = self.ram[0xcf + i]

                ex = (enemy_x + 8) // 16
                ey = (enemy_y + 8 - 32) // 16

                if 0 <= ex < 16 and 0 <= ey < 13:
                    screen[ey, ex] = -1

        return screen
