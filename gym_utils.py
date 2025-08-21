
# Custom wrapper for cropping and stacking RAM observations
import time
import imageio
import matplotlib.pyplot as plt
from matplotlib import colors

import gym
from gym import spaces
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import obs_as_tensor

from smb_utils import *
import numpy as np


class MarioRamWrapper(gym.ObservationWrapper):
    def __init__(self, env, crop_window=[0, 16, 0, 13], stack_size=4, frame_gap=2):
        """
        crop_window: [x_start, x_end, y_start, y_end]
        Final observation shape -> (height, width, stack_size)
        stack_size = number of frames stored
        frame_gap = interval between consecutive frames used in the stack
        """
        super().__init__(env)
        self.crop_window = crop_window
        self.stack_size = stack_size
        self.frame_gap = frame_gap
        self.width = crop_window[1] - crop_window[0]
        self.height = crop_window[3] - crop_window[2]

        self.observation_space = spaces.Box(
            low=-1,
            high=2,
            shape=(self.height, self.width, stack_size),
            dtype=int
        )

        # Extra frames to allow skipping
        self.frames = np.zeros((self.height, self.width, (stack_size - 1) * frame_gap + 1))

    def observation(self, obs):
        # Extract grid and crop current frame
        grid = smb_grid(self.env)
        current_frame = self._crop_frame(grid.rendered_screen)

        # Shift stored frames and insert new one
        self.frames[:, :, 1:] = self.frames[:, :, :-1]
        self.frames[:, :, 0] = current_frame

        # Take every `frame_gap`-th frame
        return self.frames[:, :, ::self.frame_gap]

    def reset(self):
        self.env.reset()
        self.frames = np.zeros_like(self.frames)
        grid = smb_grid(self.env)
        initial_frame = self._crop_frame(grid.rendered_screen)

        # Fill stack with the same initial frame
        for i in range(self.frames.shape[-1]):
            self.frames[:, :, i] = initial_frame

        return self.frames[:, :, ::self.frame_gap]

    def _crop_frame(self, image):
        """ Reduce input by cropping observed screen """
        x0, x1, y0, y1 = self.crop_window
        return image[y0:y1, x0:x1]


def init_mario_env(env_name='SuperMarioBros-1-1-v0',
                   crop_window=[0, 16, 0, 13],
                   stack_size=2,
                   frame_gap=4):
    """
    Build SMB environment with preprocessing wrappers
    """
    base_env = gym_super_mario_bros.make(env_name)
    base_env = JoypadSpace(base_env, SIMPLE_MOVEMENT)
    wrapped_env = MarioRamWrapper(base_env, crop_window, stack_size, frame_gap)
    return DummyVecEnv([lambda: wrapped_env])


class MarioAgent:
    """
    Container class holding both the trained agent and its environment
    """
    def __init__(self, env, model):
        self.env = env
        self.model = model

    def run(self, episodes=5, deterministic=False, show=True, return_stats=False):
        for ep in range(1, episodes + 1):
            state = self.env.reset()
            done, score = False, 0

            while not done:
                if show:
                    self.env.render()

                action, _ = self.model.predict(state, deterministic=deterministic)
                state, reward, done, info = self.env.step(action)
                score += reward

                if show:
                    time.sleep(0.01)

            if show:
                print(f"Episode {ep}: Score {score}")

        return (score, info) if return_stats else None

    def evaluate(self, episodes=20, deterministic=False):
        """ Return per-episode rewards and step counts """
        return evaluate_policy(
            self.model,
            self.env,
            n_eval_episodes=episodes,
            deterministic=deterministic,
            render=False,
            return_episode_rewards=True
        )

    def action_probabilities(self, state):
        """ Get action distribution for given state """
        obs_tensor = obs_as_tensor(state, self.model.policy.device)
        distribution = self.model.policy.get_distribution(obs_tensor)
        return distribution.distribution.probs.detach().numpy()


    # Visualisation utilities

    def record_episode_frames(self, deterministic=False):
        """
        Generate frames showing both agent's view and environment rendering
        """
        state = self.env.reset()
        done, score = False, [0]

        while not done:
            probs = self.action_probabilities(state)
            action, _ = self.model.predict(state, deterministic=deterministic)
            state, reward, done, _ = self.env.step(action)
            score += reward
            self._plot_combined(state, score, probs)

    def _plot_combined(self, state, score, probs):
        """ Show stacked frames, action distribution, and rendered environment """
        render_img = self.env.render(mode="rgb_array")
        n_frames = state.shape[-1]

        cmap = colors.ListedColormap(['red', 'skyblue', 'brown', 'blue'])
        norm = colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)

        frame_labels = ['t', 't-4', 't-8', 't-12']
        actions = ['NOOP', '→', '→+A', '→+B', '→+A+B', 'A', '←']

        fig = plt.figure(dpi=100, figsize=(6, 6))
        gs = fig.add_gridspec(4, 2, width_ratios=[3, 1])

        # Show stacked observations
        for i in range(n_frames):
            ax = fig.add_subplot(gs[i, 1])
            ax.imshow(state[0, :, :, i], cmap=cmap, norm=norm)
            ax.set_axis_off()
            ax.text(-0.5, 14.5, frame_labels[i])

        # Action probability distribution
        ax = fig.add_subplot(gs[3, 0])
        ax.bar(actions, probs[0])
        plt.xticks(rotation=45)
        ax.set_ylim(0, 1.05)

        # Rendered game screen
        ax = fig.add_subplot(gs[0:3, 0])
        ax.imshow(render_img)
        ax.set_axis_off()
        ax.text(0, -5, f"score: {int(score[0])}")

        plt.show()

    def create_gif(self, deterministic=True, filename='mario_run.gif', return_frames=False):
        """
        Save an episode as a gif animation
        """
        frames = []
        state = self.env.reset()
        done = False

        while not done:
            frame = self.env.render(mode="rgb_array")
            frames.append(frame.copy())
            action, _ = self.model.predict(state, deterministic=deterministic)
            state, _, done, _ = self.env.step(action)

        if not return_frames:
            imageio.mimsave(filename, frames, fps=50)
        else:
            return frames


    