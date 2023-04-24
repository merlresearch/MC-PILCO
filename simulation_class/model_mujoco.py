# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL contact:	Diego Romeres (romeres@merl.com)
"""

import time

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers
from scipy import signal
from scipy.integrate import odeint


class Mujoco_Model:
    """
    MuJoCo Gym environment
    """

    def __init__(self, env_name, sim_timestep):
        """
        env_name: string containing environment name
        """
        self.env = gym.make(env_name)
        self.sim_timestep = (
            sim_timestep  # simulator timestep, it must be the same defined in the .xml file in envs/assets/...
        )

    def rollout(self, s0, policy, T, dt, noise):
        """
        Generate a rollout of length T (s) for the Mujoco environment 'env_name' with
        control inputs computed by 'policy' and applied with a sampling time 'dt'.
        'noise' defines the standard deviation of a Gaussian measurement noise.
            s0: initial state
            policy: policy function
            T: length rollout (s)
            dt: sampling time (s)
            noise: measurement noise std
        """

        state_dim = len(s0)
        init_pos = s0[0 : int(len(s0) / 2)]
        init_vel = s0[int(len(s0) / 2) :]
        times = np.linspace(0, T, int(T / dt))

        # init MuJoCo simulation
        self.env.frame_skip = int(dt / self.sim_timestep)
        self.env.init_qpos[0 : int(len(s0) / 2)] = init_pos
        self.env.init_qvel[-int(len(s0) / 2) :] = init_vel

        states = self.env.reset().reshape(1, -1)

        noisy_states = states + np.random.randn(state_dim) * noise
        # get initial input
        inputs = np.array([policy(noisy_states[0, :], 0)]).reshape(1, -1)

        for k in range(1, len(times)):
            self.env.render()
            # apply input
            new_state = self.env.step(inputs[k - 1, :])
            noisy_new_state = new_state[0] + np.random.randn(state_dim) * noise
            # append 'new_state' to 'states'
            states = np.append(states, [new_state[0]], axis=0)
            noisy_states = np.append(noisy_states, [noisy_new_state], axis=0)

            # compute next inputs
            u_next = np.array([policy(noisy_states[k, :], k)]).reshape(1, -1)
            # append u_next to 'inputs'
            inputs = np.append(inputs, u_next, axis=0)

        return noisy_states, inputs, states
