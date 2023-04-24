# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import absolute_import, division, print_function

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class UR5_Env(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.viewer = None
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, "%s/assets/UR5.xml" % dir_path, 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward = 0.0
        done = False
        return ob, reward, done, dict(reward_dist=0, reward_ctrl=0)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = 2.5
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 270

    def reset_model(self):
        qpos, qvel = np.copy(self.init_qpos), np.copy(self.init_qvel)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])
