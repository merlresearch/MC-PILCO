# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


# CUSTOM ENV #
class CartpoleSwingupEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(
            self, "%s/assets/cartpole_swingup.xml" % dir_path, 5
        )  # the number is self.frame_skip

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        # we only collect observations, we don't need 'reward' and 'done'
        reward = 0.0
        done = False
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent
