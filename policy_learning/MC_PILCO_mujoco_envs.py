# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL contact:	Diego Romeres (romeres@merl.com)
"""
import sys

import torch

sys.path.append("..")
import copy
import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.linalg import block_diag
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import simulation_class.model_mujoco as model
from policy_learning.MC_PILCO import MC_PILCO


class MC_PILCO_Mujoco(MC_PILCO):
    """
    MC-PILCO implementation for Mujoco Environment
    """

    def __init__(
        self,
        T_sampling,
        state_dim,
        input_dim,
        f_sim,
        sim_timestep,
        f_model_learning,
        model_learning_par,
        f_rand_exploration_policy,
        rand_exploration_policy_par,
        f_control_policy,
        control_policy_par,
        f_cost_function,
        cost_function_par,
        std_meas_noise=None,
        log_path=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):
        super(MC_PILCO_Mujoco, self).__init__(
            T_sampling=T_sampling,
            state_dim=state_dim,
            input_dim=input_dim,
            f_sim=f_sim,
            f_model_learning=f_model_learning,
            model_learning_par=model_learning_par,
            f_rand_exploration_policy=f_rand_exploration_policy,
            rand_exploration_policy_par=rand_exploration_policy_par,
            f_control_policy=f_control_policy,
            control_policy_par=control_policy_par,
            f_cost_function=f_cost_function,
            cost_function_par=cost_function_par,
            std_meas_noise=std_meas_noise,
            log_path=log_path,
            dtype=dtype,
            device=device,
        )

        self.system = model.Mujoco_Model(f_sim, sim_timestep)  # MuJoCo-simulated system
