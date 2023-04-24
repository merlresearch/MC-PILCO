# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL:	    Diego Romeres (romeres@merl.com)
"""


"""
Test MC-PILCO on a MuJoCo UR5 robot for learning a joint-space controller
"""

import argparse
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch
from gym.envs.registration import register

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import gpr_lib.Utils.Parameters_covariance_functions as cov_func
import model_learning.Model_learning as ML
import policy_learning.Cost_function as Cost_function
import policy_learning.MC_PILCO_mujoco_envs as MC_PILCO_mujoco_envs
import policy_learning.Policy as Policy

# Register gym environment
register(
    id="UR5_Env-v0",
    entry_point="envs.ur5:UR5_Env",
)

# Load random seed from command line
p = argparse.ArgumentParser("test ur5 mujoco")
p.add_argument("-seed", type=int, default=1, help="seed")
locals().update(vars(p.parse_known_args()[0]))

# Set the seed
torch.manual_seed(seed)
np.random.seed(seed)

# Default data type
dtype = torch.float64

# Set the device
device = torch.device("cpu")
# device=torch.device('cuda:0')

# Set number of computational threads
num_threads = 1
torch.set_num_threads(num_threads)

print("---- Set environment parameters ----")
num_trials = 2
T_sampling = 0.02
T_control = 4.0
T_exploration = T_control
state_dim = 12
input_dim = 6
num_gp = 6
gp_input_dim = state_dim + int(state_dim / 2) + input_dim
env_name = "UR5_Env-v0"
sim_timestep = 0.001  # Simulator timestep, it must be the same defined in:envs/assets/UR5.xml
u_max = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
std_noise = 10 ** (-3)
std_list = std_noise * np.ones(state_dim)

print("\n---- Set model learning parameters ----")
f_model_learning = ML.Speed_Model_learning_RBF_MPK_angle_state
model_learning_par = {}
model_learning_par["num_gp"] = num_gp
model_learning_par["angle_indeces"] = [0, 1, 2, 3, 4, 5]
model_learning_par["not_angle_indeces"] = [6, 7, 8, 9, 10, 11]
model_learning_par["T_sampling"] = T_sampling
model_learning_par["vel_indeces"] = [6, 7, 8, 9, 10, 11]
model_learning_par["not_vel_indeces"] = [0, 1, 2, 3, 4, 5]
model_learning_par["device"] = device
model_learning_par["dtype"] = dtype
model_learning_par["approximation_mode"] = "SOD"
model_learning_par["approximation_dict"] = {
    "SOD_threshold_mode": "absolute",
    "SOD_threshold": [0.001] * num_gp,
    "flg_SOD_permutation": False,
}  # Set SoD threshold
# RBF initial par
init_dict_RBF = {}
init_dict_RBF["active_dims"] = np.arange(0, gp_input_dim)
init_dict_RBF["lengthscales_init"] = np.ones(init_dict_RBF["active_dims"].size)
init_dict_RBF["flg_train_lengthscales"] = True
init_dict_RBF["lambda_init"] = np.ones(1)
init_dict_RBF["flg_train_lambda"] = False
init_dict_RBF["sigma_n_init"] = 1 * np.ones(1)
init_dict_RBF["flg_train_sigma_n"] = True
init_dict_RBF["dtype"] = dtype
init_dict_RBF["device"] = device
# MPK initial par
init_dict_MPK = {}
init_dict_MPK["active_dims"] = np.arange(0, gp_input_dim)
init_dict_MPK["poly_deg"] = 1
init_dict_MPK["Sigma_pos_par_init_list"] = [np.ones(gp_input_dim + 1)] + [
    np.ones((deg + 1) * (gp_input_dim)) for deg in range(1, init_dict_MPK["poly_deg"])
]
init_dict_MPK["flg_train_Sigma_pos_par_list"] = [True] * init_dict_MPK["poly_deg"]
init_dict_MPK["dtype"] = dtype
init_dict_MPK["device"] = device
model_learning_par["init_dict_list"] = [[init_dict_RBF, init_dict_MPK]] * num_gp

print("\n---- Load target trajectory ----")
target_traj = np.genfromtxt("envs/target_q_trajectory.csv", delimiter=",")

print("\n---- Set exploration policy ----")
f_rand_exploration_policy = Policy.PD_controller
rand_exploration_policy_par = {}
rand_exploration_policy_par["state_dim"] = state_dim
rand_exploration_policy_par["input_dim"] = input_dim
rand_exploration_policy_par["u_max"] = u_max
rand_exploration_policy_par["dtype"] = dtype
rand_exploration_policy_par["device"] = device
rand_exploration_policy_par["sqrt_Kp_gains"] = [1.0, 1.0, 1.0, 1.0, 1, 1.0]
rand_exploration_policy_par["sqrt_Kd_gains"] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
rand_exploration_policy_par["target_traj"] = torch.tensor(target_traj, dtype=dtype, device=device)

print("\n---- Set control policy ----")
num_basis = 400
f_control_policy = Policy.Sum_of_gaussians_with_target_trajectory
control_policy_par = {}
control_policy_par["state_dim"] = state_dim * 2
control_policy_par["input_dim"] = input_dim
control_policy_par["u_max"] = u_max
control_policy_par["num_basis"] = num_basis
control_policy_par["dtype"] = dtype
control_policy_par["device"] = device
control_policy_par["centers_init"] = np.concatenate(
    [
        np.pi / 2 * 2 * (np.random.rand(num_basis, state_dim) - 0.5),
        0.1 * 2 * (np.random.rand(num_basis, state_dim) - 0.5),
    ],
    1,
)
control_policy_par["lengthscales_init"] = np.pi * np.ones(state_dim * 2)
control_policy_par["weight_init"] = 1.0 * 2 * (np.random.rand(input_dim, num_basis) - 0.5)
control_policy_par["target_traj"] = torch.tensor(target_traj, dtype=dtype, device=device)
control_policy_par["flg_squash"] = True
control_policy_par["flg_drop"] = True
policy_reinit_dict = {}
policy_reinit_dict["lenghtscales_par"] = control_policy_par["lengthscales_init"]
policy_reinit_dict["centers_par"] = np.concatenate(
    [np.pi / 2 * np.ones([num_basis, state_dim]), 0.1 * np.ones([num_basis, state_dim])], 1
)
policy_reinit_dict["weight_par"] = 1.0

print("\n---- Set cost function ----")
cost_function_par = {}
f_cost_function = Cost_function.Expected_saturated_distance_from_trajectory
cost_function_par["target_traj"] = torch.tensor(target_traj, dtype=dtype, device=device)
cost_function_par["lengthscales"] = torch.tensor(
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype, device=device
)
cost_function_par["used_indeces"] = list(range(0, 12))

print("\n---- Init policy learning object ----")
MC_PILCO_init_dict = {}
MC_PILCO_init_dict["T_sampling"] = T_sampling
MC_PILCO_init_dict["state_dim"] = state_dim
MC_PILCO_init_dict["input_dim"] = input_dim
MC_PILCO_init_dict["f_sim"] = env_name
MC_PILCO_init_dict["std_meas_noise"] = std_list
MC_PILCO_init_dict["f_model_learning"] = f_model_learning
MC_PILCO_init_dict["model_learning_par"] = model_learning_par
MC_PILCO_init_dict["f_rand_exploration_policy"] = f_rand_exploration_policy
MC_PILCO_init_dict["rand_exploration_policy_par"] = rand_exploration_policy_par
MC_PILCO_init_dict["f_control_policy"] = f_control_policy
MC_PILCO_init_dict["control_policy_par"] = control_policy_par
MC_PILCO_init_dict["f_cost_function"] = f_cost_function
MC_PILCO_init_dict["cost_function_par"] = cost_function_par
MC_PILCO_init_dict["log_path"] = "results_tmp/" + str(seed)
MC_PILCO_init_dict["dtype"] = dtype
MC_PILCO_init_dict["device"] = device
MC_PILCO_init_dict["sim_timestep"] = sim_timestep
PL_obj = MC_PILCO_mujoco_envs.MC_PILCO_Mujoco(**MC_PILCO_init_dict)

print("\n---- Set MC-PILCO options ----")
# Model optimization options
model_optimization_opt_dict = {}
model_optimization_opt_dict["f_optimizer"] = "lambda p : torch.optim.Adam(p, lr=0.01)"
model_optimization_opt_dict["criterion"] = Likelihood.Marginal_log_likelihood
model_optimization_opt_dict["N_epoch"] = 2001
model_optimization_opt_dict["N_epoch_print"] = 500
model_optimization_opt_list = [model_optimization_opt_dict] * num_gp
# Policy optimization options
policy_optimization_dict = {}
policy_optimization_dict["num_particles"] = 200
policy_optimization_dict["opt_steps_list"] = [5000, 5000]
policy_optimization_dict["lr_list"] = [0.01, 0.01]
policy_optimization_dict["f_optimizer"] = "lambda p, lr : torch.optim.Adam(p, lr)"
policy_optimization_dict["num_step_print"] = 200
policy_optimization_dict["p_dropout_list"] = [0.25, 0.25]
policy_optimization_dict["p_drop_reduction"] = 0.25 / 2
policy_optimization_dict["alpha_diff_cost"] = 0.99
policy_optimization_dict["min_diff_cost"] = 0.04
policy_optimization_dict["num_min_diff_cost"] = 400
policy_optimization_dict["min_step"] = 400
policy_optimization_dict["lr_reduction_ratio"] = 0.5
policy_optimization_dict["lr_min"] = 0.0025
policy_optimization_dict["policy_reinit_dict"] = policy_reinit_dict
# Options for method reinforce
reinforce_param_dict = {}
reinforce_param_dict["initial_state"] = target_traj[0, :]
reinforce_param_dict["initial_state_var"] = np.array(
    [
        10**-6,
        10**-6,
        10**-6,
        10**-6,
        10**-6,
        10**-6,
        10**-6,
        10**-6,
        10**-6,
        10**-6,
        10**-6,
        10**-6,
    ]
)
reinforce_param_dict["T_exploration"] = T_exploration
reinforce_param_dict["T_control"] = T_control
reinforce_param_dict["num_trials"] = num_trials
reinforce_param_dict["random_initial_state"] = False
reinforce_param_dict["model_optimization_opt_list"] = model_optimization_opt_list
reinforce_param_dict["policy_optimization_dict"] = policy_optimization_dict

print("\n---- Save test configuration ----")
config_log_dict = {}
config_log_dict["MC_PILCO_init_dict"] = MC_PILCO_init_dict
config_log_dict["reinforce_param_dict"] = reinforce_param_dict
pkl.dump(config_log_dict, open("results_tmp/" + str(seed) + "/config_log.pkl", "wb"))

# Start the learning algorithm
PL_obj.reinforce(**reinforce_param_dict)
