# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL:	    Diego Romeres (romeres@merl.com)
"""


"""
Test MC-PILCO on a simulated cart-pole system (GPs equipped with square-exponential kernels)
"""

import argparse
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import gpr_lib.Utils.Parameters_covariance_functions as cov_func
import model_learning.Model_learning as ML
import policy_learning.Cost_function as Cost_function
import policy_learning.MC_PILCO as MC_PILCO
import policy_learning.Policy as Policy
import simulation_class.ode_systems as f_ode

# Load random seed from command line
p = argparse.ArgumentParser("test cartpole")
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
num_trials = 5
T_sampling = 0.05
T_exploration = 3.0
T_control = 3.0
state_dim = 4
input_dim = 1
num_gp = int(state_dim / 2)
gp_input_dim = 6
ode_fun = f_ode.cartpole
u_max = 10.0
std_noise = 10 ** (-2)
std_list = [std_noise, std_noise, std_noise, std_noise]
fl_SOD_GP = True  # Flag to select if to use or not a Subset of Data (SoD) approximation in the GPs

print("---- Set model learning parameters ----")
f_model_learning = ML.Speed_Model_learning_RBF_angle_state
print(f_model_learning)
model_learning_par = {}
model_learning_par["num_gp"] = num_gp
model_learning_par["angle_indeces"] = [2]
model_learning_par["not_angle_indeces"] = [0, 1, 3]
model_learning_par["T_sampling"] = T_sampling
model_learning_par["vel_indeces"] = [1, 3]
model_learning_par["not_vel_indeces"] = [0, 2]
model_learning_par["device"] = device
model_learning_par["dtype"] = dtype
if fl_SOD_GP:
    model_learning_par["approximation_mode"] = "SOD"
    model_learning_par["approximation_dict"] = {
        "SOD_threshold_mode": "relative",
        "SOD_threshold": 0.5,
        "flg_SOD_permutation": False,
    }  # Set SoD threshold
init_dict = {}
# RBF initial par
init_dict["active_dims"] = np.arange(0, gp_input_dim)
init_dict["lengthscales_init"] = np.ones(init_dict["active_dims"].size)
init_dict["flg_train_lengthscales"] = True
init_dict["lambda_init"] = np.ones(1)
init_dict["flg_train_lambda"] = False
init_dict["sigma_n_init"] = 1 * np.ones(1)
init_dict["sigma_n_num"] = None
init_dict["flg_train_sigma_n"] = True
init_dict["dtype"] = dtype
init_dict["device"] = device
model_learning_par["init_dict_list"] = [init_dict] * num_gp

print("\n---- Set exploration policy ----")
f_rand_exploration_policy = Policy.Random_exploration
rand_exploration_policy_par = {}
rand_exploration_policy_par["state_dim"] = state_dim
rand_exploration_policy_par["input_dim"] = input_dim
rand_exploration_policy_par["u_max"] = u_max
rand_exploration_policy_par["dtype"] = dtype
rand_exploration_policy_par["device"] = device

print("\n---- Set control policy ----")
num_basis = 200
f_control_policy = Policy.Sum_of_gaussians_with_angles
control_policy_par = {}
control_policy_par["state_dim"] = state_dim
control_policy_par["input_dim"] = input_dim
control_policy_par["angle_indices"] = np.array([2])
control_policy_par["non_angle_indices"] = np.array([0, 1, 3])
control_policy_par["u_max"] = u_max
control_policy_par["num_basis"] = num_basis
control_policy_par["dtype"] = dtype
control_policy_par["device"] = device
angle_centers = np.pi * 2 * (np.random.rand(num_basis, 1) - 0.5)
cos_centers = np.cos(angle_centers)
sin_centers = np.sin(angle_centers)
not_angle_centers = np.pi * 2 * (np.random.rand(num_basis, 3) - 0.5)
control_policy_par["centers_init"] = np.concatenate([not_angle_centers, cos_centers, sin_centers], 1)
control_policy_par["lengthscales_init"] = 1 * np.ones(state_dim + 1)
control_policy_par["weight_init"] = u_max * (np.random.rand(input_dim, num_basis) - 0.5)
control_policy_par["flg_squash"] = True
control_policy_par["flg_drop"] = True
policy_reinit_dict = {}
policy_reinit_dict["lenghtscales_par"] = control_policy_par["lengthscales_init"]
policy_reinit_dict["centers_par"] = np.array([np.pi, np.pi, np.pi, 1.0, 1.0])
policy_reinit_dict["weight_par"] = u_max

print("\n---- Set cost function ----")
f_cost_function = Cost_function.Cart_pole_cost
cost_function_par = {}
cost_function_par["pos_index"] = 0
cost_function_par["angle_index"] = 2
cost_function_par["target_state"] = torch.tensor([np.pi, 0.0], dtype=dtype, device=device)
cost_function_par["lengthscales"] = torch.tensor([3.0, 1.0], dtype=dtype, device=device)

print("\n---- Init policy learning object ----")
MC_PILCO_init_dict = {}
MC_PILCO_init_dict["T_sampling"] = T_sampling
MC_PILCO_init_dict["state_dim"] = state_dim
MC_PILCO_init_dict["input_dim"] = input_dim
MC_PILCO_init_dict["f_sim"] = ode_fun
MC_PILCO_init_dict["std_meas_noise"] = np.array(std_list)
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
PL_obj = MC_PILCO.MC_PILCO(**MC_PILCO_init_dict)

print("\n---- Set MC-PILCO options ----")
# Model optimization options
model_optimization_opt_dict = {}
model_optimization_opt_dict["f_optimizer"] = "lambda p : torch.optim.Adam(p, lr=0.01)"
model_optimization_opt_dict["criterion"] = Likelihood.Marginal_log_likelihood
model_optimization_opt_dict["N_epoch"] = 1501
model_optimization_opt_dict["N_epoch_print"] = 500
model_optimization_opt_list = [model_optimization_opt_dict] * num_gp
# Policy optimization options
policy_optimization_dict = {}
policy_optimization_dict["num_particles"] = 400
policy_optimization_dict["opt_steps_list"] = [2000, 4000, 4000, 4000, 4000]
policy_optimization_dict["lr_list"] = [0.01, 0.01, 0.01, 0.01, 0.01]
policy_optimization_dict["f_optimizer"] = "lambda p, lr : torch.optim.Adam(p, lr)"
policy_optimization_dict["num_step_print"] = 100
policy_optimization_dict["p_dropout_list"] = [0.25, 0.25, 0.25, 0.25, 0.25]
policy_optimization_dict["p_drop_reduction"] = 0.25 / 2
policy_optimization_dict["alpha_diff_cost"] = 0.99
policy_optimization_dict["min_diff_cost"] = 0.08
policy_optimization_dict["num_min_diff_cost"] = 200
policy_optimization_dict["min_step"] = 200
policy_optimization_dict["lr_min"] = 0.0025
policy_optimization_dict["policy_reinit_dict"] = policy_reinit_dict
# Options for method reinforce
reinforce_param_dict = {}
reinforce_param_dict["initial_state"] = np.array([0.0, 0.0, 0.0, 0.0])
reinforce_param_dict["initial_state_var"] = np.array([0.0001, 0.0001, 0.0001, 0.0001])
reinforce_param_dict["T_exploration"] = T_exploration
reinforce_param_dict["T_control"] = T_control
reinforce_param_dict["num_trials"] = num_trials
reinforce_param_dict["model_optimization_opt_list"] = model_optimization_opt_list
reinforce_param_dict["policy_optimization_dict"] = policy_optimization_dict

print("\n---- Save test configuration ----")
config_log_dict = {}
config_log_dict["MC_PILCO_init_dict"] = MC_PILCO_init_dict
config_log_dict["reinforce_param_dict"] = reinforce_param_dict
pkl.dump(config_log_dict, open("results_tmp/" + str(seed) + "/config_log.pkl", "wb"))

# Start the learning algorithm
PL_obj.reinforce(**reinforce_param_dict)
