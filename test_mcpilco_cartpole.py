# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL:	    Diego Romeres (romeres@merl.com)
"""

"""
Test MC-PILCO on a simulated cart-pole system (GPs equipped with square-exponential + polynomial kernels)
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
num_trials = 5  # Total trials
T_sampling = 0.05  # Sampling time
T_exploration = 3.0  # Duration of the first exploration trial
T_control = 3.0  # Duration of each of the following trials during learning
state_dim = 4  # State dimension
input_dim = 1  # Input dimension
num_gp = int(state_dim / 2)  # Number of Gaussian Processes to learn
gp_input_dim = 6  # Dimension of the input the Gaussian Process Regression
ode_fun = f_ode.cartpole  # Dynamic ODE of the simulated system
u_max = 10.0  # Input upperbound limit
std_noise = 10 ** (-2)  # Standard deviation of the measurement noise ...
std_list = std_noise * np.ones(state_dim)  # ... for all state dimensions
fl_SOD_GP = True  # Flag to select if to use or not a Subset of Data (SoD) approximation in the GPs
fl_reinforce_init_dist = (
    "Gaussian"  # Initial distribution of the particles in each of the trials. ['Gaussian','Uniform']
)

print("\n---- Set model learning parameters ----")
f_model_learning = ML.Speed_Model_learning_RBF_MPK_angle_state  # Model function to be trained
print(f_model_learning)
model_learning_par = {}
model_learning_par["num_gp"] = num_gp
model_learning_par["angle_indeces"] = [2]  # Indeces of the GP input components that are angles
model_learning_par["not_angle_indeces"] = [0, 1, 3]  # Indeces of the GP input components that are not angles
model_learning_par["T_sampling"] = T_sampling
model_learning_par["vel_indeces"] = [1, 3]  # Indeces of the GP input components that are velocities
model_learning_par["not_vel_indeces"] = [0, 2]  # Indeces of the GP input components that are not velocities
model_learning_par["device"] = device
model_learning_par["dtype"] = dtype
if fl_SOD_GP:
    model_learning_par["approximation_mode"] = "SOD"
    model_learning_par["approximation_dict"] = {
        "SOD_threshold_mode": "relative",
        "SOD_threshold": 0.5,
        "flg_SOD_permutation": False,
    }  # Set SoD threshold
# Kernel RBF initial parameters
init_dict_RBF = {}
init_dict_RBF["active_dims"] = np.arange(0, gp_input_dim)  # Select the GP input components considered inside the kernel
init_dict_RBF["lengthscales_init"] = np.ones(init_dict_RBF["active_dims"].size)  # Initial GP lenghtscales
init_dict_RBF["flg_train_lengthscales"] = True  # Set GP lengthscales trainable
init_dict_RBF["lambda_init"] = np.ones(1)  # Initial GP lambda
init_dict_RBF["flg_train_lambda"] = False  # Set GP lambda trainable
init_dict_RBF["sigma_n_init"] = 1 * np.ones(1)  # Initial GP noise std
init_dict_RBF["flg_train_sigma_n"] = True  # Set GP noise std trainable
init_dict_RBF["sigma_n_num"] = None  # Add fixed noise std for handling numerical issues (if necessary)
init_dict_RBF["dtype"] = dtype
init_dict_RBF["device"] = device
# Kernel MPK initial parameters
init_dict_MPK = {}
init_dict_MPK["active_dims"] = np.arange(0, gp_input_dim)  # Select the GP input components considered inside the kernel
init_dict_MPK["poly_deg"] = 2  # Degree of the polynomial kernel
# Initialize weights of the polynomial kernel
init_dict_MPK["Sigma_pos_par_init_list"] = [np.ones(gp_input_dim + 1)] + [
    np.ones((deg + 1) * (gp_input_dim)) for deg in range(1, init_dict_MPK["poly_deg"])
]
init_dict_MPK["flg_train_Sigma_pos_par_list"] = [True] * init_dict_MPK["poly_deg"]
init_dict_MPK["dtype"] = dtype
init_dict_MPK["device"] = device
# Prepare a list of kernels'parameters for each of the GPs (in this case all the GPs have the same parameters)
model_learning_par["init_dict_list"] = [[init_dict_RBF, init_dict_MPK]] * num_gp

print("\n---- Set exploration policy parameters ----")
# Set the exploration policy function
f_rand_exploration_policy = Policy.Random_exploration
# Set the parameters for the selected policy function
rand_exploration_policy_par = {}
rand_exploration_policy_par["state_dim"] = state_dim
rand_exploration_policy_par["input_dim"] = input_dim
rand_exploration_policy_par["u_max"] = u_max
rand_exploration_policy_par["dtype"] = dtype
rand_exploration_policy_par["device"] = device

print("\n---- Set control policy parameters ----")
num_basis = 200  # Number of Gaussian basis functions
# Set the control policy function
f_control_policy = Policy.Sum_of_gaussians_with_angles  # policy input: [p, p_dot, theta_dot, cos(theta), sin(theta)]
# Set the parameters for the selected policy function
control_policy_par = {}
control_policy_par["state_dim"] = state_dim
control_policy_par["input_dim"] = input_dim
control_policy_par["angle_indices"] = np.array([2])  # Index of theta inside state vector
control_policy_par["non_angle_indices"] = np.array([0, 1, 3])  # Indeces of [p, p_dot, theta_dot] inside state vector
control_policy_par["u_max"] = u_max
control_policy_par["num_basis"] = num_basis
control_policy_par["dtype"] = dtype
control_policy_par["device"] = device
angle_centers = np.pi * 2 * (np.random.rand(num_basis, 1) - 0.5)  # Sample 'num_basis' random angles ...
cos_centers = np.cos(angle_centers)  # ... to initialize cos(theta) centers and ...
sin_centers = np.sin(angle_centers)  # ... sin(theta) centers
not_angle_centers = np.pi * 2 * (np.random.rand(num_basis, 3) - 0.5)  # Initial centers for [p, p_dot, theta_dot]
control_policy_par["centers_init"] = np.concatenate(
    [not_angle_centers, cos_centers, sin_centers], 1
)  # Aggregate centers
control_policy_par["lengthscales_init"] = 1 * np.ones(state_dim + 1)  # one for each component of policy input
control_policy_par["weight_init"] = u_max * (np.random.rand(input_dim, num_basis) - 0.5)  # Initial random weights
control_policy_par["flg_squash"] = True  # Enable squashing input between [-u_max, u_max]
control_policy_par["flg_drop"] = True  # Enable dropout
policy_reinit_dict = {}  # Parameters for policy re-initialization (in case of NaN cost)
policy_reinit_dict["lenghtscales_par"] = control_policy_par["lengthscales_init"]
policy_reinit_dict["centers_par"] = np.array([np.pi, np.pi, np.pi, 1.0, 1.0])
policy_reinit_dict["weight_par"] = u_max


print("\n---- Set cost function ----")
# Set the cost function
f_cost_function = Cost_function.Cart_pole_cost
# Set the parameters for the selected cost function
cost_function_par = {}
cost_function_par["pos_index"] = 0  # Index of p in the state vector
cost_function_par["angle_index"] = 2  # Index of theta in the state vector
# Targets for theta and p, respectively
cost_function_par["target_state"] = torch.tensor([np.pi, 0.0], dtype=dtype, device=device)
# Cost lengthscales for theta and p , respectively
cost_function_par["lengthscales"] = torch.tensor([3.0, 1.0], dtype=dtype, device=device)

print("\n---- Init policy learning object ----")
MC_PILCO_init_dict = {}
MC_PILCO_init_dict["T_sampling"] = T_sampling
MC_PILCO_init_dict["state_dim"] = state_dim
MC_PILCO_init_dict["input_dim"] = input_dim
MC_PILCO_init_dict["f_sim"] = ode_fun
MC_PILCO_init_dict["std_meas_noise"] = np.array(std_list)
MC_PILCO_init_dict["f_model_learning"] = f_model_learning  # Model function to be trained
MC_PILCO_init_dict["model_learning_par"] = model_learning_par  # Model function parameters
MC_PILCO_init_dict["f_rand_exploration_policy"] = f_rand_exploration_policy  # Exploration policy function
MC_PILCO_init_dict[
    "rand_exploration_policy_par"
] = rand_exploration_policy_par  # Exploration policy function parameters
MC_PILCO_init_dict["f_control_policy"] = f_control_policy  # Control policy function
MC_PILCO_init_dict["control_policy_par"] = control_policy_par  # Control policy function parameters
MC_PILCO_init_dict["f_cost_function"] = f_cost_function  # Cost function
MC_PILCO_init_dict["cost_function_par"] = cost_function_par  # Cost function parameters
MC_PILCO_init_dict["log_path"] = "results_tmp/" + str(seed)  # path to save logs of the experiments
MC_PILCO_init_dict["dtype"] = dtype
MC_PILCO_init_dict["device"] = device
PL_obj = MC_PILCO.MC_PILCO(**MC_PILCO_init_dict)  # Main object of the algorithm with MC-PILCO properties

print("\n---- Set MC-PILCO options ----")
# Model optimization options
model_optimization_opt_dict = {}
model_optimization_opt_dict["f_optimizer"] = "lambda p : torch.optim.Adam(p, lr = 0.01)"  # Specify model optimizer
model_optimization_opt_dict["criterion"] = Likelihood.Marginal_log_likelihood  # Optimize marginal likelihood
model_optimization_opt_dict["N_epoch"] = 1501  # Max number of iterations to train the model
model_optimization_opt_dict["N_epoch_print"] = 500  # Frequency of printing to screen partial results
# Prepare a list for each of the GPs (in this case all the GPs have the same parameters)
model_optimization_opt_list = [model_optimization_opt_dict] * num_gp
# Policy optimization options
policy_optimization_dict = {}
policy_optimization_dict["num_particles"] = 400  # Number of simulated particles in the Monte-Carlo method
policy_optimization_dict["opt_steps_list"] = [
    2000,
    4000,
    4000,
    4000,
    4000,
]  # Max number of optimization steps for trial
policy_optimization_dict["lr_list"] = [0.01, 0.01, 0.01, 0.01, 0.01]  # Initial learning for trial
policy_optimization_dict["f_optimizer"] = "lambda p, lr : torch.optim.Adam(p, lr)"  # Specify policy optimizer
policy_optimization_dict["num_step_print"] = 100  # Frequency of printing to screen partial results
policy_optimization_dict["p_dropout_list"] = [0.25, 0.25, 0.25, 0.25, 0.25]  # Dropout initial probability for trial
policy_optimization_dict["p_drop_reduction"] = 0.25 / 2  # Dropout reduction parameter
policy_optimization_dict["alpha_diff_cost"] = 0.99  # Monitoring signal parameter α_s for early stopping criterion
policy_optimization_dict["min_diff_cost"] = 0.08  # Monitoring signal parameter σ_s for early stopping criterion
policy_optimization_dict["num_min_diff_cost"] = 200  # Monitoring signal parameter n_s for early stopping criterion
policy_optimization_dict["min_step"] = 200  # Number of initial steps without updating learning rate and dropout.
policy_optimization_dict["lr_min"] = 0.0025  # Minimum allowed learning rate
policy_optimization_dict["policy_reinit_dict"] = policy_reinit_dict  # Load policy re-init dict
# Options for method reinforce
reinforce_param_dict = {}
if fl_reinforce_init_dist == "Gaussian":  # Set initial state Gaussian distribution
    reinforce_param_dict["initial_state"] = np.array([0.0, 0.0, 0.0, 0.0])  # Mean
    reinforce_param_dict["initial_state_var"] = np.array([0.0001, 0.0001, 0.0001, 0.0001])  # Variance
elif fl_reinforce_init_dist == "Uniform":  # Set initial state uniform distribution
    reinforce_param_dict["flg_init_uniform"] = True
    reinforce_param_dict["init_up_bound"] = np.array([0.03, 0.03, 0.03, 0.03])  # Upper bound
    reinforce_param_dict["init_low_bound"] = -np.array([0.03, 0.03, 0.03, 0.03])  # Lower bound
reinforce_param_dict["T_exploration"] = T_exploration
reinforce_param_dict["T_control"] = T_control
reinforce_param_dict["num_trials"] = num_trials
reinforce_param_dict["model_optimization_opt_list"] = model_optimization_opt_list
reinforce_param_dict["policy_optimization_dict"] = policy_optimization_dict

print("\n---- Save test configuration ----")
config_log_dict = {}  # Save test settings
config_log_dict["MC_PILCO_init_dict"] = MC_PILCO_init_dict
config_log_dict["reinforce_param_dict"] = reinforce_param_dict
pkl.dump(config_log_dict, open("results_tmp/" + str(seed) + "/config_log.pkl", "wb"))

# Start the learning algorithm
PL_obj.reinforce(**reinforce_param_dict)  # Main method of the algorithm to start the learning process
