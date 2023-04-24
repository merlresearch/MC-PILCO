# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL:	    Diego Romeres (romeres@merl.com)
"""

"""
Plot obtained results from log files (cartpole experiments)
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

# file parameters
p = argparse.ArgumentParser("plot log")
p.add_argument("-dir_path", type=str, default="results_tmp/", help="none")
p.add_argument("-seed", type=int, default=1, help="none")

# load parameters
locals().update(vars(p.parse_known_args()[0]))
file_name = dir_path + str(seed) + "/log.pkl"
print("---- Reading log file: " + file_name)
log_dict = pkl.load(open(file_name, "rb"))
particles_states_list = log_dict["particles_states_list"]
particles_inputs_list = log_dict["particles_inputs_list"]
cost_trial_list = log_dict["cost_trial_list"]
input_samples_history = log_dict["input_samples_history"]
noiseless_states_history = log_dict["noiseless_states_history"]
num_trials = len(particles_states_list)

config_log_dict = pkl.load(open(dir_path + str(seed) + "/config_log.pkl", "rb"))
MC_PILCO_init_dict = config_log_dict["MC_PILCO_init_dict"]
f_cost_function = MC_PILCO_init_dict["f_cost_function"]
cost_function_par = MC_PILCO_init_dict["cost_function_par"]
cost_function = f_cost_function(**cost_function_par)
dtype = MC_PILCO_init_dict["dtype"]
device = MC_PILCO_init_dict["device"]


print("---- Save plots")
for trial_index in range(0, num_trials):
    state_samples = particles_states_list[trial_index]
    input_samples = particles_inputs_list[trial_index]

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.title("particles rollout trial: " + str(trial_index))
    plt.grid()
    plt.ylabel("$\\theta$")
    plt.plot(np.pi * np.ones(len(state_samples[:, :, 2])), "r--")
    plt.plot(-np.pi * np.ones(len(state_samples[:, :, 2])), "r--")
    plt.plot(state_samples[:, :, 2])
    plt.subplot(4, 1, 2)
    plt.grid()
    plt.ylabel("$x$")
    plt.plot(np.zeros(len(state_samples[:, :, 0])), "r--")
    plt.plot(state_samples[:, :, 0])
    plt.subplot(4, 1, 3)
    plt.grid()
    plt.ylabel("$u$")
    plt.plot(input_samples[:, :, 0])
    cost = (
        cost_function.cost_function(
            torch.tensor(state_samples, dtype=dtype, device=device),
            torch.tensor(input_samples, dtype=dtype, device=device),
            trial_index=trial_index,
        )
        .detach()
        .cpu()
        .numpy()
        .squeeze()
    )
    plt.subplot(4, 1, 4)
    plt.grid()
    plt.ylabel("$c$")
    plt.plot(cost)
    plt.plot(np.zeros(len(state_samples[:, :, 0])), "r--")
    plt.savefig(dir_path + str(seed) + "/" + "particles_rollout_trial" + str(trial_index) + ".pdf")
    plt.close()

trial_index_cost = [0] + list(range(num_trials))
for trial_index in range(0, num_trials + 1):
    state_samples = noiseless_states_history[trial_index]
    input_samples = input_samples_history[trial_index]

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.title("true rollout trial: " + str(trial_index))
    plt.grid()
    plt.ylabel("$\\theta$")
    plt.plot(np.pi * np.ones(len(state_samples[:, 2])), "r--")
    plt.plot(-np.pi * np.ones(len(state_samples[:, 2])), "r--")
    plt.plot(state_samples[:, 2])
    plt.subplot(4, 1, 2)
    plt.grid()
    plt.ylabel("$x$")
    plt.plot(np.zeros(len(state_samples[:, 0])), "r--")
    plt.plot(state_samples[:, 0])
    plt.subplot(4, 1, 3)
    plt.grid()
    plt.ylabel("$u$")
    plt.plot(input_samples)
    cost = (
        cost_function.cost_function(
            torch.tensor(state_samples, dtype=dtype, device=device).unsqueeze(1),
            torch.tensor(input_samples, dtype=dtype, device=device).unsqueeze(1),
            trial_index=trial_index_cost[trial_index],
        )
        .detach()
        .cpu()
        .numpy()
        .squeeze()
    )
    plt.subplot(4, 1, 4)
    plt.grid()
    plt.ylabel("$c$")
    plt.plot(cost)
    plt.plot(np.zeros(len(state_samples[:, 0])), "r--")
    plt.savefig(dir_path + str(seed) + "/" + "true_rollout_trial" + str(trial_index) + ".pdf")
    plt.close()

plt.figure()
plt.title("Learning plot")
start = 0
for trial_index in range(0, num_trials):
    cost_evolution = np.array(cost_trial_list[trial_index])
    ii = np.array(range(start, start + len(cost_evolution)))
    (h,) = plt.plot(ii, cost_evolution)
    start = start + len(cost_evolution)
plt.xlabel("optimization steps")
plt.ylabel("total rollout cost")
plt.grid()
plt.savefig(dir_path + str(seed) + "/" + "learning_plot.pdf")
plt.close()
