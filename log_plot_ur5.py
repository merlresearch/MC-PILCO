# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL:	    Diego Romeres (romeres@merl.com)
"""

"""
Plot obtained results from log files (ur5 MuJoCo experiment)
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
input_samples_history = log_dict["input_samples_history"]
noiseless_states_history = log_dict["noiseless_states_history"]
cost_trial_list = log_dict["cost_trial_list"]

config_log_dict = pkl.load(open(dir_path + str(seed) + "/config_log.pkl", "rb"))
# print(config_log_dict)
MC_PILCO_init_dict = config_log_dict["MC_PILCO_init_dict"]
reinforce_param_dict = config_log_dict["reinforce_param_dict"]
f_cost_function = MC_PILCO_init_dict["f_cost_function"]
cost_function_par = MC_PILCO_init_dict["cost_function_par"]
cost_function = f_cost_function(**cost_function_par)
dtype = MC_PILCO_init_dict["dtype"]
device = MC_PILCO_init_dict["device"]

num_trials = len(particles_states_list)

target_traj = MC_PILCO_init_dict["cost_function_par"]["target_traj"]

for trial_index in range(0, num_trials + 1):
    final_rollout = noiseless_states_history[trial_index]
    final_q = final_rollout[:, 0:6]
    np.savetxt(dir_path + str(seed) + "/q_trial_" + str(trial_index) + ".csv", final_q, delimiter=",")

T = reinforce_param_dict["T_control"]
dt = MC_PILCO_init_dict["T_sampling"]
times = np.linspace(0, T, int(T / dt))

print("---- Save plots")
for trial_index in range(0, num_trials):
    state_samples = particles_states_list[trial_index]
    input_samples = particles_inputs_list[trial_index]

    theta1, theta2, theta3, theta4, theta5, theta6 = (
        state_samples[:, :, :1],
        state_samples[:, :, 1:2],
        state_samples[:, :, 2:3],
        state_samples[:, :, 3:4],
        state_samples[:, :, 4:5],
        state_samples[:, :, 5:6],
    )

    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(times, theta1[:, :, 0], label="theta 1")
    plt.plot(times, target_traj[:, 0], "r--", label="target")
    plt.grid()
    plt.ylabel("$q_1$")
    plt.legend()
    plt.subplot(3, 2, 2)
    plt.plot(times, theta2[:, :, 0], label="theta 2")
    plt.plot(times, target_traj[:, 1], "r--", label="target")
    plt.grid()
    plt.ylabel("$q_2$")

    plt.subplot(3, 2, 3)
    plt.plot(times, theta3[:, :, 0], label="theta 3")
    plt.plot(times, target_traj[:, 2], "r--", label="target")
    plt.grid()
    plt.ylabel("$q_3$")

    plt.subplot(3, 2, 4)
    plt.plot(times, theta4[:, :, 0], label="theta 4")
    plt.plot(times, target_traj[:, 3], "r--", label="target")
    plt.grid()
    plt.ylabel("$q_4$")

    plt.subplot(3, 2, 5)
    plt.plot(times, theta5[:, :, 0], label="theta 5")
    plt.plot(times, target_traj[:, 4], "r--", label="target")
    plt.grid()
    plt.ylabel("$q_5$")

    plt.subplot(3, 2, 6)
    plt.plot(times, theta6[:, :, 0], label="theta 6")
    plt.plot(times, target_traj[:, 5], "r--", label="target")
    plt.grid()
    plt.ylabel("$q_6$")
    plt.savefig(dir_path + str(seed) + "/" + "particle_joints_trial" + str(trial_index) + ".pdf")
    plt.close()

    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(times, input_samples[:, :, 0], label="torque 1")
    plt.ylabel("$u_1$")
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(times, input_samples[:, :, 1], label="torque 2")
    plt.grid()
    plt.ylabel("$u_2$")

    plt.subplot(3, 2, 3)
    plt.plot(times, input_samples[:, :, 2], label="torque 3")
    plt.grid()
    plt.ylabel("$u_3$")

    plt.subplot(3, 2, 4)
    plt.plot(times, input_samples[:, :, 3], label="torque 4")
    plt.grid()
    plt.ylabel("$u_4$")

    plt.subplot(3, 2, 5)
    plt.plot(times, input_samples[:, :, 4], label="torque 5")
    plt.grid()
    plt.ylabel("$u_5$")

    plt.subplot(3, 2, 6)
    plt.plot(times, input_samples[:, :, 5], label="torque 6")
    plt.grid()
    plt.ylabel("$u_6$")
    plt.savefig(dir_path + str(seed) + "/" + "particles_torques_trial" + str(trial_index) + ".pdf")
    plt.close()


trial_index_cost = [0] + list(range(num_trials))
for trial_index in range(0, num_trials + 1):
    state_samples = noiseless_states_history[trial_index]
    input_samples = input_samples_history[trial_index]

    theta1, theta2, theta3, theta4, theta5, theta6 = (
        state_samples[:, :1],
        state_samples[:, 1:2],
        state_samples[:, 2:3],
        state_samples[:, 3:4],
        state_samples[:, 4:5],
        state_samples[:, 5:6],
    )
    dot_theta1, dot_theta2, dot_theta3, dot_theta4, dot_theta5, dot_theta6 = (
        state_samples[:, 6:7],
        state_samples[:, 7:8],
        state_samples[:, 8:9],
        state_samples[:, 9:10],
        state_samples[:, 10:11],
        state_samples[:, 11:12],
    )

    # cost function
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
    plt.figure()
    plt.grid()
    plt.ylabel("$c$")
    plt.plot(times, cost)
    plt.plot(times, np.zeros(len(state_samples[:, 0])), "r--")
    plt.savefig(dir_path + str(seed) + "/" + "true_cost_trial" + str(trial_index) + ".pdf")
    plt.close()

    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(times, theta1[:, 0], label="theta 1")
    plt.plot(times, target_traj[:, 0], "r--", label="target")
    plt.ylabel("$q_1$ [rad]")
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(times, theta2[:, 0], label="theta 2")
    plt.plot(times, target_traj[:, 1], "r--", label="target")
    plt.grid()
    plt.ylabel("$q_2$ [rad]")

    plt.subplot(3, 2, 3)
    plt.plot(times, theta3[:, 0], label="theta 3")
    plt.plot(times, target_traj[:, 2], "r--", label="target")
    plt.grid()
    plt.ylabel("$q_3$ [rad]")

    plt.subplot(3, 2, 4)
    plt.plot(times, theta4[:, 0], label="theta 4")
    plt.plot(times, target_traj[:, 3], "r--", label="target")
    plt.grid()
    plt.ylabel("$q_4$ [rad]")

    plt.subplot(3, 2, 5)
    plt.plot(times, theta5[:, 0], label="theta 5")
    plt.plot(times, target_traj[:, 4], "r--", label="target")
    plt.grid()
    plt.ylabel("$q_5$ [rad]")

    plt.subplot(3, 2, 6)
    plt.plot(times, theta6[:, 0], label="theta 6")
    plt.plot(times, target_traj[:, 5], "r--", label="target")
    plt.grid()
    plt.ylabel("$q_6$ [rad]")
    plt.savefig(dir_path + str(seed) + "/" + "true_joints_trial" + str(trial_index) + ".pdf")
    plt.close()

    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(times, 180 / np.pi * np.abs(theta1[:, 0] - target_traj[:, 0].numpy()), label="error joint 1")
    plt.ylabel("$q_1$ [deg]")
    plt.grid()
    plt.legend()
    plt.subplot(3, 2, 2)
    plt.plot(times, 180 / np.pi * np.abs(theta2[:, 0] - target_traj[:, 1].numpy()), label="error joint 2")
    plt.grid()
    plt.ylabel("$q_2$ [deg]")

    plt.subplot(3, 2, 3)
    plt.plot(times, 180 / np.pi * np.abs(theta3[:, 0] - target_traj[:, 2].numpy()), label="error joint 3")
    plt.grid()
    plt.ylabel("$q_3$ [deg]")

    plt.subplot(3, 2, 4)
    plt.plot(times, 180 / np.pi * np.abs(theta4[:, 0] - target_traj[:, 3].numpy()), label="error joint 4")
    plt.grid()
    plt.ylabel("$q_4$ [deg]")

    plt.subplot(3, 2, 5)
    plt.plot(times, 180 / np.pi * np.abs(theta5[:, 0] - target_traj[:, 4].numpy()), label="error joint 5")
    plt.grid()
    plt.ylabel("$q_5$ [deg]")

    plt.subplot(3, 2, 6)
    plt.plot(times, 180 / np.pi * np.abs(theta6[:, 0] - target_traj[:, 5].numpy()), label="error joint 6")
    plt.grid()
    plt.ylabel("$q_6$ [deg]")
    plt.savefig(dir_path + str(seed) + "/" + "true_joints_error_trial" + str(trial_index) + ".pdf")
    plt.close()

    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(dot_theta1[:, 0], label="theta 1")
    plt.plot(target_traj[:, 6], "r--", label="target")
    plt.ylabel("$\dot{q}_1$")
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(dot_theta2[:, 0], label="theta 2")
    plt.plot(target_traj[:, 7], "r--", label="target")
    plt.grid()
    plt.ylabel("$\dot{q}_2$")

    plt.subplot(3, 2, 3)
    plt.plot(dot_theta3[:, 0], label="theta 3")
    plt.plot(target_traj[:, 8], "r--", label="target")
    plt.grid()
    plt.ylabel("$\dot{q}_3$")

    plt.subplot(3, 2, 4)
    plt.plot(dot_theta4[:, 0], label="theta 4")
    plt.plot(target_traj[:, 9], "r--", label="target")
    plt.grid()
    plt.ylabel("$\dot{q}_4$")

    plt.subplot(3, 2, 5)
    plt.plot(dot_theta5[:, 0], label="theta 5")
    plt.plot(target_traj[:, 10], "r--", label="target")
    plt.grid()
    plt.ylabel("$\dot{q}_5$")

    plt.subplot(3, 2, 6)
    plt.plot(dot_theta6[:, 0], label="theta 6")
    plt.plot(target_traj[:, 11], "r--", label="target")
    plt.grid()
    plt.ylabel("$\dot{q}_6$")
    plt.savefig(dir_path + str(seed) + "/" + "true_speeds_trial" + str(trial_index) + ".pdf")
    plt.close()

    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(input_samples[:, 0], label="torque 1")
    plt.ylabel("$u_1$")
    plt.grid()
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(input_samples[:, 1], label="torque 2")
    plt.grid()
    plt.ylabel("$u_2$")

    plt.subplot(3, 2, 3)
    plt.plot(input_samples[:, 2], label="torque 3")
    plt.grid()
    plt.ylabel("$u_3$")

    plt.subplot(3, 2, 4)
    plt.plot(input_samples[:, 3], label="torque 4")
    plt.grid()
    plt.ylabel("$u_4$")

    plt.subplot(3, 2, 5)
    plt.plot(input_samples[:, 4], label="torque 5")
    plt.grid()
    plt.ylabel("$u_5$")

    plt.subplot(3, 2, 6)
    plt.plot(input_samples[:, 5], label="torque 6")
    plt.grid()
    plt.ylabel("$u_6$")
    plt.savefig(dir_path + str(seed) + "/" + "true_torques_trial" + str(trial_index) + ".pdf")
    plt.close()


plt.figure()
plt.title("Learning plot")
start = 0
for trial_index in range(0, num_trials):
    cost_evolution = np.array(cost_trial_list[trial_index])
    steps = np.array(range(start, start + len(cost_evolution)))
    plt.plot(steps, cost_evolution)
    start = start + len(cost_evolution)

plt.xlabel("optimization steps")
plt.ylabel("total rollout cost")
plt.yscale("log")
plt.grid()
plt.savefig(dir_path + str(seed) + "/" + "learning_plot.pdf")
plt.close()
