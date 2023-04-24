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

import simulation_class.model as model


class MC_PILCO(torch.nn.Module):
    """
    Monte-Carlo Probabilistic Inference for Learning COntrol
    """

    def __init__(
        self,
        T_sampling,
        state_dim,
        input_dim,
        f_sim,
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
        super(MC_PILCO, self).__init__()
        # model parameters
        self.T_sampling = T_sampling  # sampling time
        self.dtype = dtype  # data type
        self.device = device  # device
        self.state_dim = state_dim  # state dimension
        self.input_dim = input_dim  # input dimension
        # get the simulated system
        print("\n\nGet the system...")
        self.system = model.Model(f_sim)  # ODE-simulated system
        if std_meas_noise is None:
            std_meas_noise = np.zeros(state_dim)
        self.std_meas_noise = std_meas_noise  # measurement noise
        # get the model learning object
        print("\n\nGet the learning object...")
        self.model_learning = f_model_learning(**model_learning_par)
        # get the get the random explorataion policy object
        print("\n\nGet the exploration policy...")
        self.rand_exploration_policy = f_rand_exploration_policy(**rand_exploration_policy_par)
        # get the initialize the policy parameters and get the policy object
        print("\n\nGet the control policy...")
        self.control_policy = f_control_policy(**control_policy_par)
        # set the cost function
        print("\n\nGet the cost function...")
        self.cost_function = f_cost_function(**cost_function_par)
        # state and input samples hystory list
        self.state_samples_history = []
        self.input_samples_history = []
        self.noiseless_states_history = []
        # initialize num_data_collection
        self.num_data_collection = 0
        # create log file dictionary
        self.log_path = log_path
        if self.log_path is not None:
            self.log_dict = {}

    def reinforce(
        self,
        initial_state,
        initial_state_var,
        T_exploration,
        T_control,
        num_trials,
        model_optimization_opt_list,
        policy_optimization_dict,
        num_explorations=1,
        flg_init_uniform=False,
        init_up_bound=None,
        init_low_bound=None,
        flg_init_multi_gauss=False,
        random_initial_state=True,
        loaded_model=False,
    ):
        """
        Model learning + policy learning method
        """
        # get initial data
        if not loaded_model:
            print("\n\n\n\n----------------- INITIAL EXPLORATIONS -----------------")
            # perform 'num_explorations' interactions with the system to learn initial model
            for expl_index in range(0, num_explorations):
                print("\nEXPLORATION # " + str(expl_index))
                if random_initial_state == True:  # initial state randomly sampled
                    if flg_init_uniform == True:  # uniform initial distribution
                        x0 = np.random.uniform(init_low_bound, init_up_bound)
                    elif flg_init_multi_gauss == True:  # multimodal gaussians initial distribution
                        num_init = np.random.randint(initial_state.shape[0])
                        x0 = np.random.normal(initial_state[num_init, :], np.sqrt(initial_state_var[num_init, :]))
                    else:  # gaussian initial distribution
                        x0 = np.random.normal(initial_state, np.sqrt(initial_state_var))
                else:  # deterministic initial state
                    x0 = initial_state

                # interact with the system
                self.get_data_from_system(
                    initial_state=x0,
                    T_exploration=T_exploration,
                    flg_exploration=True,  # exploration interaction
                    trial_index=expl_index,
                )
            cost_trial_list = []
            std_cost_trial_list = []
            parameters_trial_list = []
            particles_states_list = []
            particles_inputs_list = []

            first_trial_index = num_explorations - 1
            last_trial_index = num_trials + num_explorations - 1

        else:
            cost_trial_list = self.log_dict["cost_trial_list"]
            std_cost_trial_list = self.log_dict["std_cost_trial_list"]
            parameters_trial_list = self.log_dict["parameters_trial_list"]
            particles_states_list = self.log_dict["particles_states_list"]
            particles_inputs_list = self.log_dict["particles_inputs_list"]

            num_past_trials = len(self.state_samples_history)
            first_trial_index = num_past_trials - 1
            last_trial_index = num_trials + num_past_trials - 1

        # reinforce the model and the policy
        for trial_index in range(first_trial_index, last_trial_index):
            print("\n\n\n\n----------------- TRIAL " + str(trial_index) + " -----------------")
            # train GPs on observed interaction data
            print("\n\n----- REINFORCE THE MODEL -----")
            self.model_learning.reinforce_model(optimization_opt_list=model_optimization_opt_list)

            with torch.no_grad():
                if self.log_path is not None:
                    print("Save log file...")
                    self.log_dict["parameters_gp_" + str(trial_index)] = [
                        copy.deepcopy(self.model_learning.gp_list[k].state_dict())
                        for k in range(0, self.model_learning.num_gp)
                    ]
                    self.log_dict["gp_inputs_" + str(trial_index)] = self.model_learning.gp_inputs
                    self.log_dict["gp_output_list_" + str(trial_index)] = self.model_learning.gp_output_list
                    self.log_dict["state_samples_history"] = self.state_samples_history
                    self.log_dict["input_samples_history"] = self.input_samples_history
                    self.log_dict["noiseless_states_history"] = self.noiseless_states_history
                    pkl.dump(self.log_dict, open(self.log_path + "/log.pkl", "wb"))

                # get model learning performance
                print("\n\n----- CHECK THE LEARNING PERFORMANCE (after model update) -----")
                _, _, _, _ = self.get_model_learning_performance(data_collection_index=trial_index)
                # get policy performance
                print("\n\n----- CHECK THE ROLLOUT PERFORMANCE (after model update) -----")
                _, _, _ = self.get_rollout_prediction_performance(data_collection_index=trial_index, add_name="post_tr")

                if flg_init_uniform == True:
                    particles_init_up_bound = torch.tensor(init_up_bound, dtype=self.dtype, device=self.device)
                    particles_init_low_bound = torch.tensor(init_low_bound, dtype=self.dtype, device=self.device)
                else:
                    particles_init_up_bound = None
                    particles_init_low_bound = None

                particles_initial_state_mean = torch.tensor(initial_state, dtype=self.dtype, device=self.device)
                particles_initial_state_var = torch.tensor(initial_state_var, dtype=self.dtype, device=self.device)

            print("\n\n----- REINFORCE THE POLICY -----")
            self.model_learning.set_eval_mode()

            # update the policy based on particle-simulation with the learned model
            cost_list, std_cost_list, particles_states, particles_inputs = self.reinforce_policy(
                T_control=T_control,
                particles_initial_state_mean=particles_initial_state_mean,
                particles_initial_state_var=particles_initial_state_var,
                flg_particles_init_uniform=flg_init_uniform,
                particles_init_up_bound=particles_init_up_bound,
                particles_init_low_bound=particles_init_low_bound,
                flg_particles_init_multi_gauss=flg_init_multi_gauss,
                trial_index=trial_index,
                **policy_optimization_dict
            )

            # save cost components
            cost_trial_list.append(cost_list)
            std_cost_trial_list.append(std_cost_list)
            particles_states_list.append(particles_states)
            particles_inputs_list.append(particles_inputs)
            parameters_trial_list.append(copy.deepcopy(self.control_policy.state_dict()))

            if self.log_path is not None:
                print("Save log file...")
                self.log_dict["cost_trial_list"] = cost_trial_list
                self.log_dict["std_cost_trial_list"] = std_cost_trial_list
                self.log_dict["parameters_trial_list"] = parameters_trial_list
                self.log_dict["particles_states_list"] = particles_states_list
                self.log_dict["particles_inputs_list"] = particles_inputs_list
                pkl.dump(self.log_dict, open(self.log_path + "/log.pkl", "wb"))
            self.model_learning.set_training_mode()

            # test policy
            if random_initial_state == True:
                if flg_init_uniform == True:
                    x0 = np.random.uniform(init_low_bound, init_up_bound)
                elif flg_init_multi_gauss == True:
                    num_init = np.random.randint(initial_state.shape[0])
                    x0 = np.random.normal(initial_state[num_init, :], np.sqrt(initial_state_var[num_init, :]))
                else:
                    x0 = np.random.normal(initial_state, np.sqrt(initial_state_var))
            else:
                x0 = initial_state

            print("\n\n----- APPLY THE CONTROL POLICY -----")
            # interact with the system
            self.get_data_from_system(
                initial_state=x0,
                T_exploration=T_control,
                flg_exploration=False,  # control policy interaction
                trial_index=trial_index + 1,
            )

            if self.log_path is not None:
                print("Save log file...")
                self.log_dict["state_samples_history"] = self.state_samples_history
                self.log_dict["input_samples_history"] = self.input_samples_history
                self.log_dict["noiseless_states_history"] = self.noiseless_states_history
                pkl.dump(self.log_dict, open(self.log_path + "/log.pkl", "wb"))

            print("\n\n----- CHECK THE MODEL LEARNING PERFORMANCE (before model update) -----")
            _, _, _, _ = self.get_model_learning_performance(data_collection_index=trial_index + 1)

            print("\n\n----- CHECK THE ROLLOUT PERFORMANCE (before model update) -----")
            _, _, _ = self.get_rollout_prediction_performance(data_collection_index=trial_index + 1, add_name="pre_tr")

        return cost_trial_list, particles_states_list, particles_inputs_list

    def get_model_learning_performance(self, data_collection_index, flg_pretrain=False):
        """
        Test model learning performance in the data_collection_index simulation
        """
        # get gp predictions
        (
            gp_inputs,
            gp_outputs_target_list,
            gp_output_mean_list,
            gp_output_var_list,
        ) = self.model_learning.get_gp_estimate_from_data(
            states=torch.tensor(
                self.state_samples_history[data_collection_index], dtype=self.dtype, device=self.device
            ),
            inputs=torch.tensor(
                self.input_samples_history[data_collection_index], dtype=self.dtype, device=self.device
            ),
            flg_pretrain=flg_pretrain,
        )
        for i in range(self.model_learning.num_gp):
            gp_output_var_list[i] = gp_output_var_list[i] * self.model_learning.norm_list[i] ** 2

        # move data to numpy
        gp_outputs_target_list = [
            gp_outputs_target_list[i].detach().cpu().numpy() for i in range(0, self.model_learning.num_gp)
        ]
        gp_output_mean_list = [
            gp_output_mean_list[i].detach().cpu().numpy() for i in range(0, self.model_learning.num_gp)
        ]
        # get gp performance
        for gp_index in range(0, self.model_learning.num_gp):

            print(
                "MSE gp" + str(gp_index) + ": ",
                ((gp_outputs_target_list[gp_index] - gp_output_mean_list[gp_index]) ** 2).mean(),
            )
        #     # uncomment to plot model learning performance
        #     plt.figure()
        #     plt.plot(gp_outputs_target_list[gp_index], label = 'y '+str(gp_index))
        #     plt.plot(gp_output_mean_list[gp_index], label = 'y '+str(gp_index)+' hat')
        #     plt.grid()
        #     plt.legend()
        #     # plt.savefig('results_tmp/'+'gp'+str(gp_index)+'_trial'+str(data_collection_index)+'.pdf')
        #     # plt.close()
        # plt.show()

        return gp_inputs, gp_outputs_target_list, gp_output_mean_list, gp_output_var_list

    def get_rollout_prediction_performance(
        self, data_collection_index, T_rollout=None, add_name="", particle_pred=False
    ):
        """
        Test rollout prediction
        """
        # simulate rollout with inputs from 'data_collection_index' trial
        rollout_states = self.rollout(
            data_collection_index=data_collection_index, T_rollout=T_rollout, particle_pred=particle_pred
        )
        # get rollout performance
        for state_dim_index in range(self.state_dim):
            print(
                "MSE Rollout dim" + str(state_dim_index) + ": ",
                (
                    (
                        self.state_samples_history[data_collection_index][:, state_dim_index]
                        - rollout_states[:, state_dim_index]
                    )
                    ** 2
                ).mean(),
            )
        #     # uncomment to plot rollout performance
        #     plt.figure()
        #     plt.plot(rollout_states[:,state_dim_index],'r', label = 'predicted rollout dim '+str(state_dim_index))
        #     plt.plot(self.state_samples_history[data_collection_index][:,state_dim_index], label = 'true rollout '+str(state_dim_index))
        #     plt.grid()
        #     plt.legend()
        #     plt.grid()
        #     # plt.savefig('results_tmp/'+'dim'+str(state_dim_index)+'_trial'+str(data_collection_index)+'_'+add_name+'.pdf')
        #     # plt.close()
        # plt.show()

        return (
            rollout_states,
            self.state_samples_history[data_collection_index],
            self.input_samples_history[data_collection_index],
        )

    def rollout(self, data_collection_index, T_rollout=None, particle_pred=False):
        """
        Performs rollout of the data_collection_index trajectory
        """
        # check T_rollout
        if T_rollout is None:
            T_rollout = self.state_samples_history[data_collection_index].shape[0]
        # get initial state
        current_state_tc = torch.tensor(
            self.state_samples_history[data_collection_index][0:1, :], dtype=self.dtype, device=self.device
        )
        # input trajectory as tensor
        inputs_trajectory_tc = torch.tensor(
            self.input_samples_history[data_collection_index], dtype=self.dtype, device=self.device
        )
        # allocate the space for the rollout trajectory
        rollout_trj = torch.zeros([T_rollout, self.state_dim], dtype=self.dtype, device=self.device)
        rollout_trj[0:1, :] = current_state_tc
        # simulate system evolution for 'T_rollout' steps
        for t in range(1, T_rollout):
            # get next state
            rollout_trj[t : t + 1, :], _, _ = self.model_learning.get_next_state(
                current_state=rollout_trj[t - 1 : t, :],
                current_input=inputs_trajectory_tc[t - 1 : t, :],
                particle_pred=particle_pred,
            )
        return rollout_trj.detach().cpu().numpy()

    def reinforce_policy(
        self,
        T_control,
        num_particles,
        trial_index,
        particles_initial_state_mean,
        particles_initial_state_var,
        flg_particles_init_uniform,
        particles_init_up_bound,
        particles_init_low_bound,
        flg_particles_init_multi_gauss,
        opt_steps_list,
        lr_list,
        f_optimizer,
        num_step_print=10,
        policy_reinit_dict=None,
        p_dropout_list=None,
        std_cost_filt_order=None,
        std_cost_filt_cutoff=None,
        max_std_cost=None,
        alpha_cost=0.99,
        alpha_input=0.99,
        alpha_diff_cost=0.99,
        lr_reduction_ratio=0.5,
        lr_min=0.001,
        p_drop_reduction=0.0,
        min_diff_cost=0.1,
        num_min_diff_cost=200,
        min_step=np.inf,
    ):

        """
        Improve the policy parameters with MC optimization
        """

        # init cost variables
        control_horizon = int(T_control / self.T_sampling)
        num_opt_steps = opt_steps_list[trial_index]
        cost_list = torch.zeros(num_opt_steps, device=self.device, dtype=self.dtype)
        std_cost_list = torch.zeros(num_opt_steps, device=self.device, dtype=self.dtype)
        previous_cost = 0.0
        current_min_step = min_step
        reinit_counter = 0

        # check dropout parameters
        if p_dropout_list is None:
            p_dropout = 0.0
        else:
            p_dropout = p_dropout_list[trial_index]
            print("\nDROPOUT ACTIVE:")
            print("p_dropout:", p_dropout)
        p_dropout_applied = p_dropout
        flg_drop = False

        # initilize the SE filter for monitoring cost improvement
        with torch.no_grad():
            num_attempts = 0
            flg_nan = True
            # repeat 'apply_policy' if nan is obtained
            while num_attempts < 10 and flg_nan:
                # apply the policy in simulation
                states_sequence_NODROP, inputs_sequence_NODROP = self.apply_policy(
                    particles_initial_state_mean=particles_initial_state_mean,
                    particles_initial_state_var=particles_initial_state_var,
                    flg_particles_init_uniform=flg_particles_init_uniform,
                    flg_particles_init_multi_gauss=flg_particles_init_multi_gauss,
                    particles_init_up_bound=particles_init_up_bound,
                    particles_init_low_bound=particles_init_low_bound,
                    num_particles=num_particles,
                    T_control=control_horizon,
                    p_dropout=p_dropout_applied,
                )
                # initial cost with no dropout applied
                cost_NODROP, std_cost_NODROP = self.cost_function(
                    states_sequence_NODROP, inputs_sequence_NODROP, trial_index
                )
                if torch.isnan(cost_NODROP):
                    num_attempts += 1
                    print("\nSE filter initialization: Cost is NaN - reinit the policy")
                    self.control_policy.reinit(**policy_reinit_dict)
                else:
                    flg_nan = False

        # initilize filters
        ES1_diff_cost = torch.zeros(num_opt_steps + 1, device=self.device, dtype=self.dtype)
        ES2_diff_cost = 0.0
        diff_cost_ratio = torch.zeros(num_opt_steps + 1, device=self.device, dtype=self.dtype)
        cost_tm1 = cost_NODROP
        current_min_diff_cost = min_diff_cost

        # get the optimizer
        lr = lr_list[trial_index]  # list of learning rates for trial
        f_optim = eval(f_optimizer)
        optimizer = f_optim(p=self.control_policy.parameters(), lr=lr)

        # optimize the policy
        opt_step = 0
        opt_step_done = 0
        t_start = time.time()
        # max optimization steps = num_opt_steps
        while opt_step < num_opt_steps:
            # set the gradient to zero
            optimizer.zero_grad()

            num_attempts = 0
            flg_nan = True
            # repeat 'apply_policy' if nan is obtained
            while num_attempts < 10 and flg_nan:
                # apply the policy
                states_sequence, inputs_sequence = self.apply_policy(
                    particles_initial_state_mean=particles_initial_state_mean,
                    particles_initial_state_var=particles_initial_state_var,
                    flg_particles_init_uniform=flg_particles_init_uniform,
                    flg_particles_init_multi_gauss=flg_particles_init_multi_gauss,
                    particles_init_up_bound=particles_init_up_bound,
                    particles_init_low_bound=particles_init_low_bound,
                    num_particles=num_particles,
                    T_control=control_horizon,
                    p_dropout=p_dropout_applied,
                )
                # compute the expected cost
                cost, std_cost = self.cost_function(states_sequence, inputs_sequence, trial_index)
                if torch.isnan(cost):
                    num_attempts += 1
                    print("\nCost is NaN: try sampling again")
                else:
                    flg_nan = False

            # save current step's cost
            cost_list[opt_step] = cost.data.clone().detach()
            std_cost_list[opt_step] = std_cost.data.clone().detach()

            # update filters
            with torch.no_grad():
                # compute the mean of the diff cost
                ES1_diff_cost[opt_step + 1] = alpha_diff_cost * ES1_diff_cost[opt_step] + (1 - alpha_diff_cost) * (
                    cost - cost_tm1
                )
                ES2_diff_cost = alpha_diff_cost * (
                    ES2_diff_cost + (1 - alpha_diff_cost) * ((cost - cost_tm1 - ES1_diff_cost[opt_step]) ** 2)
                )
                cost_tm1 = cost_list[opt_step]
                diff_cost_ratio[opt_step + 1] = alpha_diff_cost * diff_cost_ratio[opt_step] + (1 - alpha_diff_cost) * (
                    ES1_diff_cost[opt_step + 1] / (ES2_diff_cost.sqrt())
                )

            # compute the gradient and optimize the policy
            cost.backward(retain_graph=False)

            # updata parameters
            optimizer.step()

            # check improvement
            if opt_step % num_step_print == 0:
                t_stop = time.time()
                improvement = previous_cost - cost.data.cpu().numpy()
                previous_cost = cost.data.cpu().numpy()
                print("\nOptimization step: ", opt_step)
                print("cost: ", previous_cost)
                print("cost improvement: ", improvement)
                print("p_dropout_applied: ", p_dropout_applied)
                print("current_min_diff_cost; ", current_min_diff_cost)
                print("current_min_step: ", current_min_step)
                print("diff_cost_ratio: ", torch.abs(diff_cost_ratio[opt_step + 1]).cpu().numpy())
                print("time elapsed: ", t_stop - t_start)
                t_start = time.time()

            # check learning rate and exit conditions
            if opt_step > current_min_step:
                if (
                    torch.sum(
                        torch.abs(diff_cost_ratio[opt_step + 1 - num_min_diff_cost : opt_step + 1])
                        < current_min_diff_cost
                    )
                    >= num_min_diff_cost
                ):
                    if lr > lr_min:
                        print("Optimization_step:", opt_step)
                        print("\nREDUCING THE LEARNING RATE:")
                        lr = max(lr * lr_reduction_ratio, lr_min)
                        print("lr: ", lr)
                        current_min_diff_cost = max(current_min_diff_cost / 2, 0.01)
                        current_min_step = opt_step + num_min_diff_cost
                        optimizer = f_optim(p=self.control_policy.parameters(), lr=lr)
                        print("\nREDUCING THE DROPOUT:")
                        p_dropout_applied = p_dropout_applied - p_drop_reduction
                        if p_dropout_applied < 0:
                            p_dropout_applied = 0.0
                        print("p_dropout_applied: ", p_dropout_applied)
                    else:
                        print("\nEXIT FROM OPTIMIZATION: diff_cost_ratio < min_diff_cost for num_min_diff_cost steps")
                        opt_step = num_opt_steps

            # increase step counter
            opt_step = opt_step + 1
            opt_step_done = opt_step_done + 1

            # reinit policy if NaN appeared
            if flg_nan:
                reinit_counter = reinit_counter + 1
                error_message = "Cost is NaN:"
                print("\n" + error_message + " re-initialize control policy [attempt #" + str(reinit_counter) + "]")
                self.control_policy.reinit(**policy_reinit_dict)
                # reset counter to 0
                opt_step = 0
                opt_step_done = 0
                current_min_step = min_step
                previous_cost = 0.0
                # re-init cost variables
                cost_list = torch.zeros(num_opt_steps, device=self.device, dtype=self.dtype)
                std_cost_list = torch.zeros(num_opt_steps, device=self.device, dtype=self.dtype)
                gradients_list = []
                states_sequence_std_list = torch.zeros(
                    [self.state_dim, num_opt_steps], device=self.device, dtype=self.dtype
                )
                cost_list_NODROP = torch.zeros(num_opt_steps, device=self.device, dtype=self.dtype)
                std_cost_list_NODROP = torch.zeros(num_opt_steps, device=self.device, dtype=self.dtype)
                mean_states_list = torch.zeros(
                    [self.state_dim, num_opt_steps, control_horizon], device=self.device, dtype=self.dtype
                )
                mean_inputs_list = torch.zeros(
                    [self.input_dim, num_opt_steps, control_horizon], device=self.device, dtype=self.dtype
                )
                drop_list = []
                ES1_diff_cost = torch.zeros(num_opt_steps + 1, device=self.device, dtype=self.dtype)
                diff_cost_ratio = torch.zeros(num_opt_steps + 1, device=self.device, dtype=self.dtype)
                current_min_diff_cost = min_diff_cost
                # re-init the optimizer
                lr = lr_list[trial_index]
                f_optim = eval(f_optimizer)
                optimizer = f_optim(p=self.control_policy.parameters(), lr=lr)
                # Reinit dropout
                p_dropout_applied = p_dropout

        # move log variables to numpy
        cost_list = cost_list[0:opt_step_done].detach().cpu().numpy()
        std_cost_list = std_cost_list[0:opt_step_done].detach().cpu().numpy()

        return cost_list, std_cost_list, states_sequence.detach().cpu().numpy(), inputs_sequence.detach().cpu().numpy()

    def apply_policy(
        self,
        particles_initial_state_mean,
        particles_initial_state_var,
        flg_particles_init_uniform,
        particles_init_up_bound,
        particles_init_low_bound,
        flg_particles_init_multi_gauss,
        num_particles,
        T_control,
        p_dropout=0.0,
    ):
        """
        Apply the policy in simulation to a batch of particles:
        """
        # initialize variables
        states_sequence_list = []
        inputs_sequence_list = []

        # get initial particles
        if flg_particles_init_uniform == True:
            # initial uniform distribution
            uniform_ub_particles = particles_init_up_bound.repeat(num_particles, 1)
            uniform_lb_particles = particles_init_low_bound.repeat(num_particles, 1)
            state_distribution = Uniform(uniform_lb_particles, uniform_ub_particles)
        elif flg_particles_init_multi_gauss == True:
            # initial multimodal Gaussian distribution
            indices = torch.randint(0, particles_initial_state_mean.shape[0], [num_particles])
            initial_particles_mean = particles_initial_state_mean[indices, :]
            initial_particles_cov_mat = torch.stack([torch.diag(particles_initial_state_var[i, :]) for i in indices])
            state_distribution = MultivariateNormal(
                loc=initial_particles_mean, covariance_matrix=initial_particles_cov_mat
            )
        else:
            # initial Gaussian distribution
            initial_particles_mean = particles_initial_state_mean.repeat(num_particles, 1)
            initial_particles_cov_mat = torch.stack([torch.diag(particles_initial_state_var)] * num_particles)
            state_distribution = MultivariateNormal(
                loc=initial_particles_mean, covariance_matrix=initial_particles_cov_mat
            )

        # sample particles at t=0 from initial state distribution
        states_sequence_list.append(state_distribution.rsample())

        # compute initial inputs
        inputs_sequence_list.append(self.control_policy(states_sequence_list[0], t=0, p_dropout=p_dropout))

        for t in range(1, int(T_control)):

            # get next state mean and variance (given the states sampled and the inputs computed)
            particles, _, _ = self.model_learning.get_next_state(
                current_state=states_sequence_list[t - 1], current_input=inputs_sequence_list[t - 1]
            )
            states_sequence_list.append(particles)

            # compute next input
            inputs_sequence_list.append(self.control_policy(states_sequence_list[t], t=t, p_dropout=p_dropout))

        # returns states/inputs trajectories
        return torch.stack(states_sequence_list), torch.stack(inputs_sequence_list)

    def get_data_from_system(self, initial_state, T_exploration, trial_index, flg_exploration=False):
        """
        Apply exploration/control policy to the system and collect interaction data
        """
        # select the policy
        if flg_exploration:
            current_policy = self.rand_exploration_policy
        else:
            current_policy = self.control_policy

        # method for interacting with ODE-simulated system
        state_samples, input_samples, noiseless_samples = self.system.rollout(
            s0=initial_state,
            policy=current_policy.get_np_policy(),
            T=T_exploration,
            dt=self.T_sampling,
            noise=self.std_meas_noise,
        )
        self.state_samples_history.append(state_samples)
        self.input_samples_history.append(input_samples)
        self.noiseless_states_history.append(noiseless_samples)
        self.num_data_collection += 1
        # add data to model_learning object
        self.model_learning.add_data(new_state_samples=state_samples, new_input_samples=input_samples)

    def load_policy_from_log(self, num_trial, folder="results_tmp/1/"):
        """
        Load control policy of trial: num_trial from log file inside 'folder'
        """
        log_file_path = folder + "log.pkl"
        print("\nLoading policy from: " + log_file_path)
        log_dict = pkl.load(open(log_file_path, "rb"))
        trial_index = num_trial - 1
        self.control_policy.load_state_dict(log_dict["parameters_trial_list"][trial_index])

    def load_model_from_log(self, num_trial, folder="results_tmp/1/"):
        """
        Load model of trial: num_trial from log file inside 'folder'
        """
        log_file_path = folder + "log.pkl"
        print("\nLoading model from: " + log_file_path)

        log_dict = pkl.load(open(log_file_path, "rb"))
        self.log_dict = log_dict

        self.log_dict["cost_trial_list"] = self.log_dict["cost_trial_list"][0:num_trial]
        self.log_dict["parameters_trial_list"] = self.log_dict["parameters_trial_list"][0:num_trial]
        self.log_dict["particles_states_list"] = self.log_dict["particles_states_list"][0:num_trial]
        self.log_dict["particles_inputs_list"] = self.log_dict["particles_inputs_list"][0:num_trial]

        # load data up to num_trial (included)
        for j in range(num_trial + 1):
            print("\nGet data from trial: " + str(j) + "/" + str(num_trial))
            state_samples = log_dict["state_samples_history"][j]
            input_samples = log_dict["input_samples_history"][j]
            noiseless_state_samples = log_dict["noiseless_states_history"][j]
            # add samples history
            self.state_samples_history.append(state_samples)
            self.input_samples_history.append(input_samples)
            self.noiseless_states_history.append(noiseless_state_samples)
            self.num_data_collection += 1
            # add data to model_learning object
            self.model_learning.add_data(new_state_samples=state_samples, new_input_samples=input_samples)

        trial_index = num_trial - 1

        # load gp models of trial: trial_index
        print("\nGet parameters")
        self.model_learning.gp_inputs = log_dict["gp_inputs_" + str(trial_index)]
        self.model_learning.gp_output_list = log_dict["gp_output_list_" + str(trial_index)]
        for k in range(self.model_learning.num_gp):
            self.model_learning.gp_list[k].load_state_dict(log_dict["parameters_gp_" + str(trial_index)][k])

        # pre-train gp models
        for k in range(self.model_learning.num_gp):
            self.model_learning.pretrain_gp(k)


class MC_PILCO4PMS(MC_PILCO):
    """
    MC-PILCO for Partially Measurable Systems (PMS)
    """

    def __init__(
        self,
        T_sampling,
        state_dim,
        input_dim,
        f_sim,
        f_model_learning,
        model_learning_par,
        f_rand_exploration_policy,
        rand_exploration_policy_par,
        f_control_policy,
        control_policy_par,
        f_cost_function,
        cost_function_par,
        pos_indeces,
        vel_indeces,
        std_meas_noise=None,
        log_path=None,
        filtering_dict={},
        std_meas_noise_sim=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):
        super(MC_PILCO4PMS, self).__init__(
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

        self.system = model.PMS_Model(f_sim, filtering_dict)
        self.filtering_dict = filtering_dict  # defines the filtering performed online
        self.pos_indeces = pos_indeces
        self.vel_indeces = vel_indeces
        if std_meas_noise_sim is None:
            self.std_meas_noise_sim = std_meas_noise

    def apply_policy(
        self,
        particles_initial_state_mean,
        particles_initial_state_var,
        flg_particles_init_uniform,
        particles_init_up_bound,
        particles_init_low_bound,
        flg_particles_init_multi_gauss,
        num_particles,
        T_control,
        p_dropout,
    ):
        """
        Apply the policy in simulation having the possibility to only measure some state
        variables (positions) and the necessity to derive the others (velocities) by filtering.
        ###################################################################################
        IMPORTANT: change the code according to how the measures are taken and filtered!!!
        ###################################################################################
        """
        # initialize variables
        states_sequence_list = []
        inputs_sequence_list = []

        # get initial particles
        if flg_particles_init_uniform == True:
            # initial uniform distribution
            uniform_ub_particles = particles_init_up_bound.repeat(num_particles, 1)
            uniform_lb_particles = particles_init_low_bound.repeat(num_particles, 1)
            state_distribution = Uniform(uniform_lb_particles, uniform_ub_particles)
        elif flg_particles_init_multi_gauss == True:
            # initial multimodal Gaussian distribution
            indices = torch.randint(0, particles_initial_state_mean.shape[0], [num_particles])
            initial_particles_mean = particles_initial_state_mean[indices, :]
            initial_particles_cov_mat = torch.stack([torch.diag(particles_initial_state_var[i, :]) for i in indices])
            state_distribution = MultivariateNormal(
                loc=initial_particles_mean, covariance_matrix=initial_particles_cov_mat
            )
        else:
            # initial Gaussian distribution
            initial_particles_mean = particles_initial_state_mean.repeat(num_particles, 1)
            initial_particles_cov_mat = torch.stack([torch.diag(particles_initial_state_var)] * num_particles)
            state_distribution = MultivariateNormal(
                loc=initial_particles_mean, covariance_matrix=initial_particles_cov_mat
            )

        # sample particles at t=0 from initial state distribution
        states_sequence_list.append(state_distribution.rsample())
        # simulated measurement (at t=0 it is the true state)
        meas_state_samples = states_sequence_list[0].clone()

        # init low-pass filter
        b, a = signal.butter(1, self.filtering_dict["fc"])

        # measurement noise
        std_pos_noise = torch.tensor(self.std_meas_noise_sim[self.pos_indeces], dtype=self.dtype, device=self.device)

        # init measured and noisy state samples
        meas_states_sequence_list = []
        meas_states_sequence_list.append(meas_state_samples)
        noisy_states_sequence_list = []
        noisy_states_sequence_list.append(meas_state_samples)

        # compute initial inputs
        inputs_sequence_list.append(self.control_policy(meas_states_sequence_list[0], t=0, p_dropout=p_dropout))

        for t in range(1, int(T_control)):
            # get next state mean and variance (given the states sampled and the inputs computed)
            particles, _, _ = self.model_learning.get_next_state(
                current_state=states_sequence_list[t - 1], current_input=inputs_sequence_list[t - 1]
            )
            states_sequence_list.append(particles)

            # get the noisy states (add noise to positions)
            noisy_state_samples = states_sequence_list[t].clone()
            noisy_state_samples[:, self.pos_indeces] = noisy_state_samples[
                :, self.pos_indeces
            ] + std_pos_noise * torch.randn(num_particles, len(self.pos_indeces), device=self.device, dtype=self.dtype)
            noisy_states_sequence_list.append(noisy_state_samples)

            # measure velocities by numerical differentiation
            noisy_states_sequence_list[t][:, self.vel_indeces] = (
                noisy_states_sequence_list[t][:, self.pos_indeces]
                - noisy_states_sequence_list[t - 1][:, self.pos_indeces]
            ) / self.T_sampling

            # online low-pass filtering of velocities
            meas_state_samples = noisy_state_samples.clone()
            meas_state_samples[:, self.vel_indeces] = (
                b[0] * noisy_states_sequence_list[t][:, self.vel_indeces]
                + b[1] * noisy_states_sequence_list[t - 1][:, self.vel_indeces]
                - a[1] * meas_states_sequence_list[t - 1][:, self.vel_indeces]
            ) / a[0]
            meas_states_sequence_list.append(meas_state_samples)

            # compute next input
            inputs_sequence_list.append(self.control_policy(meas_states_sequence_list[t], t=t, p_dropout=p_dropout))

        # returns states/inputs trajectory
        return torch.stack(states_sequence_list), torch.stack(inputs_sequence_list)

    def get_data_from_system(self, initial_state, T_exploration, trial_index, flg_exploration=False):
        """
        Apply exploration/control policy to the system and collect interaction data
        """
        # select the policy
        if flg_exploration:
            current_policy = self.rand_exploration_policy
        else:
            current_policy = self.control_policy

        meas_states, input_samples, noiseless_samples, noisy_samples = self.system.rollout(
            s0=initial_state,
            policy=current_policy.get_np_policy(),
            T=T_exploration,
            dt=self.T_sampling,
            noise=self.std_meas_noise,
            vel_indeces=self.vel_indeces,
            pos_indeces=self.pos_indeces,
        )

        # approximate velocities using position measurements
        state_samples, meas_states, input_samples, noiseless_samples, noisy_samples = self.get_velocities(
            meas_states, input_samples, noiseless_samples, noisy_samples
        )

        self.state_samples_history.append(state_samples)
        self.input_samples_history.append(input_samples)
        self.noiseless_states_history.append(noiseless_samples)
        self.num_data_collection += 1
        # add data to model_learning object
        self.model_learning.add_data(new_state_samples=state_samples, new_input_samples=input_samples)

    def get_velocities(self, meas_states, input_samples, noiseless_samples, noisy_samples):
        """
        Offline state filtering for modeling
        ###################################################################################
        IMPORTANT: change the code according to how you intend to filter data for the model
        ###################################################################################
        """
        state_samples = np.zeros([noisy_samples.shape[0] - 2, noisy_samples.shape[1]])
        for i in range(len(self.pos_indeces)):
            b, a = signal.butter(2, 0.5)
            # positions are filtered
            pos = signal.filtfilt(b, a, noisy_samples[:, self.pos_indeces[i]])
            # velocities are computed by means of central difference
            vel = (pos[2:] - pos[:-2]) / (2 * self.T_sampling)
            # discard first and last samples
            state_samples[:, self.pos_indeces[i]] = pos[1:-1]
            state_samples[:, self.vel_indeces[i]] = vel
        input_samples = input_samples[1:-1, :]
        noiseless_samples = noiseless_samples[1:-1, :]
        meas_states = meas_states[1:-1, :]
        noisy_samples = noisy_samples[1:-1, :]

        return state_samples, meas_states, input_samples, noiseless_samples, noisy_samples


class MC_PILCO_Experiment(MC_PILCO4PMS):
    """
    Extend MC_PILCO_4RS class for hardware experiments.
    """

    def __init__(
        self,
        T_sampling,
        state_dim,
        input_dim,
        f_sim,
        f_model_learning,
        model_learning_par,
        f_rand_exploration_policy,
        rand_exploration_policy_par,
        f_control_policy,
        control_policy_par,
        f_cost_function,
        cost_function_par,
        pos_indeces,
        vel_indeces,
        std_meas_noise=None,
        log_path=None,
        filtering_dict={},
        std_meas_noise_sim=None,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ):
        super(MC_PILCO_Experiment, self).__init__(
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
            pos_indeces=pos_indeces,
            vel_indeces=vel_indeces,
            std_meas_noise=std_meas_noise,
            log_path=log_path,
            filtering_dict=filtering_dict,
            std_meas_noise_sim=std_meas_noise_sim,
            dtype=dtype,
            device=device,
        )

    def get_data_from_system(self, initial_state, T_exploration, trial_index, flg_exploration=False):
        """
        Load interaction data from txt files.
        """
        # select the policy
        if flg_exploration:
            print("Execute initial exploration policy")
        else:
            print("Export control policy parameters")
            for name, param in self.control_policy.named_parameters():
                np.savetxt("policy_" + name + ".csv", param.data.cpu().numpy(), delimiter=",")

        # from the txt we need:
        # -input_samples
        # -noisy_samples (positions and velocities, velocities are ignored)

        done = False
        while not done:
            print("Save noisy state samples in: " + self.log_path + "/DATA_" + str(trial_index) + "/noisy_samples.csv")
            print("Save input samples in: " + self.log_path + "/DATA_" + str(trial_index) + "/input_samples.csv")
            print('Press any key when done (press "q" to exit)')
            cmd = str(input())
            if cmd == "q":
                done = True
            else:
                try:
                    noisy_samples = np.genfromtxt(
                        self.log_path + "/DATA_" + str(trial_index) + "/noisy_samples.csv", delimiter=","
                    )
                    input_samples = np.genfromtxt(
                        self.log_path + "/DATA_" + str(trial_index) + "/input_samples.csv", delimiter=","
                    ).reshape([-1, self.input_dim])
                except:
                    print("Files not found!")
                else:
                    if noisy_samples.shape[1] == self.state_dim and input_samples.shape[0] == noisy_samples.shape[0]:
                        done = True
                    else:
                        print("Data dimensions are not correct! Try again.")

        meas_states = noisy_samples
        noiseless_samples = noisy_samples

        # approximate velocities using position measures
        state_samples, meas_states, input_samples, noiseless_samples, noisy_samples = self.get_velocities(
            meas_states, input_samples, noiseless_samples, noisy_samples
        )

        self.state_samples_history.append(state_samples)
        self.input_samples_history.append(input_samples)
        self.num_data_collection += 1
        # add data to model_learning object
        self.model_learning.add_data(new_state_samples=state_samples, new_input_samples=input_samples)
