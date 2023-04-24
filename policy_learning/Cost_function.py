# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL contact:	Diego Romeres (romeres@merl.com)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


class Expected_cost(torch.nn.modules.loss._Loss):
    """
    Expected cost class. The cost is computed through a torch function defined in the initialization
    """

    def __init__(self, cost_function):
        """Initialize the object"""
        super(Expected_cost, self).__init__()
        self.cost_function = cost_function

    def forward(self, states_sequence, inputs_sequence, trial_index=None):
        """Computes the global cost applying self.cost_function.
        States_sequence.shape: [num_instants, num_particles, state_dim]
        inputs_sequence.shape: [num_instants, num_particles, input_dim]
        """

        # Returns the sum of the expected costs
        costs = self.cost_function(states_sequence, inputs_sequence, trial_index)
        mean_costs = torch.mean(costs, 1)  # average cost at each time step over particles ...
        std_costs = torch.std(costs.detach(), 1)  # ... and corresponding std

        return torch.sum(mean_costs), torch.sum(std_costs)


class Expected_distance(Expected_cost):
    """
    Cost function given by the sum of the expected distances from target state
    """

    def __init__(self, target_state, lengthscales, active_dims):
        # get the distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: distance_from_target(
            x, u, trial_index, target_state=target_state, lengthscales=lengthscales, active_dims=active_dims
        )
        # initit the superclass with the lambda function
        super(Expected_distance, self).__init__(f_cost)


def distance_from_target(states_sequence, inputs_sequence, trial_index, target_state, lengthscales, active_dims):
    # normalize states and targets (consider only used states)
    norm_states = states_sequence[:, :, active_dims] / lengthscales
    norm_target = target_state / lengthscales

    # get the square distance
    dist = torch.sum(norm_states**2, dim=2, keepdim=True)
    dist = dist + torch.sum(norm_target**2, dim=1, keepdim=True).transpose(0, 1)
    dist -= 2 * torch.matmul(norm_states, norm_target.transpose(dim0=0, dim1=1))
    # return the cost
    return dist


class Expected_saturated_distance(Expected_cost):
    """
    Cost function given by the sum of the expected saturated distances from target state
    """

    def __init__(self, target_state, lengthscales, active_dims):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: saturated_distance_from_target(
            x, u, trial_index, target_state=target_state, lengthscales=lengthscales, active_dims=active_dims
        )
        # initit the superclass with the lambda function
        super(Expected_saturated_distance, self).__init__(f_cost)


def saturated_distance_from_target(
    states_sequence, inputs_sequence, trial_index, target_state, lengthscales, active_dims
):
    """
    The saturated distance defined as:
    1 - exp(-(target_state - states_sequence)^T*(diag(lengthscales^2)^(-1)*(target_state - states_sequence))
    """

    # get state components evaluated in the cost
    active_states = states_sequence[:, :, active_dims]

    # normalize states and targets
    norm_states = active_states / lengthscales
    norm_target = target_state / lengthscales
    # get the square distance
    dist = torch.sum(norm_states**2, dim=2, keepdim=True)
    dist = dist + torch.sum(norm_target**2, dim=1, keepdim=True).transpose(0, 1)
    dist -= 2 * torch.matmul(norm_states, norm_target.transpose(dim0=0, dim1=1))

    cost = 1 - torch.exp(-dist)

    return cost


class Expected_saturated_distance_from_trajectory(Expected_cost):
    """
    Cost function given by the sum of the expected saturated distances from a target trajectory
    """

    def __init__(self, target_traj, lengthscales, flg_var_lengthscales=False, used_indeces=None):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: saturated_distance_from_trajectory(
            x,
            u,
            trial_index,
            target_traj=target_traj,
            lengthscales=lengthscales,
            flg_var_lengthscales=flg_var_lengthscales,
            used_indeces=used_indeces,
        )
        # initit the superclass with the lambda function
        super(Expected_saturated_distance_from_trajectory, self).__init__(f_cost)


def saturated_distance_from_trajectory(
    states_sequence, inputs_sequence, trial_index, target_traj, lengthscales, flg_var_lengthscales, used_indeces
):
    """
    The saturated distance defined as:
    1 - exp(-(target_state - states_sequence)^T*(diag(lengthscales^2)^(-1)*(target_state - states_sequence))
    """
    if used_indeces == None:
        used_indeces = list(range(0, states_sequence.shape[2]))

    # get state components evaluated in the cost
    targets = target_traj.repeat(1, states_sequence.shape[1]).view(states_sequence.shape)
    if flg_var_lengthscales:
        dist = torch.sum(
            ((states_sequence[:, :, used_indeces] - targets[:, :, used_indeces]) / lengthscales[trial_index]) ** 2,
            dim=2,
        )
    else:
        dist = torch.sum(
            ((states_sequence[:, :, used_indeces] - targets[:, :, used_indeces]) / lengthscales) ** 2, dim=2
        )
    cost = 1 - torch.exp(-dist)

    return cost


class Cart_pole_cost(Expected_cost):
    """Cost for the cart pole system:
    target is assumed in the instable equilibrium configuration defined in 'target_state' (target angle [rad], target position [m]).
    """

    def __init__(self, target_state, lengthscales, angle_index, pos_index):
        # get the saturated distance function as a function of states and inputs
        f_cost = lambda x, u, trial_index: cart_pole_cost(
            x,
            u,
            trial_index,
            target_state=target_state,
            lengthscales=lengthscales,
            angle_index=angle_index,
            pos_index=pos_index,
        )
        # initit the superclass with the lambda function
        super(Cart_pole_cost, self).__init__(f_cost)


def cart_pole_cost(states_sequence, inputs_sequence, trial_index, target_state, lengthscales, angle_index, pos_index):
    """
    Cost function given by the combination of the saturated distance between |theta| and 'target angle', and between x and 'target position'.
    """
    x = states_sequence[:, :, pos_index]
    theta = states_sequence[:, :, angle_index]

    target_x = target_state[1]
    target_theta = target_state[0]

    return 1 - torch.exp(
        -(((torch.abs(theta) - target_theta) / lengthscales[0]) ** 2) - ((x - target_x) / lengthscales[1]) ** 2
    )
