# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL contact:	Diego Romeres (romeres@merl.com)
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.integrate import odeint


class Model:
    """
    Dynamic System simulation
    """

    def __init__(self, fcn):
        """
        fcn: function defining the dynamics dx/dt = fcn(x, t, policy)
        x: state (it can be an array)
        t: time instant
        policy: function u = policy(x)
        """
        self.fcn = fcn  # ODE of system dynamics

    def rollout(self, s0, policy, T, dt, noise):
        """
        Generate a rollout of length T (s)  with control inputs computed by 'policy' and applied with a sampling time 'dt'.
        'noise' defines the standard deviation of a Gaussian measurement noise.
            s0: initial state
            policy: policy function
            T: length rollout (s)
            dt: sampling time (s)
        """
        state_dim = len(s0)
        time = np.linspace(0, T, int(T / dt) + 1)
        num_samples = len(time)

        # get first input
        u0 = np.array(policy(s0, 0.0))

        num_inputs = u0.size
        # init variables
        inputs = np.zeros([num_samples, num_inputs])
        states = np.zeros([num_samples, state_dim])
        noisy_states = np.zeros([num_samples, state_dim])
        states[0, :] = s0
        noisy_states[0, :] = s0 + np.random.randn(state_dim) * noise

        for i, t in enumerate(time[:-1]):
            # get input
            u = np.array(policy(noisy_states[i, :], t))
            inputs[i, :] = u
            # get state
            odeint_out = odeint(self.fcn, states[i, :], [t, t + dt], args=(u,))
            states[i + 1, :] = odeint_out[1]
            noisy_states[i + 1, :] = odeint_out[1] + np.random.randn(state_dim) * noise

        # last u (only to have the same number of input and state samples)
        inputs[-1, :] = np.array(policy(noisy_states[-1, :], T))

        return noisy_states, inputs, states


class PMS_Model:
    """
    Partially Measurable System simulation
    """

    def __init__(self, fcn, filtering_dict):
        """
        fcn: function defining the dynamics dx/dt = fcn(x, t, policy)
        x: state (it can be an array)
        t: time instant
        policy: function u = policy(x)
        In filtering dict are passed the parameters of the online filter
        """
        self.fcn = fcn
        self.filtering_dict = filtering_dict

    def rollout(self, s0, policy, T, dt, noise, vel_indeces, pos_indeces):
        """
        Generate a rollout of length T (s) for the system defined by 'fcn' with
        control inputs computed by 'policy' and applied with a sampling time 'dt'.
        'noise' defines the standard deviation of a Gaussian measurement noise.
        In this implementation we simulate the interaction with a real mechanical system where
        velocities cannot be measured, but only inferred from the positions.
            s0: initial state
            policy: policy function
            T: length rollout (s)
            dt: sampling time (s)

        """
        state_dim = len(s0)
        time = np.linspace(0, T, int(T / dt) + 1)
        num_samples = len(time)
        # get input size
        num_inputs = np.array(policy(s0, 0.0)).size
        # allocate the space
        inputs = np.zeros([num_samples, num_inputs])
        states = np.zeros([num_samples, state_dim])
        noisy_states = np.zeros([num_samples, state_dim])
        meas_states = np.zeros([num_samples, state_dim])
        # initialize state vectors
        states[0, :] = s0
        noisy_states[0, :] = s0
        meas_states[0, :] = np.copy(noisy_states[0, :])

        # init low-pass filter
        b, a = signal.butter(1, self.filtering_dict["fc"])

        for i, t in enumerate(time[:-1]):
            # get input
            u = np.array(policy(meas_states[i, :], t))
            inputs[i, :] = u
            # get state
            odeint_out = odeint(self.fcn, states[i, :], [t, t + dt], args=(u,))
            states[i + 1, :] = odeint_out[1]
            noisy_states[i + 1, :] = odeint_out[1] + np.random.randn(state_dim) * noise

            # positions are measured directly
            meas_states[i + 1, pos_indeces] = noisy_states[i + 1, pos_indeces]
            # velocities are estimated online by causal numerical differentiation ...
            noisy_states[i + 1, vel_indeces] = (meas_states[i + 1, pos_indeces] - meas_states[i, pos_indeces]) / dt
            # ... and low-pass filtered
            meas_states[i + 1, vel_indeces] = (
                b[0] * noisy_states[i + 1, vel_indeces]
                + b[1] * noisy_states[i, vel_indeces]
                - a[1] * meas_states[i, vel_indeces]
            ) / a[0]

        # last u (only to have the same number of input and state samples)
        inputs[-1, :] = np.array(policy(meas_states[-1, :], T))

        return meas_states, inputs, states, noisy_states
