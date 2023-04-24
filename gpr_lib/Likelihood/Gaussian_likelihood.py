# Copyright (C) 2023 Alberto Dalla Libera
#
# SPDX-License-Identifier: MIT
"""
Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
"""

import numpy as np
import torch


class Marginal_log_likelihood(torch.nn.modules.loss._Loss):
    """Computes the negative marginal log likelihood under gaussian assumption"""

    def forward(self, output_GP_prior, Y):
        """returns the negative marginal log likelihood:
        0.5*( (Y-m_X)^T*K_X_inv*(Y-m_x) + log_det(K_X) + N*log(2*pi) )"""
        m_X, _, K_X_inv, log_det = output_GP_prior
        Y = Y - m_X
        N = Y.size()[0]
        MLL = torch.matmul(Y.transpose(0, 1), torch.matmul(K_X_inv, Y))
        # MLL += log_det + N*np.log( 2*np.pi)
        MLL += log_det  # + N*np.log( 2*np.pi)
        return 0.5 * MLL


class Posterior_log_likelihood(torch.nn.modules.loss._Loss):
    """Computes an approxmation of the posterior negative marginal log likelihood,
    where each sample is assumed gaussian and independent, i.e. the
    covariance matrix is diagonal"""

    def forward(self, Y, Y_hat, var):
        # subtract the mean
        Y -= Y_hat
        # get the approximate likelihood
        MLL = torch.sum(Y**2 / (2 * var) + 0.5 * torch.log(var))
        return MLL
