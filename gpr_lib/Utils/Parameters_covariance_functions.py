# Copyright (C) 2023 Alberto Dalla Libera
#
# SPDX-License-Identifier: MIT
"""
Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
"""
"""
This file contains a collections of functions that returns valid covariance functions
The inputs of these functions is standardize in order to be:
-pos_parameters: a torch tensor that must be positive
-free_par: a torch tensors whose values are unconstrained
-additional_par: additional parameters of the function
"""
import numpy as np
import torch


def diagonal_covariance(pos_par=None, free_par=None, num_par=None, flg_ARD=False):
    """Returns a diagonal covariance matrix. if flg_ARD is false all the element alonf the diagonal are equals"""
    if flg_ARD:
        # check dimensions and return the matrix
        if num_par == pos_par.size()[0]:
            return torch.diag(pos_par**2)
        else:
            raise RuntimeError("The number of positive parameters and num_par must be equal when flg_ARD=True")
    else:
        return pos_par**2 * torch.eye(num_par, dtype=pos_par.dtype, device=pos_par.device)


def diagonal_covariance_ARD(pos_par=None, free_par=None):
    """Returns a diagonal covariance matrix with ARD"""
    return torch.diag(pos_par**2)


def diagonal_covariance_semi_def(pos_par=None, free_par=None):
    """Returns a diagonal covariance matrix with dimension num_pos_par+num_free_par.
    The firsts elements of the diag are equal to free_par**2, while the last elements
    are equal to pos_par**2
    """
    if pos_par is None:
        pos_par = torch.tensor([], dtype=free_par.dtype, device=free_par.device)
    return torch.diag(torch.cat([free_par, pos_par]) ** 2)
    # return torch.diag(torch.cat([free_par, pos_par]))


def full_covariance(pos_par, free_par, num_row):
    """Returns a full covariance parametrixed through the elements of the cholesky decomposition"""
    # map the par in a vect
    parameters_vector = par2vect_chol(pos_par, free_par, num_row)
    # map the vect in the uppper triangular matrix
    U = torch.zeros(num_row, num_row, dtype=pos_par.dtype, device=pos_par.device)
    U[torch.triu(torch.ones(num_row, num_row, dtype=pos_par.dtype, device=pos_par.device)) == 1] = parameters_vector
    # get the sigma
    return torch.matmul(U.transpose(1, 0), U)


def par2vect_chol(pos_par_vect, free_par_vect, num_row):
    """Maps pos_par and free_par in the chol vector"""
    vect = torch.tensor([], dtype=pos_par_vect.dtype, device=pos_par_vect.device)
    free_par_index_to = 0
    for row in range(0, num_row):
        free_par_index_from = free_par_index_to
        free_par_index_to = free_par_index_to + num_row - row - 1
        vect = torch.cat([vect, pos_par_vect[row].reshape(1), free_par_vect[free_par_index_from:free_par_index_to]])
    return vect


def get_initial_par_chol(num_row, mode="Identity"):
    """Returns numpy initialization of pos_par and free par for the upper triangular
    cholesky decomposition
    """
    num_free_par = int(num_row * (num_row - 1) / 2)
    if mode == "Identity":
        pos_par = np.ones(num_row)
        free_par = np.zeros(num_free_par)
    elif mode == "Random":
        pos_par = np.ones(num_row)
        free_par = 0.01 * np.random.randn(num_free_par)
    else:
        print("Specify an initialization mode!")
        raise RuntimeError
    return pos_par, free_par
