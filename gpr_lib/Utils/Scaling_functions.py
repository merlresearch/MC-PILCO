# Copyright (C) 2023 Alberto Dalla Libera
#
# SPDX-License-Identifier: MIT

"""
Author: Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
"""
"""
This file contains a collections of functions that can be used to scale GP_prior.
the inputs of the scaling funcitons are:
-X_active
-pos_par
-free_par
-additional_par
If the input X_active is NxD the output mus be a Nx1 vector
"""

import torch


def f_get_sign(X_active, pos_par=None, free_par=None, flg_sign_pos=True):
    """Returns a vecor containing ones and zeros,
    depending on the fact that X is positive or negative.
    """
    offset = torch.zeros(1)
    if not (free_par is None):
        offset = offset + free_par
    if flg_sign_pos:
        print("pos")
        return torch.prod(X_active > offset, 1, keepdim=True, dtype=offset.dtype)
    else:
        return torch.prod(X_active < offset, 1, keepdim=True, dtype=offset.dtype)


def f_get_sign_abs(X_active, pos_par=None, free_par=None, flg_sign_pos=True):
    """Returns a vecor containing zeros and ones,
    depending on the fact that X is positive or negative.
    """
    if flg_sign_pos:
        return torch.prod(torch.abs(X_active) > pos_par, 1, keepdim=True, dtype=pos_par.dtype)
    else:
        return torch.prod(torch.abs(X_active) < pos_par, 1, keepdim=True, dtype=pos_par.dtype)
