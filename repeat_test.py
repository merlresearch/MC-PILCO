# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL:	    Diego Romeres (romeres@merl.com)
"""

"""
Repeat the same test file with different random seeds
"""
import os

seed_list = range(1, 51)
file_name = "test_mcpilco_cartpole.py"
# file_name = 'test_mcpilco_cartpole_rbf_kernel.py'


for seed in seed_list:
    str_command = "python " + file_name + " -seed " + str(seed)
    print("\n##########\nstr_command: " + str_command + "\n##########")
    os.system(str_command)
