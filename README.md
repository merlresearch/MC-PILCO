<!--
Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->
# MC-PILCO

This package implements a Model-based Reinforcement Learning algorithm called Monte Carlo Probabilistic Inference for Learning and COntrol (MC-PILCO), for modeling and control of dynamical system. The algorithm relies on Gaussian Processes (GPs) to model the system dynamics and on a Monte Carlo approach to estimate the policy gradient during optimization. The Monte Carlo approach is shown to be effective for policy optimization thanks to a proper cost function shaping and use of dropout. This defines a framework where we can study: (i) the selection of the
cost function, (ii) the optimization of policies using dropout,
(iii) an improved data efficiency through the use of structured
kernels in the GP models
The algorithm is also extended to real systems and in particular to Partially Measurable Systems and it takes the name of MC-PILCO-4PMS.
We discuss the importance of modeling both the measurement system and
the state estimators during policy optimization. Please, see the [paper](https://ieeexplore.ieee.org/abstract/document/9827590) for more details.

## Features

The code is implemented in python3 and reproduces all the simulation examples in [MC-PILCO](https://ieeexplore.ieee.org/abstract/document/9827590), namely, multiple ablation studies and the solution of a cart-pole system swing-up (available both in a python simulated environment and in the physic engine MuJoCo), a trajectory controller for a UR5 (implemented in Mujoco). The results can be reproduced with statistical values via Monte Carlo simulations.
The user has the possibility to add his own python system or Mujoco Environment and solve it with MC-PILCO or MC-PILCO-4PMS.

Please refer to the [guide](./MC_PILCO_Software_Package.pdf) for a more detailed explanation of the code base.

## Installation

1. Download / Git clone this repo.
2. Create a python environment with the following packages:

### Dependencies

- [PyTorch 1.4 or higher](<https://pytorch.org/>)
- [NumPy](<https://numpy.org/>)
- [Matplotlib](<https://matplotlib.org/>)
- [Pickle](<https://docs.python.org/3/library/pickle.html>)
- [Argparse](<https://docs.python.org/3/library/argparse.html>)
- `gpr_lib` is provided courtesy of Alberto Dalla Libera with permission to redistribute as part of this software package (License: `MIT`).

If you want to test the code on MuJoCo environments, make sure to have also MuJoCo_py and Gym libraries.

### Optional Dependencies

- [MuJoCo 2.00](http://www.mujoco.org/) (License: `Apache-2.0`)
- [MuJoCo-Py](<https://github.com/openai/mujoco-py>) (License: `MIT`)
- [Gym](<http://gym.openai.com/>) (License: `MIT`)

You can create a conda environment with the above dependencies by executing:

```bash
conda env create --file environment.yaml
```

## Usage

Please refer to the guide [MC_PILCO_Software_Package.pdf](./MC_PILCO_Software_Package.pdf).

## Testing

Inside 'mc_pilco' folder:

- Run `$ python test_mcpilco_cartpole.py` to test MC-PILCO in the cartpole swing-up task (GP model with squared-exponential+polynomial kernel).
- Run `$ python test_mcpilco_cartpole_rbf_ker.py` to test MC-PILCO in the cartpole swing-up task (GP model with squared-exponential kernel).
- Run `$ python test_mcpilco_cartpole_multi_init.py` to test MC-PILCO in the cartpole swing-up task stating from two separate possible initial cart positions.
- Run `$ python test_mcpilco4pms_cartpole.py` to test MC-PILCO4PMS in the cartpole swing-up task when considering the presence of sensors and state estimation.
- Run `$ python test_mcpilco_cartpole_mujoco.py` to test MC-PILCO in the cartpole swing-up task in MuJoCo.
- Run `$ python test_mcpilco_ur5_mujoco.py` to use MC-PILCO to learn a joint-space controller for a UR5 robot arm in MuJoCo.

## Citation

If you use the software, please cite the following ([paper](https://ieeexplore.ieee.org/abstract/document/9827590)):

Amadio, F., Dalla Libera, A., Antonello, R., Nikovski, D., Carli, R., & Romeres, D. (2022). Model-Based Policy Search Using Monte Carlo Gradient Estimation With Real Systems Application. IEEE Transactions on Robotics, 38(6), 3879-3898.

```BibTeX
@article{amadio2022model,
  title={Model-Based Policy Search Using Monte Carlo Gradient Estimation With Real Systems Application},
  author={Amadio, Fabio and Dalla Libera, Alberto and Antonello, Riccardo and Nikovski, Daniel and Carli, Ruggero and Romeres, Diego},
  journal={IEEE Transactions on Robotics},
  volume={38},
  number={6},
  pages={3879--3898},
  year={2022},
  publisher={IEEE}
}
```

## Related Links

You can find more information and videos at:

[MERL research page](https://www.merl.com/research/license/MC-PILCO)

[Youtube Presentation](https://www.youtube.com/watch?v=--73hmZYaHA)

## Contact

Diego Romeres: romeres@merl.com

Alberto Dalla Libera: dallaliber@dei.unipd.it

Fabio Amadio: amadiofa@dei.unipd.it

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for our policy on contributions.

## License
Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files:
```
Copyright (c) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
```
