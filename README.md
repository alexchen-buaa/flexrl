# FlexRL

FlexRL is a deep online/offline reinforcement learning library inspired and adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl) and [CORL](https://github.com/tinkoff-ai/CORL) that provides single-file implementations of algorithms that aren't necessarily covered by these libraries. FlexRL introduces the following features:
- Consistent style across online and offline algorithms
- Easy configuration with [Pyrallis](https://github.com/eladrich/pyrallis) and [tqdm](https://github.com/tqdm/tqdm) progress bar
- A few custom environments under `gym` API

## Quick Start

### Installing FlexRL

```bash
git clone https://github.com/alexchen-buaa/flexrl.git
cd flexrl
pip install -e .
```

### Usage

Run the algorithms as individual scripts. Like CORL, we use [Pyrallis](https://github.com/eladrich/pyrallis) for configuration management. The arguments can be specified using command-line arguments, a `yaml` file, or both:
```bash
python ppo.py --config_path=some_config.yaml
```

### Algorithms Implemented

| Type     | Algorithm                          | Variants Implemented                                           |
| -------- | ---------------------------------- | -------------------------------------------------------------- |
| Online   | Proximal Policy Optimization (PPO) | [ppo.py](src/flexrl/online/ppo.py)                             |
|          |                                    | [ppo_atari.py](src/flexrl/online/ppo_atari.py)                 |
|          |                                    | [ppo_multidiscrete.py](src/flexrl/online/ppo_multidiscrete.py) |
|          | Deep Q-Networks (DQN)              | [dqn.py](src/flexrl/online/dqn.py)                             |
|          |                                    | [dqn_atari.py](src/flexrl/online/dqn_atari.py)                 |
|          | Quantile-Regression DQN (QR-DQN)   | [qr_dqn.py](src/flexrl/online/qr_dqn.py)                       |
|          |                                    | [qr_dqn_atari.py](src/flexrl/online/qr_dqn_atari.py)           |
|          | Soft Actor-Critic (SAC)            | [sac.py](src/flexrl/online/sac.py)                             |
| Offline  | Implicit Q-Learning (IQL)          | [iql.py](src/flexrl/offline/iql.py)                            |
|          |                                    | [iql_jax.py](src/flexrl/offline/iql_jax.py)                    |
|          | In-Sample Actor-Critic (InAC)      | [inac.py](src/flexrl/offline/inac.py)                          |
|          |                                    | [inac_jax.py](src/flexrl/offline/inac_jax.py)                  |

### Extra Requirements

#### Atari/ALE

According to [The Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment), you can use the command line tool to import your ROMS:

```bash
ale-import-roms roms/
```

#### MuJoCo

To use MuJoCo envs (for both online training and offline evaluation), you need to install MuJoCo first. See [mujoco-py](https://github.com/openai/mujoco-py) for instructions.

#### JAX with CUDA Support

To use JAX with CUDA support, you need to install the NVIDIA driver first. See [JAX Installation](https://github.com/google/jax#installation) for instructions.

### References

- [1] S. Huang, R. F. J. Dossa, C. Ye, and J. Braga, “CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms.” arXiv, Nov. 16, 2021. Accessed: Nov. 21, 2022. [Online]. Available: http://arxiv.org/abs/2111.08819
- [2] Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann, “Stable-Baselines3: Reliable Reinforcement Learning Implementations,” Journal of Machine Learning Research, vol. 22, no. 268, pp. 1–8, 2021.
- [3] W. Dabney, M. Rowland, M. G. Bellemare, and R. Munos, “Distributional Reinforcement Learning with Quantile Regression,” arXiv:1710.10044 [cs, stat], Oct. 2017, Accessed: Apr. 15, 2022. [Online]. Available: http://arxiv.org/abs/1710.10044
- [4] I. Kostrikov, A. Nair, and S. Levine, “Offline Reinforcement Learning with Implicit Q-Learning.” arXiv, Oct. 12, 2021. Accessed: Mar. 29, 2023. [Online]. Available: http://arxiv.org/abs/2110.06169
- [5] C. Xiao, H. Wang, Y. Pan, A. White, and M. White, “The In-Sample Softmax for Offline Reinforcement Learning.” arXiv, Feb. 28, 2023. Accessed: Apr. 02, 2023. [Online]. Available: http://arxiv.org/abs/2302.14372
