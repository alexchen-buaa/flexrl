# FlexRL

FlexRL is a deep online/offline reinforcement learning library inspired and adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl) and [CORL](https://github.com/tinkoff-ai/CORL) that provides single-file implementations of algorithms that aren't necessarily covered by these libraries. FlexRL introduces the following features:
- Consistent style across online and offline algorithms
- Easy configuration with [Pyrallis](https://github.com/eladrich/pyrallis) and progress bar
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

### Atari Support

For Atari support, `flexrl` has `ale-py` as a dependency. So according to [The Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment), you can use the command line tool to import your ROMS:

```bash
ale-import-roms roms/
```

Then you should be able to make Atari envs by passing `gym_id`.

### References

- [1] S. Huang, R. F. J. Dossa, C. Ye, and J. Braga, “CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms.” arXiv, Nov. 16, 2021. Accessed: Nov. 21, 2022. [Online]. Available: http://arxiv.org/abs/2111.08819 Repo: [https://github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl)
- [2] Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann, “Stable-Baselines3: Reliable Reinforcement Learning Implementations,” Journal of Machine Learning Research, vol. 22, no. 268, pp. 1–8, 2021. Repo: [https://github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [3] W. Dabney, M. Rowland, M. G. Bellemare, and R. Munos, “Distributional Reinforcement Learning with Quantile Regression,” arXiv:1710.10044 [cs, stat], Oct. 2017, Accessed: Apr. 15, 2022. [Online]. Available: http://arxiv.org/abs/1710.10044
