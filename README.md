# FlexRL

FlexRL is a fork of [CleanRL](https://github.com/vwxyzjn/cleanrl) that offers a bit more flexibility while still being non-modular. FlexRL introduces the following features:
- Notebook-friendly interface
- A few custom environments

> **NOTE**: This repo is mainly for research and educational purposes. For now we only have PPO and we may gradually include more algorithms and custom environments.

## Quick Start

### Installing FlexRL

```bash
git clone https://github.com/alexchen-buaa/flexrl.git
cd flexrl
pip install -e .
```

### Usage

```python
from flexrl.algorithms import PPO

args = {
    "exp_name": "test",
	"gym_id": "CartPole-v1",
	"total_timesteps": 20000,
}

PPO(args)
```

### Atari Support

For Atari support, `flexrl` has `ale-py` as a dependency. So according to [The Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment), you can use the command line tool to import your ROMS:

```bash
ale-import-roms roms/
```

Then you should be able to make Atari envs by passing `gym_id`.
