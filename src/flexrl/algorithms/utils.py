import argparse
import os


BASE_ARGS = {
    "exp_name": os.path.basename(__file__).rstrip(".py"),  # The name of this experiment
    "gym_id": "CartPole-v1",  # The id of the gym environment
    "learning_rate": 2.5e-4,  # The learning rate of the optimizer
    "seed": 1,  # Seed of the experiment
    "total_timesteps": 25000,  # Total timesteps of the experiment
    "torch_deterministic": True,  # If toggled, `torch.backends.cudnn.deterministic=False`
    "cuda": True,  # If toggled, cuda will be enabled by default
    "gamma": 0.99,  # The discount factor gamma
}

UPDATE_ARGS = {
    "ppo": {
        "num_envs": 4,
        "num_steps": 128,
        "anneal_lr": True,
        "gae": True,
        "gae_lambda": 0.95,
        "num_minibatches": 4,
        "update_epochs": 4,
        "norm_adv": True,
        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "async_envs": False,
    },
    "ppo_multidiscrete": {
        "num_envs": 4,
        "num_steps": 128,
        "anneal_lr": True,
        "gae": True,
        "gae_lambda": 0.95,
        "num_minibatches": 4,
        "update_epochs": 4,
        "norm_adv": True,
        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "async_envs": False,
    },
    "ppo_atari": {
        "num_envs": 4,
        "num_steps": 128,
        "anneal_lr": True,
        "gae": True,
        "gae_lambda": 0.95,
        "num_minibatches": 4,
        "update_epochs": 4,
        "norm_adv": True,
        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "async_envs": False,
    },
    "dqn": {
        "learning_rate": 2.5e-4,
        "buffer_size": int(1e5),
        "target_network_frequency": 500,
        "batch_size": 128,
        "start_e": 1,
        "end_e": 0.05,
        "exploration_fraction": 0.5,
        "learning_starts": 10000,
        "train_frequency": 10,
    },
    "dqn_atari": {
        "learning_rate": 1e-4,
        "buffer_size": int(1e6),
        "target_network_frequency": 1000,
        "batch_size": 32,
        "start_e": 1,
        "end_e": 0.01,
        "exploration_fraction": 0.10,
        "learning_starts": 80000,
        "train_frequency": 4,
    },
}


def update_args(_args, algorithm="ppo"):
    args = BASE_ARGS
    args.update(UPDATE_ARGS[algorithm])
    args.update(_args)  # Update args
    args = argparse.Namespace(**args)  # Convert to argparse.Namespace
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args
