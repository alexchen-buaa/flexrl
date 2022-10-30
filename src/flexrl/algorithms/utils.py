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
    "num_envs": 4,  # The number of parallel game environments
    "num_steps": 128,  # The number of steps to run in each environment per policy rollout
    "anneal_lr": True,  # Toggle learning rate annealing for policy and value networks
    "gae": True,  # Use GAE for advantage computation
    "gamma": 0.99,  # The discount factor gamma
    "gae_lambda": 0.95,  # The lambda for the general advantage estimation
    "num_minibatches": 4,  # The number of mini-batches
    "update_epochs": 4,  # The K epochs to update the policy
    "norm_adv": True,  # Toggles advantage normalization
    "clip_coef": 0.2,  # The surrogate clipping coefficient
    "ent_coef": 0.01,  # Coefficient of the entropy
    "vf_coef": 0.5,  # Coefficient of the value function
    "max_grad_norm": 0.5,  # The maximum norm for the gradient clipping
    "target_kl": None,  # The target KL divergence threshold
    "async_envs": False, # (NEW) Use AsyncVectorEnv wrapper
}

DEFAULT_ARGS = {
    "ppo": BASE_ARGS,
    "ppo_multidiscrete": BASE_ARGS,
    "ppo_sil": BASE_ARGS.update(
        {
            "buffer_size": int(1e6),  # The size of the replay buffer
        }
    ),
}


def update_args(_args, algorithm="ppo"):
    args = DEFAULT_ARGS[algorithm]
    args.update(_args)  # Update args
    args = argparse.Namespace(**args)  # Convert to argparse.Namespace
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args