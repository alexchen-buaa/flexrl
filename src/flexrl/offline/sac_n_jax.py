# source: https://github.com/snu-mllab/EDAC
# source: https://github.com/tinkoff-ai/CORL
# https://arxiv.org/abs/2110.01548

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union
from dataclasses import dataclass
import random
import time

import d4rl
import gym
import numpy as np
import pyrallis
from tqdm import tqdm
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from tensorflow_probability.substrates import jax as tfp
from tensorboardX import SummaryWriter
tfd = tfp.distributions
tfb = tfp.bijectors


@dataclass
class TrainArgs:
    # Experiment
    exp_name: str = "sac_n_jax"
    gym_id: str = "halfcheetah-medium-expert-v2"
    seed: int = 1
    log_dir: str = "runs"
    # SAC-N
    total_iterations: int = int(1e6)
    gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    alpha: float = 0.2
    polyak: float = 0.005
    eval_freq: int = int(5e3)
    eval_episodes: int = 10
    log_freq: int = 1000
    ensemble_size: int = 20

    def __post_init__(self):
        self.exp_name = f"{self.exp_name}__{self.gym_id}"


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def zero_centered_uniform(bound=1e-3):
    def init(key, shape, dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        return jax.random.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)
    return init
        

class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(256, bias_init=nn.initializers.constant(0.1))(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256, bias_init=nn.initializers.constant(0.1))(x)
        x = nn.relu(x)
        x = nn.Dense(1, kernel_init=zero_centered_uniform(3e-3), bias_init=zero_centered_uniform(3e-3))(x)
        return x


# EDAC paper
LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0


class Actor(nn.Module):
    action_dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, temperature: float = 1.0):
        x = nn.Dense(256, bias_init=nn.initializers.constant(0.1))(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256, bias_init=nn.initializers.constant(0.1))(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim, kernel_init=zero_centered_uniform(1e-3),
                        bias_init=zero_centered_uniform(1e-3))(x)
        log_std = nn.Dense(self.action_dim, kernel_init=zero_centered_uniform(1e-3),
                           bias_init=zero_centered_uniform(1e-3))(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        base_dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std) * temperature)
        dist = tfd.TransformedDistribution(distribution=base_dist, bijector=tfb.Tanh())
        return dist


class Batch(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    masks: np.ndarray
    next_observations: np.ndarray


class Dataset:
    def __init__(self):
        self.size = None
        self.observations = None
        self.actions = None
        self.rewards = None
        self.masks = None
        self.next_observations = None

    def load(self, env, eps=1e-5):
        self.env = env
        dataset = d4rl.qlearning_dataset(env)
        lim = 1 - eps # Clip to eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)
        self.size = len(dataset["observations"])
        self.observations = dataset["observations"].astype(np.float32)
        self.actions = dataset["actions"].astype(np.float32)
        self.rewards = dataset["rewards"].astype(np.float32)[:, None]
        self.masks = 1.0 - dataset["terminals"].astype(np.float32)[:, None]
        self.next_observations = dataset["next_observations"].astype(np.float32)

    def sample(self, batch_size):
        idx = np.random.randint(self.size, size=batch_size)
        data = (
            self.observations[idx],
            self.actions[idx],
            self.rewards[idx],
            self.masks[idx],
            self.next_observations[idx],
        )
        return Batch(*data)


if __name__ == "__main__":
    # Logging setup
    args = pyrallis.parse(config_class=TrainArgs)
    print(vars(args))
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"{args.log_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, critic_key = jax.random.split(key, 3)
    
    # Eval env setup
    env = make_env(args.gym_id, args.seed)()
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"
    observation = env.observation_space.sample()[np.newaxis]
    action = env.action_space.sample()[np.newaxis]
    action_dim = np.prod(env.action_space.shape)
    
    # Actor setup
    actor = Actor(action_dim=action_dim)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, observation),
        tx=optax.adam(learning_rate=args.actor_lr)
    )
    
    # Critic (ensemble) setup
    qf = CriticNetwork()
    
    def qf_init_fn(key):
        return TrainState.create(
            apply_fn=qf.apply,
            params=qf.init(key, observation, action),
            tx=optax.adam(learning_rate=args.critic_lr),
        )
    
    def qf_predict_fn(qf_state, observation, action):
        return qf.apply(qf_state.params, observation, action)
    
    critic_keys = jax.random.split(critic_key, args.ensemble_size)
    parallel_qf_init_fn = jax.vmap(qf_init_fn)
    parallel_qf_predict_fn = jax.vmap(qf_predict_fn, in_axes=(0, None, None))
    qf_states = parallel_qf_init_fn(critic_keys)
    target_qf_states = parallel_qf_init_fn(critic_keys) # Same key, same parameters
    
    # Adaptive alpha setup
    target_entropy = -float(action_dim)
    log_alpha = jnp.array([0.0], dtype=jnp.float32)
    alpha_optimizer = optax.adam(learning_rate=args.alpha_lr)
    alpha_opt_state = alpha_optimizer.init(log_alpha)

    # Dataset setup
    dataset = Dataset()
    dataset.load(env)
    start_time = time.time()
    
    def get_action_log_prob(key, params, observations):
        dist = actor.apply(params, observations)
        actions = dist.sample(seed=key)
        log_prob = dist.log_prob(actions)
        return actions, log_prob.reshape(-1, 1)
    
    def update_alpha(rng, actor_state, log_alpha, alpha_opt_state, batch):
        rng, key = jax.random.split(rng)
        actions, log_prob = get_action_log_prob(key, actor_state.params, batch.observations)

        def alpha_loss_fn(params):
            alpha_loss = (-params * (log_prob + target_entropy)).mean()
            return alpha_loss, {
                "alpha_loss": alpha_loss,
            }

        (alpha_loss, info), grads = jax.value_and_grad(alpha_loss_fn, has_aux=True)(log_alpha)
        updates, alpha_opt_state = alpha_optimizer.update(grads, alpha_opt_state, log_alpha)
        log_alpha = optax.apply_updates(log_alpha, updates)
        return rng, log_alpha, alpha_opt_state, info
    
    def update_qf(key, actor_state, qf_state, target_qf_states, log_alpha, batch):
        next_actions, next_log_prob = get_action_log_prob(
            key, actor_state.params, batch.next_observations)
        next_qs = parallel_qf_predict_fn(target_qf_states, batch.next_observations, next_actions)
        next_q = next_qs.min(axis=0) - jnp.exp(log_alpha) * next_log_prob
        target_q = batch.rewards + args.gamma * batch.masks * next_q
        
        def qf_loss_fn(params):
            q = qf.apply(params, batch.observations, batch.actions)
            qf_loss = ((q - target_q)**2).mean()
            return qf_loss, {
                "qf_loss": qf_loss,
                "q": q.mean(),
            }
        
        (qf_loss, info), grads = jax.value_and_grad(qf_loss_fn, has_aux=True)(qf_state.params)
        qf_state = qf_state.apply_gradients(grads=grads)
        return qf_state, info
    
    parallel_update_qf = jax.vmap(update_qf, in_axes=(0, None, 0, None, None, None))
    
    def update_actor(rng, actor_state, qf_states, log_alpha, batch):
        rng, key = jax.random.split(rng)
        
        def actor_loss_fn(params):
            actions, log_prob = get_action_log_prob(key, params, batch.observations)
            qs = parallel_qf_predict_fn(qf_states, batch.observations, actions)
            q = qs.min(axis=0)
            actor_loss = (jnp.exp(log_alpha) * log_prob - q).mean()
            return actor_loss, {
                "actor_loss": actor_loss,
                "entropy": -log_prob.mean(),
                "q_policy_std": jnp.std(qs, axis=0),
            }
        
        (actor_loss, info), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        return rng, actor_state, info
    
    def update_target(qf_state, target_qf_state):
        new_target_params = jax.tree_map(
            lambda p, tp: p * args.polyak + tp * (1 - args.polyak), qf_state.params,
            target_qf_state.params)
        return target_qf_state.replace(params=new_target_params)
    
    parallel_update_target = jax.vmap(update_target, in_axes=(0, 0))
    
    @jax.jit
    def update(rng, actor_state, qf_states, target_qf_states, log_alpha, alpha_opt_state, batch):
        rng, log_alpha, alpha_opt_state, alpha_info = update_alpha(
            rng, actor_state, log_alpha, alpha_opt_state, batch)
        rng, actor_state, actor_info = update_actor(
            rng, actor_state, qf_states, log_alpha, batch)
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, args.ensemble_size)
        qf_states, qf_info = parallel_update_qf(
            keys, actor_state, qf_states, target_qf_states, log_alpha, batch)
        target_qf_states = parallel_update_target(qf_states, target_qf_states)
        return rng, actor_state, qf_states, target_qf_states, log_alpha, alpha_opt_state, {
            **alpha_info, **actor_info, **qf_info,
        }
    
    @jax.jit
    def get_action(rng, actor_state, observation, temperature=1.0):
        dist = actor.apply(actor_state.params, observation, temperature)
        rng, key = jax.random.split(rng)
        action = dist.sample(seed=key)
        return rng, jnp.clip(action, -1, 1)
    
    # Main loop
    pbar = tqdm(range(args.total_iterations), unit="iter")
    for global_step in pbar:
        
        # Batch update
        batch = dataset.sample(batch_size=args.batch_size)
        key, actor_state, qf_states, target_qf_states, log_alpha, alpha_opt_state, update_info = update(
            key, actor_state, qf_states, target_qf_states, log_alpha, alpha_opt_state, batch
        )
        
        # Evaluation
        if global_step % args.eval_freq == 0:
            env.seed(args.seed)
            stats = {"return": [], "length": []}
            for _ in range(args.eval_episodes):
                obs, done = env.reset(), False
                while not done:
                    key, action = get_action(key, actor_state, obs, temperature=0.0)
                    action = np.asarray(action)
                    obs, reward, done, info = env.step(action)
                for k in stats.keys():
                    stats[k].append(info["episode"][k[0]])
            for k, v in stats.items():
                writer.add_scalar(f"charts/episodic_{k}", np.mean(v), global_step)
                if k == "return":
                    normalized_score = env.get_normalized_score(np.mean(v)) * 100
                    writer.add_scalar("charts/normalized_score", normalized_score, global_step)
                    pbar.set_description("score: {:.2f}".format(normalized_score))
            writer.flush()
        
        # Logging
        if global_step % args.log_freq == 0:
            batched = ["q", "qf_loss"]
            for k, v in update_info.items():
                if v.ndim == 0:
                    writer.add_scalar(f"losses/{k}", v, global_step)
                else:
                    if k in batched:
                        writer.add_scalar(f"losses/{k}_mean", v.mean(), global_step)
                        writer.add_scalar(f"losses/{k}_std", jnp.std(v), global_step)
                    else:
                        writer.add_histogram(f"losses/{k}", v, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.flush()

    env.close()
    writer.close()