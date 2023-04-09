# source: https://github.com/hwang-ua/inac_pytorch
# https://arxiv.org/abs/2302.14372

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
from jax.nn import softplus
import optax
from flax.training.train_state import TrainState
from tensorflow_probability.substrates import jax as tfp
from tensorboardX import SummaryWriter
tfd = tfp.distributions
tfb = tfp.bijectors


@dataclass
class TrainArgs:
    # Experiment
    exp_name: str = "inac_jax"
    gym_id: str = "halfcheetah-medium-expert-v2"
    seed: int = 1
    log_dir: str = "runs"
    # IQL
    total_iterations: int = int(1e6)
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 256
    tau: float = 0.1
    polyak: float = 0.005
    eval_freq: int = int(5e3)
    eval_episodes: int = 10
    log_freq: int = 1000

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


def layer_init():
    # Zero initialization for bias by default in flax
    return nn.initializers.xavier_uniform()


class ValueNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(256, kernel_init=layer_init())(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=layer_init())(x)
        x = nn.relu(x)
        x = nn.Dense(1, kernel_init=layer_init())(x)
        return x


class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(256, kernel_init=layer_init())(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=layer_init())(x)
        x = nn.relu(x)
        x = nn.Dense(1, kernel_init=layer_init())(x)
        return x


class DoubleCriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        critic1 = CriticNetwork()(x, a)
        critic2 = CriticNetwork()(x, a)
        return critic1, critic2


EXP_ADV_MAX = 10000
LOG_STD_MAX = 0.0
LOG_STD_MIN = -6.0


class Actor(nn.Module):
    action_dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, temperature: float = 1.0):
        x = nn.Dense(256, kernel_init=layer_init())(x)
        x = nn.relu(x)
        x = nn.Dense(256, kernel_init=layer_init())(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim, kernel_init=layer_init())(x)
        mean = nn.tanh(mean)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim, ))
        log_std = nn.sigmoid(log_std)
        log_std = LOG_STD_MIN + log_std * (LOG_STD_MAX - LOG_STD_MIN)
        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std) * temperature)
        return dist


class TargetTrainState(TrainState):
    target_params: flax.core.FrozenDict


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
    key, actor_key, behav_key, critic_key, value_key = jax.random.split(key, 5)

    # Eval env setup
    env = make_env(args.gym_id, args.seed)()
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"
    observation = env.observation_space.sample()[np.newaxis]
    action = env.action_space.sample()[np.newaxis]
    
    # Agent setup
    actor = Actor(action_dim=np.prod(env.action_space.shape))
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, observation),
        tx=optax.adam(learning_rate=args.learning_rate)
    )
    # Behavior policy is just another set of trainable parameters
    behav_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(behav_key, observation),
        tx=optax.adam(learning_rate=args.learning_rate)
    )
    vf = ValueNetwork()
    vf_state = TrainState.create(
        apply_fn=vf.apply,
        params=vf.init(value_key, observation),
        tx=optax.adam(learning_rate=args.learning_rate)
    )
    qf = DoubleCriticNetwork()
    qf_state = TargetTrainState.create(
        apply_fn=qf.apply,
        params=qf.init(critic_key, observation, action),
        target_params=qf.init(critic_key, observation, action),
        tx=optax.adam(learning_rate=args.learning_rate)
    )
    
    # Dataset setup
    dataset = Dataset()
    dataset.load(env)
    start_time = time.time()
    
    def get_log_prob(params, observation, action):
        dist = actor.apply(params, observation)
        log_prob = dist.log_prob(action)
        log_prob -= (2*(np.log(2) - action - softplus(-2*action))).sum(axis=1)
        return log_prob.reshape(-1, 1)
    
    def get_action_log_prob(rng, params, observation):
        dist = actor.apply(params, observation)
        rng, key = jax.random.split(rng)
        action = dist.sample(seed=key)
        log_prob = dist.log_prob(action)
        log_prob -= (2*(np.log(2) - action - softplus(-2*action))).sum(axis=1)
        return rng, action, log_prob.reshape(-1, 1)
    
    def update_behav(behav_state, batch):
        def behav_loss_fn(params):
            log_probs = get_log_prob(params, batch.observations, batch.actions)
            behav_loss = -log_probs.mean()
            return behav_loss, {
                "behav_loss": behav_loss,
            }
        
        (behav_loss, info), grads = jax.value_and_grad(behav_loss_fn, has_aux=True)(behav_state.params)
        behav_state = behav_state.apply_gradients(grads=grads)
        return behav_state, info
    
    def update_vf(rng, actor_state, vf_state, qf_state, batch):
        rng, actions, log_probs = get_action_log_prob(rng, actor_state.params, batch.observations)
        q1, q2 = qf.apply(qf_state.target_params, batch.observations, actions)
        q = jnp.minimum(q1, q2)
        
        def vf_loss_fn(params):
            v = vf.apply(params, batch.observations)
            v_target = q - args.tau * log_probs
            vf_loss = 0.5 * ((v - v_target)**2).mean()
            return vf_loss, {
                "vf_loss": vf_loss,
                "v": v.mean(),
            }
        
        (vf_loss, info), grads = jax.value_and_grad(vf_loss_fn, has_aux=True)(vf_state.params)
        vf_state = vf_state.apply_gradients(grads=grads)
        return rng, vf_state, info
    
    def update_qf(rng, actor_state, vf_state, qf_state, batch):
        rng, next_actions, next_log_probs = get_action_log_prob(rng, actor_state.params, batch.next_observations)
        next_q1, next_q2 = qf.apply(qf_state.target_params, batch.next_observations, next_actions)
        next_q = jnp.minimum(next_q1, next_q2)
        td_target = batch.rewards + args.gamma * batch.masks * (next_q - args.tau * next_log_probs)
        
        def qf_loss_fn(params):
            q1, q2 = qf.apply(params, batch.observations, batch.actions)
            qf1_loss = 0.5 * ((q1 - td_target)**2).mean()
            qf2_loss = 0.5 * ((q2 - td_target)**2).mean()
            qf_loss = (qf1_loss + qf2_loss) * 0.5
            return qf_loss, {
                "qf_loss": qf_loss,
                "q1": q1.mean(),
                "q2": q2.mean(),
            }
        
        (qf_loss, info), grads = jax.value_and_grad(qf_loss_fn, has_aux=True)(qf_state.params)
        qf_state = qf_state.apply_gradients(grads=grads)
        return rng, qf_state, info
        
    def update_actor(actor_state, behav_state, vf_state, qf_state, batch):
        q1, q2 = qf.apply(qf_state.target_params, batch.observations, batch.actions)
        q = jnp.minimum(q1, q2)
        v = vf.apply(vf_state.params, batch.observations)
        log_probs_b = get_log_prob(behav_state.params, batch.observations, batch.actions)
        exp_adv = jnp.clip(jnp.exp((q - v) / args.tau - log_probs_b), a_max=EXP_ADV_MAX)
        
        def actor_loss_fn(params):
            log_probs = get_log_prob(params, batch.observations, batch.actions)
            actor_loss = -(exp_adv * log_probs).mean()
            return actor_loss, {
                "actor_loss": actor_loss,
                "adv": q - v,
            }
        
        (actor_loss, info), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        return actor_state, info
            
    def update_target(qf_state):
        new_target_params = jax.tree_map(
            lambda p, tp: p * args.polyak + tp * (1 - args.polyak), qf_state.params,
            qf_state.target_params)
        return qf_state.replace(target_params=new_target_params)
    
    @jax.jit
    def update(rng, actor_state, behav_state, vf_state, qf_state, batch):
        behav_state, behav_info = update_behav(behav_state, batch)
        rng, vf_state, vf_info = update_vf(rng, actor_state, vf_state, qf_state, batch)
        rng, qf_state, qf_info = update_qf(rng, actor_state, vf_state, qf_state, batch)
        actor_state, actor_info = update_actor(actor_state, behav_state, vf_state, qf_state, batch)
        qf_state = update_target(qf_state)
        return rng, actor_state, behav_state, vf_state, qf_state, {
            **behav_info, **vf_info, **qf_info, **actor_info
        }
    
    @jax.jit
    def get_action(rng, actor_state, observation, temperature=1.0):
        dist = actor.apply(actor_state.params, observation, temperature)
        rng, key = jax.random.split(rng)
        action = dist.sample(seed=key)
        return rng, jnp.clip(action, -1, 1)
    
    # Main loop
    for global_step in tqdm(range(args.total_iterations), desc="Training", unit="iter"):
        
        # Batch update
        batch = dataset.sample(batch_size=args.batch_size)
        key, actor_state, behav_state, vf_state, qf_state, update_info = update(
            key, actor_state, behav_state, vf_state, qf_state, batch
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
            writer.flush()

        # Logging
        if global_step % args.log_freq == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    writer.add_scalar(f"losses/{k}", v, global_step)
                else:
                    writer.add_histogram(f"losses/{k}", v, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.flush()

    env.close()
    writer.close()