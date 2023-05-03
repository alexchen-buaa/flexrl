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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter


@dataclass
class TrainArgs:
    # Experiment
    exp_name: str = "inac"
    gym_id: str = "halfcheetah-medium-expert-v2"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    log_dir: str = "runs"
    # InAC
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


def layer_init(layer, bias=True):
    torch.nn.init.xavier_uniform_(layer.weight)
    if bias:
        torch.nn.init.constant_(layer.bias.data, 0)
    return layer


class ValueNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 256))
        self.fc2 = layer_init(nn.Linear(256, 256))
        self.fc3 = layer_init(nn.Linear(256, 1))
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CriticNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256))
        self.fc2 = layer_init(nn.Linear(256, 256))
        self.fc3 = layer_init(nn.Linear(256, 1))
    
    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


EXP_ADV_MAX = 10000
LOG_STD_MAX = 0.0
LOG_STD_MIN = -6.0


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 256))
        self.fc2 = layer_init(nn.Linear(256, 256))
        self.fc_mean = layer_init(nn.Linear(256, np.prod(env.action_space.shape)))
        self.log_std = nn.Parameter(torch.zeros(np.prod(env.action_space.shape), dtype=torch.float32))
        self.register_buffer("max_action", torch.tensor(env.action_space.high, dtype=torch.float32))
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.sigmoid(self.log_std)
        log_std = LOG_STD_MIN + log_std * (LOG_STD_MAX - LOG_STD_MIN)
        return mean, log_std
    
    def get_action(self, x, a=None):
        mean, log_std = self(x)
        mean = torch.tanh(mean)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = mean if not self.training else dist.rsample()
        action = action if (a is None) else a
        log_prob = dist.log_prob(action).sum(axis=-1)
        log_prob -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1)
        return action, log_prob.reshape(-1, 1)


class Batch(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    masks: torch.Tensor
    next_observations: torch.Tensor


class Dataset:
    def __init__(self, device):
        self.device = device
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
        self.rewards = dataset["rewards"].astype(np.float32)
        self.masks = 1.0 - dataset["terminals"].astype(np.float32)
        self.next_observations = dataset["next_observations"].astype(np.float32)

    def to_torch(self, array):
        return torch.tensor(array, device=self.device)

    def sample(self, batch_size):
        idx = np.random.randint(self.size, size=batch_size)
        data = (
            self.observations[idx],
            self.actions[idx],
            self.rewards[idx],
            self.masks[idx],
            self.next_observations[idx],
        )
        return Batch(*tuple(map(self.to_torch, data)))


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
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Eval env setup
    env = make_env(args.gym_id, args.seed)()
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"
    
    # Agent setup
    actor = Actor(env).to(device)
    actor_beta = Actor(env).to(device)
    vf = ValueNetwork(env).to(device)
    qf1 = CriticNetwork(env).to(device)
    qf2 = CriticNetwork(env).to(device)
    qf1_target = CriticNetwork(env).to(device)
    qf2_target = CriticNetwork(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)
    actor_beta_optimizer = optim.Adam(list(actor_beta.parameters()), lr=args.learning_rate)
    v_optimizer = optim.Adam(list(vf.parameters()), lr=args.learning_rate)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    
    # Dataset setup
    dataset = Dataset(device)
    dataset.load(env)
    start_time = time.time()
    
    # Main loop
    for global_step in tqdm(range(args.total_iterations), desc="Training", unit="iter"):
        batch = dataset.sample(batch_size=args.batch_size)
        
        # Actor_beta update
        _, log_probs_beta = actor_beta.get_action(batch.observations, a=batch.actions)
        actor_beta_loss = -log_probs_beta.mean()
        
        actor_beta_optimizer.zero_grad()
        actor_beta_loss.backward()
        actor_beta_optimizer.step()
        
        # Value update
        old_val = vf(batch.observations)
        with torch.no_grad():
            actions, log_probs = actor.get_action(batch.observations)
            q1_target = qf1_target(batch.observations, actions)
            q2_target = qf2_target(batch.observations, actions)
            q_target = torch.minimum(q1_target, q2_target)
        v_target = q_target - args.tau * log_probs
        vf_loss = 0.5 * F.mse_loss(old_val, v_target)
        
        v_optimizer.zero_grad()
        vf_loss.backward()
        v_optimizer.step()
        
        # Critic update
        # Follow the official implementation (different from paper)
        with torch.no_grad():
            next_actions, next_log_probs = actor.get_action(batch.next_observations)
            next_q1_target = qf1_target(batch.next_observations, next_actions)
            next_q2_target = qf2_target(batch.next_observations, next_actions)
            next_q_target = torch.minimum(next_q1_target, next_q2_target).view(-1)
        td_target = batch.rewards + args.gamma * batch.masks * (next_q_target - args.tau * next_log_probs.view(-1))
        q1 = qf1(batch.observations, batch.actions).view(-1)
        q2 = qf2(batch.observations, batch.actions).view(-1)
        qf1_loss = 0.5 * F.mse_loss(q1, td_target)
        qf2_loss = 0.5 * F.mse_loss(q2, td_target)
        qf_loss = (qf1_loss + qf2_loss) * 0.5
        
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()
        
        # Actor update
        _, log_probs_ = actor.get_action(batch.observations, batch.actions)
        with torch.no_grad():
            q1_target = qf1_target(batch.observations, batch.actions)
            q2_target = qf2_target(batch.observations, batch.actions)
            q_target = torch.minimum(q1_target, q2_target)
            new_val = vf(batch.observations)
            _, log_prob_beta_ = actor_beta.get_action(batch.observations, batch.actions)
            adv = q_target - new_val
            exp_adv = torch.exp(adv / args.tau - log_prob_beta_).clamp(max=EXP_ADV_MAX)
        actor_loss = -(exp_adv * log_probs_).mean()
        
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        # Target network update
        for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
            target_param.data.copy_(args.polyak * param.data + (1 - args.polyak) * target_param.data)
        for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
            target_param.data.copy_(args.polyak * param.data + (1 - args.polyak) * target_param.data)
        
        # Evaluation
        if global_step % args.eval_freq == 0:
            env.seed(args.seed)
            stats = {"return": [], "length": []}
            actor.eval()
            for _ in range(args.eval_episodes):
                obs, done = env.reset(), False
                while not done:
                    action, _ = actor.get_action(torch.Tensor(obs).to(device).reshape(1, -1))
                    action = action.detach().cpu().numpy().flatten()
                    obs, reward, done, info = env.step(action)
                for k in stats.keys():
                    stats[k].append(info["episode"][k[0]])
            actor.train()
            for k, v in stats.items():
                writer.add_scalar(f"charts/episodic_{k}", np.mean(v), global_step)
                if k == "return":
                    normalized_score = env.get_normalized_score(np.mean(v)) * 100
                    writer.add_scalar("charts/normalized_score", normalized_score, global_step)
            writer.flush()

        # Logging
        if global_step % args.log_freq == 0:
            writer.add_scalar("losses/v", old_val.mean().item(), global_step)
            writer.add_scalar("losses/vf_loss", vf_loss.item(), global_step)
            writer.add_scalar("losses/q1", q1.mean().item(), global_step)
            writer.add_scalar("losses/q2", q2.mean().item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
            writer.add_scalar("losses/adv", adv.mean().item(), global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.flush()

    env.close()
    writer.close()