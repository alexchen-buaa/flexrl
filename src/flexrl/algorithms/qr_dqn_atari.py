import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from flexrl.algorithms.utils import update_args


def make_env(env_id, seed, idx, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


class QNetwork(nn.Module):
    def __init__(self, env, num_quants):
        super().__init__()
        self.num_actions = env.single_action_space.n
        self.num_quants = num_quants
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions * self.num_quants),
        )

    def forward(self, x):
        batch_size = x.size(0)
        return self.network(x / 255.0).view(batch_size, self.num_actions, self.num_quants)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def QR_DQN_Atari(_args):
    # Logging setup
    args = update_args(_args, algorithm="qr_dqn_atari")
    print(vars(args))
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
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

    # Env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id, args.seed, 0, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Model/agent setup
    q_network = QNetwork(envs, args.num_quants).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs, args.num_quants).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # Storage setup
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # Main loop
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # Action logic
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # Take mean across quantiles to get q_values
            q_values = q_network(torch.Tensor(obs).to(device)).mean(dim=2)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Step the environment
        next_obs, rewards, dones, infos = envs.step(actions)

        # Logging
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                break

        # Save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # CRUCIAL step easy to overlook
        obs = next_obs

        # Training
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            # Perform quantile regression
            with torch.no_grad():
                target = target_network(data.next_observations)  # Raw quantiles
                actions_max = target.mean(dim=2, keepdim=True).argmax(dim=1, keepdim=True)
                actions_max = actions_max.expand(args.batch_size, 1, args.num_quants)
                target_max = target.gather(1, actions_max).squeeze(dim=1)
                td_target = data.rewards + args.gamma * target_max * (1 - data.dones)
                td_target = td_target.unsqueeze(1)  # (batch_size, 1, num_quants)
            actions = data.actions[..., None].long().expand(args.batch_size, 1, args.num_quants)
            old_val = q_network(data.observations).gather(1, actions).squeeze(dim=1)
            old_val = old_val.unsqueeze(2)  # (batch_size, num_quants, 1)
            # Quantile Huber loss
            u = td_target - old_val  # (batch_size, num_quants, num_quants)
            tau = ((torch.arange(args.num_quants, device=old_val.device, dtype=torch.float) + 0.5) / args.num_quants).view(1, -1, 1)
            weight = torch.abs(tau - u.le(0.).float())
            loss = F.smooth_l1_loss(old_val, td_target, reduction="none")
            loss = (weight * loss).mean()

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    envs.close()
    writer.close()
    return q_network
