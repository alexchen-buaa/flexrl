from flexrl.algorithms import PPO_Atari, DQN_Atari

args = {
    "exp_name": "test_atari",
    "gym_id": "PongNoFrameskip-v4",
    "total_timesteps": 256,
}

print("Testing PPO_Atari...")
PPO_Atari(args)
print("Testing DQN_Atari...")
DQN_Atari(args)
