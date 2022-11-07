from flexrl.algorithms import PPO, DQN

args = {
    "exp_name": "test_classic_control",
    "gym_id": "CartPole-v1",
    "total_timesteps": 256,
}

print("Testing PPO...")
PPO(args)
print("Testing DQN...")
DQN(args)
