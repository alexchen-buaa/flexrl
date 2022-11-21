from flexrl.algorithms import PPO, DQN, QR_DQN

args = {
    "exp_name": "test_classic_control",
    "gym_id": "CartPole-v1",
    "total_timesteps": 512,
    "learning_starts": 256,
}

print("Testing PPO...")
PPO(args)
print("Testing DQN...")
DQN(args)
print("Testing QR_DQN...")
QR_DQN(args)
