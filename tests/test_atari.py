import subprocess


def test_ppo_atari():
    subprocess.run(
        "python src/flexrl/online/ppo_atari.py --num_envs 1 --num_steps 64 --total_timesteps 256",
        shell=True,
        check=True,
    )


def test_dqn_atari():
    subprocess.run(
        "python src/flexrl/online/dqn_atari.py --learning_starts 10 --total_timesteps 16 --buffer_size 10 --batch_size 4",
        shell=True,
        check=True,
    )


def test_qr_dqn_atari():
    subprocess.run(
        "python src/flexrl/online/qr_dqn_atari.py --learning_starts 10 --total_timesteps 16 --buffer_size 10 --batch_size 4",
        shell=True,
        check=True,
    )
