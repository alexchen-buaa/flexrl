import subprocess


def test_ppo():
    subprocess.run(
        "python src/flexrl/algorithms/ppo.py --num_envs 1 --num_steps 64 --total_timesteps 256",
        shell=True,
        check=True,
    )


def test_dqn():
    subprocess.run(
        "python src/flexrl/algorithms/dqn.py --learning_starts 200 --total_timesteps 205",
        shell=True,
        check=True,
    )


def test_qr_dqn():
    subprocess.run(
        "python src/flexrl/algorithms/qr_dqn.py --learning_starts 200 --total_timesteps 205",
        shell=True,
        check=True,
    )
