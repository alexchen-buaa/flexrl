import subprocess


def test_sac_continuous_action():
    subprocess.run(
        "python src/flexrl/online/sac.py --gym_id Hopper-v2 --batch_size 128 --total_timesteps 135",
        shell=True,
        check=True,
    )