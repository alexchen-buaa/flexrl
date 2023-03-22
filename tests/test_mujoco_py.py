import subprocess


def test_sac_continuous_action():
    subprocess.run(
        "python src/flexrl/algorithms/sac_continuous_action.py --gym_id Hopper-v2 --batch_size 128 --total_timesteps 135",
        shell=True,
        check=True,
    )