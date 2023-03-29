import subprocess


def test_iql_d4rl():
    subprocess.run(
        "python src/flexrl/offline/iql.py --gym_id halfcheetah-medium-expert-v2 --total_iterations 200 --eval_freq 100 --eval_episodes 1 --log_freq 100",
        shell=True,
        check=True,
    )