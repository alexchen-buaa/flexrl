from setuptools import setup

requires = [
    "torch==1.13.0",
    "stable-baselines3==1.1.0",
    "tensorboardX==2.6",
    "opencv-python==4.6.0.66",
    "gym[mujoco_py, classic_control]==0.23.1",
    "jax==0.4.6",
    "flax==0.6.7",
    "optax==0.1.4",
    "tensorflow-probability==0.19.0",
    "ale-py==0.7.4",
    "mujoco-py==2.1.2.14",
    "tqdm==4.64.0",
    "protobuf==3.19.4",
    "pyrallis==0.3.1",
    "pytest==7.2.2",
    "d4rl @ git+https://github.com/alexchen-buaa/d4rl.git",
]

setup(
    name="flexrl",
    version="0.0.1",
    description="Non-modular implementation of common RL algorithms",
    author="alexchen-buaa",
    author_email="975106676@qq.com",
    project_urls={
        "Source": "https://github.com/alexchen-buaa/flexrl",
    },
    python_requires=">=3.8, <4",
    install_requires=requires,
)
