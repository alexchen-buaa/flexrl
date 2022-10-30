from setuptools import setup

requires = [
    "torch==1.7.1",
    "stable-baselines3==1.1.0",
    "tensorboard==2.5.0",
    "opencv-python==4.6.0.66",
    "gym==0.21.0",
    "ale-py==0.7.4",
    "protobuf==3.19.4",
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
