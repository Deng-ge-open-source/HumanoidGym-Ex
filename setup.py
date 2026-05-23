from setuptools import find_packages, setup


setup(
    name="humanoid-gym-ex",
    version="0.1.0",
    author="HumanoidGym-Ex contributors",
    license="BSD-3-Clause",
    packages=find_packages(),
    description="Humanoid-Gym-style extension framework for humanoid robot reinforcement learning",
    install_requires=[
        "isaacgym",
        "wandb",
        "DateTime",
        "tensorboard",
        "tqdm",
        "numpy==1.23.5",
        "opencv-python",
        "mujoco==2.3.6",
        "mujoco-python-viewer",
        "matplotlib",
    ],
)
