from setuptools import find_packages, setup

setup(
    name="aetos_multiam",
    version="0.1.0",
    description="A benchmark for multi aerial-manipulator (AM) collaboration, built on Isaac Sim.",
    packages=find_packages("."),
    install_requires=[
        "torch",
        "torchrl==0.3.1",
        "tensordict",
        "hydra-core",
        "omegaconf",
        # `omni_drones` is a sibling repo (/home/benhua/OmniDrones), not on
        # PyPI -- install it with `pip install -e /home/benhua/OmniDrones`
        # (or add it to PYTHONPATH) before using this package. See
        # docs/ARCHITECTURE.md's "Dependency on OmniDrones" section.
    ],
)
