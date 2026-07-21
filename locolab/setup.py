"""Installation script for the 'locolab' python package."""

from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "psutil",
    "gymnasium",
    "onnx",
    "onnxruntime",
]

# ``setup.py`` lives inside the ``locolab`` package directory.  Keep the
# vendored RL library importable as ``rsl_rl`` while installing the training
# stack under its canonical ``locolab.source`` namespace.
discovered_packages = find_packages(where=".")
all_packages = ["locolab"]
all_packages.extend(
    f"locolab.{package}" if package == "source" or package.startswith("source.") else package
    for package in discovered_packages
)

PACKAGE_DATA = {
    "locolab.source.locomotion.assets": [
        "Go2W/usd/*.usd",
        "Go2W/usd/configuration/*.usd",
    ],
    "locolab.source.locomotion.mdp": ["reference_motions/*.json"],
}

setup(
    name="locolab",
    packages=all_packages,
    package_dir={"locolab": ".", "": "."},
    author="OmniBot Team",
    url="https://github.com/ssh-robotics/OmniBot",
    version="0.1.0",
    description="LocoLab - Locomotion Training Lab for OmniBot",
    keywords=["Isaac", "Legged", "Locomotion", "Wheel-Legged Robot"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    package_data=PACKAGE_DATA,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)
