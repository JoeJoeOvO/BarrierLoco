"""Installation script for the 'locolab' python package."""

from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "psutil",
    "gymnasium",
    "onnx",
    "onnxruntime",
]

all_packages = ["locolab"] + find_packages(where=".")
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
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)
