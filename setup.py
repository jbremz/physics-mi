from setuptools import find_packages, setup

setup(
    name="physics_mi",
    version="0.1.0",
    description="exploring emergent physical models in neural networks from a mechanistic interpretability perspective",
    author="Jim Bremner",
    url="https://github.com/yourusername/mechanistic-interpretability",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
