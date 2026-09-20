from setuptools import find_packages, setup

setup(
    name="routemind-ai",
    version="1.0.0",
    description="Predictive Delivery Disruption & Recovery System",
    packages=find_packages(include=["src", "src.*", "api", "api.*"]),
    python_requires=">=3.11",
)
