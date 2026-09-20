"""Packaging metadata for the external Isaac Lab task."""

from setuptools import find_packages, setup


setup(
    name="x2_recovery_isaac",
    version="0.1.0",
    description="External Isaac Lab task for AgiBot X2 ground recovery",
    packages=find_packages(),
    python_requires=">=3.12",
    zip_safe=False,
)
