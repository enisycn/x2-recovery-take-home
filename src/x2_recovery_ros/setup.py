from glob import glob
from setuptools import find_packages, setup


package_name = "x2_recovery_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/artifacts", glob("artifacts/*")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Candidate",
    maintainer_email="candidate@example.com",
    description="AgiBot X2 ground-recovery baseline and ROS 2 interface.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "x2_train = x2_recovery_ros.train:main",
            "x2_evaluate = x2_recovery_ros.evaluate:main",
            "x2_recovery_node = x2_recovery_ros.recovery_node:main",
            "x2_telemetry_node = x2_recovery_ros.telemetry_node:main",
        ],
    },
)

