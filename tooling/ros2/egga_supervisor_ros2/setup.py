from setuptools import find_packages, setup

package_name = "egga_supervisor_ros2"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="EGGA Safety Engineering",
    maintainer_email="safety@egga-control.org",
    description="ROS 2 Runtime Safety Cage and Envelope Guard for Autonomous Vehicles",
    license="Proprietary / Automotive Enterprise License",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "supervisor_node = egga_supervisor_ros2.supervisor_node:main",
        ],
    },
)
