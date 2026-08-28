from setuptools import find_packages, setup


setup(
    name="advancewars",
    version="0.1.0",
    description="Advance Wars-style simulator with a PettingZoo-style API.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    package_data={"advancewars": ["data/**/*.yaml", "data/**/*.map"]},
    python_requires=">=3.10",
    install_requires=["numpy>=1.21", "PyYAML>=5.4"],
)
