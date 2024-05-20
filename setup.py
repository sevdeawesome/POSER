import setuptools
import os
from io import open
from setuptools.command.install import install
from setuptools import find_packages
# from download_data import download_data

src_dir = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()

# Build requirements
requirements_path = f"{src_dir}/requirements.txt"
requirements = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        requirements = f.read().splitlines()

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        print("RUNNING POST INSTALL 1")
        install.run(self)
        print("RUNNING POST INSTALL")
        # download_data()


setuptools.setup(
    name="genies-benchmark",
    version="0.0.1",
    author="Joshua Clymer, Garrett Baker, Rohan Subramani, and Sam Wang",
    author_email="joshuamclymer@gmail.com",
    description="The fig benchmark repository contains datasets and tooling for evaluating the generalization of preferrence models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Joshuaclymer/GENIES",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    cmdclass={
        'install': PostInstallCommand,
    },
    python_requires='>=3.10',
    install_requires=requirements,
    packages=find_packages(where='src'),  # Specify 'src' as the root
    package_dir={'': 'src'},
    package_data={'genies-benchmark': ['LICENCE', 'requirements.txt']},
)