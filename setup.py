from os import path
from io import open
from setuptools import setup, find_packages

this_directory = path.abspath ( path.dirname (__file__) )

## get __version__ from version.py
__version__ = None
ver_file = path.join ("tf_gen_models", "version.py")
with open (ver_file) as file:
  exec ( file.read() )

## load README
def readme():
  readme_path = path.join (this_directory, "README.md")
  with open (readme_path, encoding = 'utf-8') as file:
    return file.read()

## load requirements
def requirements():
  requirements_path = path.join (this_directory, "requirements/base.txt")
  with open (requirements_path, encoding="utf-8") as file:
    return file.read() . splitlines()

setup (
        name = "tf-gen-models",
        version = __version__,
        description  = "tf-gen-models",
        long_description = readme(),
        long_description_content_type = "text/markdown",
        url = "https://github.com/mbarbetti/tf-gen-models",
        author = "Matteo Barbetti",
        author_email = "matteo.barbetti@fi.infn.it",
        maintainer = "Matteo Barbetti",
        maintainer_email = "matteo.barbetti@fi.infn.it",
        license = "MIT",
        keywords = ["machine-learning", "deep-learning", "tensorflow", "generative-models"],
        packages = find_packages(),
        package_data = {},
        include_package_data = True,
        install_requires = requirements(),
        python_requires  = ">=3.7, <4",
        classifiers = [
                        "Development Status :: 3 - Alpha",
                        "Intended Audience :: Education",
                        "Intended Audience :: Developers",
                        "Intended Audience :: Science/Research",
                        "License :: OSI Approved :: MIT License",
                        "Programming Language :: Python :: 3",
                        "Programming Language :: Python :: 3 :: Only",
                        "Programming Language :: Python :: 3.7",
                        "Programming Language :: Python :: 3.8",
                        "Programming Language :: Python :: 3.9",
                        "Topic :: Scientific/Engineering",
                        "Topic :: Scientific/Engineering :: Mathematics",
                        "Topic :: Scientific/Engineering :: Artificial Intelligence",
                        "Topic :: Software Development",
                        "Topic :: Software Development :: Libraries",
                        "Topic :: Software Development :: Libraries :: Python Modules",
                      ],
  )