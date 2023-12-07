from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

setup(name="aic_algorithms",
      version="0.0.1",
      author="Sangwon Seo",
      author_email="sangwon.seo@rice.edu",
      description="Algorithms for AI Coach",
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_packages(
          exclude=["tests", "tests.*", "examples", "examples.*"]),
      python_requires='>=3.8',
      install_requires=[
          'numpy',
          'tqdm',
          'scipy',
          'sparse',
          'torch',
          'termcolor',
          'tensorboard',
          'gym==0.21.0',
          "stable-baselines3<=1.8.0,>=1.1.0",
      ])
