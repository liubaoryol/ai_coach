from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

setup(name="ai_coach_domain",
      version="0.0.1",
      author="Sangwon Seo",
      author_email="sangwon.seo@rice.edu",
      description="Domains for AI Coach Experiments",
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_packages(exclude=["tests", "tests.*"]),
      python_requires='>=3.8',
      install_requires=[
          'numpy', 'matplotlib', 'click', 'mujoco-py<2.2,>=2.1', 'gym',
          'opencv-python'
      ])
