from setuptools import find_packages, setup

with open("README.md", "r") as fh:
  long_description = fh.read()

install_requires = [
    "tqdm",
    "pandas"
]

setup(
    name="llmtask",
    version="0.0.2",
    author="Slightwind",
    author_email="slightwindsec@gmail.com",
    description="LLM Downstream Task Evaluation Tool.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    package_dir={"": "src"},
    package_data={"": ['*.csv']},
    include_package_data=True,
    packages=find_packages("src"),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)