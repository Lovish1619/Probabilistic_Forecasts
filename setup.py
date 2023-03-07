from setuptools import setup, find_packages

# Declaring variables for setup functions
PROJECT_NAME = "Probabilistic Predictor"
VERSION = "0.0.1"
AUTHOR_NAME = "Dr. Suryachandra A. Rao and his subordinate Lovish Mittal"
DESCRIPTION = "This project aims to provide the quantification of uncertainties associated with the post processing " \
              "of the environment variables. "
REQUIREMENTS_FILE_NAME = "requirements.txt"


def get_requirements_list() -> None:
    """
    Description: This function is going to return list of libraries that will be used in this project

    return: List of libraries as string
    """
    with open(REQUIREMENTS_FILE_NAME) as requirement_file:
        return requirement_file.readlines().remove("-e .")


setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR_NAME,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=get_requirements_list()
)
