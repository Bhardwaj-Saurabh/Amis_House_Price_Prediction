from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

# funtion to add requirement to the list
def get_requirements()->List[str]:
    '''
    Returns the list of requirements
    '''
    requirement_list:List[str] = []
    with open('requirements.txt', 'r') as r:
        requirements = r.read()

    for requirement in requirements.split('\n'):
        requirement_list.append(requirement)

    if HYPEN_E_DOT in requirement_list:
        requirement_list.remove(HYPEN_E_DOT)

    return requirement_list

setup(
    name="AmisHousingPrediction",
    version="0.0.1",
    author="saurabh bhardwaj",
    author_email="aryan.saurabhbhardwaj@gmail.com",
    packages = find_packages(),
    install_requires=get_requirements(),
)