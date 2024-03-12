from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .' #declaring a constant 
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requiremnets
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
name='mlproject',
version='0.0.1',
author='shilpa',
author_email='shilpashankar005@gmail.com',
packages=find_packages(),
# install_requires=['pandas','numpy','seaborn']
#we can't manually write all the packages like this, hence write a function instead
install_requires=get_requirements('requirements.txt')

)