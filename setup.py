from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.readlines()
    
setup(
    name='CoLeCT',
    version='0.1',
    packages=find_packages(),
    author='Lorenzo Busellato',
    description='An end-to-end learning from demonstration framework for force-sensitive applications',
    install_requires=requirements
)
