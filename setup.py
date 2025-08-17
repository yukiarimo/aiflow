import os
from setuptools import setup, find_packages
cwd = os.path.dirname(os.path.abspath(__file__))

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setup(
    name='aiflow',
    version='3.0.0',
    author='Yuki Arimo',
    author_email='yukiarimo@gmail.com',
    description="AiFlow Package for Testing and Running LLMs",
    url='https://www.yukiarimo.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=reqs,
    package_data={'': ['*.txt', 'cmudict_*']},
    entry_points={
        "console_scripts": [
            "aiflow = aiflow.main:main",
        ],
    },
    python_requires='>=3.8',
)