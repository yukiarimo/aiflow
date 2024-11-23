from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='aiflow',
    version='7.0.1',
    author='Yuki Arimo',
    author_email='yukiarimo@gmail.com',
    description="AiFlow Package for Testing and Running LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://yukiarimo.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)