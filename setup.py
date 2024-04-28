from setuptools import setup

setup(
    name='aiflow',
    version='0.1',
    license='BSD',
    author='gyeh',
    author_email='hello@world.com',
    url='http://www.hello.com',
    long_description="hi there!",
    packages=['aiflow', 'aiflow.images'],
    include_package_data=True,
    package_data={'aiflow.images' : ['hello.gif']},
    description="Hello World testing setuptools",
)