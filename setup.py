from setuptools import setup, find_packages
import codecs
import os

# Setting up
here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '1.0.0'
DESCRIPTION = 'LLM evaluation and data management'
LONG_DESCRIPTION = 'Python library for LLM AI model evaluation and data management.'

# Setting up
setup(
    name="aiflow",
    version=VERSION,
    author="Yuki Arimo",
    author_email="<yukiarimo@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages = find_packages(),
    keywords=['python', 'ai', 'llm', 'nlp', 'neural network', 'artificial intelligence', 'machine learning', 'data'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)