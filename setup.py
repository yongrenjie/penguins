from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="penguins",
    version="0.1",
    author="Jonathan Yong",
    author_email="yongrenjie@gmail.com",
    description="Penguins: an Easy, NPE-free Gateway to Unpacking and Illustrating NMR Spectra",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yongrenjie/penguins",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "numpy>=1.17.0",
        "pandas",
        "matplotlib",
        "seaborn",
    ]

)
