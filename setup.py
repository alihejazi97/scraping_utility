from setuptools import find_packages, setup

long_description = 'don"t bother'

setup(
    name="idgenerator",
    version="0.0.10",
    description="An id generator that generated various types and lengths ids",
    package_dir={"scraping_utility": "src"},
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="aliagha",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=[""],
)