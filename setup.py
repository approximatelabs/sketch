import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sketch",
    version="0.0.0",
    author="Justin Waugh",
    author_email="justin@approximatelabs.com",
    description="Compute, store and operate on data sketches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/approximatelabs/sketch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pandas>1.4",
        "datasketches>=3.4.0",
        "datasketch>=1.5",
        "fastapi[all]",
        "pyarrow>=8.0"
    ],
    keywords='data sketch model etl automatic join ai embedding profiling',
    project_urls={
        'Homepage': 'https://github.com/approximatelabs/sketch',
    }
)