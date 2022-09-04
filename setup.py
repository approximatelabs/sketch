import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sketch",
    version="0.0.1",
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
        "altair>=4.2",
        "pandas>1.4",
        "bcrypt>=3.2.2",
        "python-jose>=3.3",
        "passlib>=1.7.4",
        "databases[aiosqlite]>=0.6",
        "datasketch>=1.5",
        "fastapi[all]>=0.79",
        "pyarrow>=8.0",
    ],
    keywords="data sketch model etl automatic join ai embedding profiling",
    project_urls={
        "Homepage": "https://github.com/approximatelabs/sketch",
    },
)
