import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dataframe-logger",
    version="0.0.1",
    author="Omar Shouman",
    author_email="omar.shouman@gmail.com",
    description="Logger for iterative processes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/omsh/DataFrameLogger",
    packages=setuptools.find_packages(),
    install_requires=["pandas", "numpy", "matplotlib"],
    extras_require={
        "dev": ["pytest >= 3.7", "pytest-cov", "black", "twine", "setuptools", "wheel"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 1 - Planning",
    ],
)
