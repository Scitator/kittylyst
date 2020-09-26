import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kittylyst",
    version="0.1.0",
    author="Sergey Kolesnikov",
    author_email="scitator@gmail.com",
    description="A tiny Catalyst-like experiment runner framework on top of micrograd.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scitator/kittylyst",
    packages=setuptools.find_packages(),
    install_requires=["micrograd==0.1.0",],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
