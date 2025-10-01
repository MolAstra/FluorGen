import os
import setuptools

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="molecule_generation",
    use_scm_version=True,
    license="MIT",
    author="Krzysztof Maziarz",
    author_email="krzysztof.maziarz@microsoft.com",
    description="Implementations of deep generative models of molecules.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/molecule-generation/",
    setup_requires=["setuptools_scm"],
    python_requires=">=3.7",
    install_requires=[],
    packages=setuptools.find_packages(),
    package_data={"": ["test_datasets/*.pkl.gz", "test_datasets/*.smiles"]},
    include_package_data=True,
    entry_points={
        "console_scripts": ["molecule_generation = molecule_generation.cli.cli:main"]
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
