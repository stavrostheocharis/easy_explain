from distutils.core import setup

long_description = open("README.md", "r").read()

setup(
    name="easy_explain",
    packages=["easy_explain"],
    version="0.3.0",
    license="MIT",
    description="A library that helps to explain AI models in a really quick & easy way",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Stavros Theocharis",
    author_email="stavrostheocharis@yahoo.gr",
    url="https://github.com/stavrostheocharis/stavrostheocharis",
    download_url="https://github.com/stavrostheocharis/easy_explain/archive/refs/tags/v0.1.1.tar.gz",
    keywords=[
        "explainable ai",
        "xai",
        "easy explain",
    ],  # Keywords that define your package best
    install_requires=["torch", "torchvision", "captum", "backports.weakref"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
