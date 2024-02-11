from distutils.core import setup
from setuptools import setup, find_packages


long_description = open("README.md", "r").read()

setup(
    name="easy_explain",
    packages=find_packages(),
    version="0.4.3",
    license="MIT",
    description="A library that helps to explain AI models in a really quick & easy way",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Stavros Theocharis",
    author_email="stavrostheocharis@yahoo.gr",
    url="https://github.com/stavrostheocharis/stavrostheocharis",
    download_url="https://github.com/stavrostheocharis/easy_explain/archive/refs/tags/v0.4.3.tar.gz",
    keywords=[
        "explainable ai",
        "xai",
        "easy explain",
    ],
    install_requires=[
        "backports.weakref",
        "captum",
        "matplotlib",
        "opencv-python",
        "Pillow",
        "scikit-learn",
        "scipy",
        "torch",
        "torchcam",
        "torchvision",
        "ultralytics",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
