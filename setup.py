from distutils.core import setup

setup(
    name="easy_explain",
    packages=["easy_explain"],
    version="0.1.4",
    license="MIT",
    description="A library that helps to explain AI models in a really quick & easy way",
    author="Stavros Theocharis",
    author_email="stavrostheocharis@yahoo.gr",
    url="https://github.com/stavrostheocharis/stavrostheocharis",
    download_url="https://github.com/stavrostheocharis/easy_explain/archive/refs/tags/v0.1.3.tar.gz",
    keywords=[
        "explainable ai",
        "xai",
        "easy explain",
    ],  # Keywords that define your package best
    install_requires=["torch", "torchvision", "captum", "backports.weakref", "backports.lzma"],  # I get to this in a second
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
