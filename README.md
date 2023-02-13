<div align="center">

# Easy explain
[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue.svg)](#supported-python-versions) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-MIT-informational.svg)](https://github.com/stavrostheocharis/easy_explain/blob/main/LICENSE)

**Explain AI models easily**


</div>

## Requirements
### Python version
* Main supported version : <strong>3.9</strong> <br>
* Other supported versions : <strong>3.7</strong> & <strong>3.8</strong>

To use the scripts on your computer, please make sure you have one of these versions installed.

### Install environment & dependencies

In order to install the current repo you have 2 options:
- Pip install it directly from git inside your prefered repo and use it as a package

#### Installation as a package

In order to use the current repo as a package you need to run the command below inside your project.

```bash
pip install easy-explain
```

## Information about the functionality

easy-explain uses under the hood [Captum](https://captum.ai/). Captum aids to comprehend how the data properties impact the model predictions or neuron activations, offering insights on how the model performs. Captum comes together with [Pytorch library](https://pytorch.org/).

Currently easy-explain is working only for images and only for Pytorch.

You can import the main function of the package 'run_easy_explain' directly as:

```python
from easy_explain.easy_explain import run_easy_explain

```
For more information about how to begin have a look at the [examples notebooks](https://github.com/stavrostheocharis/easy_explain/tree/main/examples).

## How to contribute?

We welcome any suggestions, problem reports, and contributions!
For any changes you would like to make to this project, we invite you to submit an [issue](https://github.com/stavrostheocharis/easy_explain/issues).

For more information, see [`CONTRIBUTING`](https://github.com/stavrostheocharis/easy_explain/blob/main/CONTRIBUTING.md) instructions.

