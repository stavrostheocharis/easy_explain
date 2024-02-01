<div align="center">
 
# Easy explain
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-green.svg)](#supported-python-versions) 
[![GitHub][github_badge]][github_link]
[![PyPI][pypi_badge]][pypi_link]
[![Download][download_badge]][download_link]
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Licence][licence_badge]][licence_link] 

**Explain AI models easily**


</div>

## Requirements
### Python version
* Main supported version : <strong>3.10</strong> <br>
* Other supported versions : <strong>3.8</strong> & <strong>3.9</strong>

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


[github_badge]: https://badgen.net/badge/icon/GitHub?icon=github&color=black&label

[github_link]: https://github.com/stavrostheocharis/easy_explain

[pypi_badge]: https://badge.fury.io/py/easy-explain.svg

[pypi_link]: https://pypi.org/project/easy-explain/

[download_badge]: https://badgen.net/pypi/dm/easy-explain

[download_link]: https://pypi.org/project/easy-explain/#files

[licence_badge]: https://img.shields.io/github/license/stavrostheocharis/streamlit-token-craft

[licence_link]: LICENSE