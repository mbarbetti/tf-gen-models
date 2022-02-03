# Generative Models in TensorFlow

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/tf-gen-models.svg)](https://pypi.python.org/pypi/tf-gen-models/)
[![PyPI version shields.io](https://img.shields.io/pypi/v/tf-gen-models.svg)](https://pypi.python.org/pypi/tf-gen-models/)
[![PyPI status](https://img.shields.io/pypi/status/tf-gen-models.svg)](https://pypi.python.org/pypi/tf-gen-models/)
[![DOI](https://zenodo.org/badge/451160183.svg)](https://zenodo.org/badge/latestdoi/451160183)

Ready to use implementations of state-of-the-art generative models in TensorFlow.

## Installation

#### Dependencies

tf-gen-models requires:

* Python (>= 3.7, < 3.10)
* TensorFlow (>= 2.5)
* Matplotlib (>= 3.4)
* NumPy (>= 1.20)
* Pillow (>= 8.0)

- - -

The `tf-gen-models` package is built upon TensorFlow 2. See the [TensorFlow install guide](https://www.tensorflow.org/install) for the [pip package](https://www.tensorflow.org/install/pip) while, to enable GPU support, the use [Docker container](https://www.tensorflow.org/install/docker) is recommended. Alternatively, GPU-enabled TensorFlow can be easily installed using the `tensorflow-gpu` package on [conda-forge](https://conda-forge.org/blog/posts/2021-11-03-tensorflow-gpu/).

#### User installation

If you already have a working installation of TensorFlow 2 (preferably with the GPU support enabled), the easiest way to install tf-gen-models is using `pip`:

```shell
$ pip install tf-gen-models
```

## Available generative models

|                     Algorithms                    |   Implementation   |       Notebook      |  Trend  |
|                    :----------:                   |  :--------------:  |      :--------:     | :-----: |
| <a href="#Generative Aversarial Networks">GAN</a> | :heavy_check_mark: | :hammer_and_wrench: |   :x:   |
| <a href="#Variational Autoencoder">VAE</a>        |        :x:         |         :x:         |   :x:   |
| <a href="#Normalizing Flows">Norm Flow</a>        |        :x:         |         :x:         |   :x:   |

#### Generative Adversarial Networks

...

### Variational Autoencoder

#### Normalizing Flows

...

## Jupyter notebooks

bla bla bla

## License

[MIT License](LICENSE)
