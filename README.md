<div align="center">
  <img src="https://raw.githubusercontent.com/mbarbetti/tf-gen-models/main/.github/images/tfg-logo.png" width="800"/>
</div>

<h3 align="center">
  <em>Ready to use implementations of state-of-the-art generative models in TensorFlow</em>
</h3>

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/tf-gen-models.svg)](https://pypi.python.org/pypi/tf-gen-models/)
[![PyPI version shields.io](https://img.shields.io/pypi/v/tf-gen-models.svg)](https://pypi.python.org/pypi/tf-gen-models/)
[![PyPI status](https://img.shields.io/pypi/status/tf-gen-models.svg)](https://pypi.python.org/pypi/tf-gen-models/)
[![DOI](https://zenodo.org/badge/451160183.svg)](https://zenodo.org/badge/latestdoi/451160183)

## Installation

### Dependencies

tf-gen-models requires:

* Python (>= 3.7, < 3.10)
* TensorFlow (>= 2.5)
* Matplotlib (>= 3.4)
* Pillow (>= 8.0)

- - -

The `tf-gen-models` package is built upon TensorFlow 2. See the [TensorFlow install guide](https://www.tensorflow.org/install) for the [pip package](https://www.tensorflow.org/install/pip) while, to enable GPU support, the use [Docker container](https://www.tensorflow.org/install/docker) is recommended. Alternatively, GPU-enabled TensorFlow can be easily installed using the `tensorflow-gpu` package on [conda-forge](https://conda-forge.org/blog/posts/2021-11-03-tensorflow-gpu/).

### User installation

If you already have a working installation of TensorFlow 2 (preferably with the GPU support enabled), the easiest way to install tf-gen-models is using `pip`:

```shell
pip install tf-gen-models
```

## Available generative models

|                 Generative models                 | Implementation | Notebooks | Trends |
|                :-----------------:                |:--------------:|:---------:|:------:|
| <a href="#Generative Aversarial Networks">GAN</a> |       ✔️      |     🛠️    |        |
| <a href="#Variational Autoencoder">VAE</a>        |       ❌      |     ❌    |        |
| <a href="#Normalizing Flows">Norm Flow</a>        |       ❌      |     ❌    |        |

### Generative Adversarial Networks

| Algorithms | Implementation | Conditioning*| Notebooks |                              Paper                              |
|:----------:|:--------------:|:------------:|:---------:|:---------------------------------------------------------------:|
|    `GAN`   |      ✔️       |      🛠️      |    ✔️    |  <a href="https://arxiv.org/abs/1406.2661">arXiv:1406.2661</a>  |
|  `BceGAN`  |      ✔️       |      ❌      |    ✔️    |                                                                 |
|   `WGAN`   |      ✔️       |      ❌      |    ✔️    | <a href="https://arxiv.org/abs/1701.07875">arXiv:1701.07875</a> |
|  `WGAN_GP` |      ✔️       |      ❌      |    ✔️    | <a href="https://arxiv.org/abs/1704.00028">arXiv:1704.00028</a> |
| `CramerGAN`|      ✔️       |      ❌      |    ✔️    | <a href="https://arxiv.org/abs/1705.10743">arXiv:1705.10743</a> |

*Referring to the **conditional version** of GANs proposed in [arXiv:1411.1784](https://arxiv.org/abs/1411.1784).

### Variational Autoencoder

_Planned for release v0.2.0_

### Normalizing Flows

_Planned for release v0.2.0_

## Jupyter notebooks

* MNIST generation with GANs [[GAN](https://github.com/mbarbetti/tf-gen-models/blob/main/notebooks/gan/0_MNIST_gen_DC-GAN.ipynb)] [[BceGAN](https://github.com/mbarbetti/tf-gen-models/blob/main/notebooks/gan/0_MNIST_gen_DC-BceGAN.ipynb)] [[WGAN](https://github.com/mbarbetti/tf-gen-models/blob/main/notebooks/gan/0_MNIST_gen_DC-WGAN.ipynb)] [[WGAN-GP](https://github.com/mbarbetti/tf-gen-models/blob/main/notebooks/gan/0_MNIST_gen_DC-WGAN-GP.ipynb)] [[CramerGAN](https://github.com/mbarbetti/tf-gen-models/blob/main/notebooks/gan/0_MNIST_gen_DC-CramerGAN.ipynb)]

## License

[MIT License](LICENSE)
