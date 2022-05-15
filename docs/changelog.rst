
Implemented
^^^^^^^^^^^

* Introduced MLflow for hyperparameter search and model evaluation.
* Implemented Vision Transformer architecture. However Mac M1 chip does not support certain nodes of this architecture. So do not proceed.
* Implemented pydantic for validation of input hyperparam configuration and general package configuration

Roadmap
^^^^^^^

* Augmentation Technique RandAugment, https://keras.io/examples/vision/randaugment/, https://arxiv.org/abs/1909.13719
* Augmentation Technique  3-Augment, https://arxiv.org/abs/2204.07118
* Mixup Regularization, https://keras.io/examples/vision/mixup, https://arxiv.org/abs/1710.09412
* Stochastic Depth Regularization, in TFA: https://www.tensorflow.org/addons/api_docs/python/tfa/layers/StochasticDepth
