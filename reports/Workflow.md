# Workflow

## EffNetV2

* Baseline: ...
* Bayesian Hyperparam Tuning: Fine Tune last layer
    * effnetv2_base: tune dropout and learning rate.
    * effnetv2_mixmodel: vary dropout, learning rate, but also now effnetv2 small, medium, large sizes.
    * effnetv2_label_smoothing: keep all hyperparams constant (lr = 3e-4, dropout = 0.65, size=small), vary only label smoothing

TODO:
* augmentation via RandAug: https://keras.io/examples/vision/randaugment/
* augmentation via Mixup: https://keras.io/examples/vision/mixup/
* Use warmup: cosine decay with linear warmup, when finetuning whole network only?

## ViT

* patch size strongly affects performance, choose 32 patch size?
* choose models trained on more image data: imagenet21k
* apply ViT fine-tune tuning strategy and augmentaiton.
