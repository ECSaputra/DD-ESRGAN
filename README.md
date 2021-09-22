# DD-ESRGAN
Enhancing the performance of ESRGAN (Enhanced Super Resolution Generative Adversarial Network) using dual discriminators.

The dual discriminator approach was originated by Nguyen et al. (https://arxiv.org/abs/1709.03831) to improve the training of GANs.

In adapting the strategy to ESRGAN, 3 new loss functions for the generator and the two discriminators were formulated to adapt to the original design of the ESRGAN model, while adding on the strategy of 'fooling' not one, but two discriminators that work in opposite ways.
The repository also includes the implementation of the original ESRGAN model (https://arxiv.org/abs/1809.00219).
