import torch
import torch.nn as nn


def gradient_penalty(critic, real, fake,labels=None, device="cpu"):
    """
        see paper:  1. 'Wasserstein GAN' -  https://arxiv.org/pdf/1701.07875.pdf
                        https://www.alexirpan.com/2017/02/22/wasserstein-gan.html
                    2. 'Improved Training of Wasserstein GANs'
                    https://arxiv.org/pdf/1704.00028.pdf
    """
    BATCH_SIZE, C, H, W = real.shape
    # print(f"real.shape:{real.shape}")
    # print(f"fake.shape:{fake.shape}")


    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    if labels is None:
        mixed_scores = critic(interpolated_images)
    else:
        mixed_scores = critic(interpolated_images, labels)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(state, filename="celeba_wgan_gp.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, gen, disc):
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])

