"""
Discriminator and Generator implementation from DCGAN paper,
with removed Sigmoid() as output from Discriminator (and therefor
it should be called critic)

---------------
here we can support both regular GAN and conditional GAN
"""

import torch
import torch.nn as nn


def choose_cuda(cuda_num):
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if 0 <= cuda_num <= device_count - 1:  # devieces starts from '0'
            device = torch.device(f"cuda:{cuda_num}")
        else:
            device = torch.device(f"cuda:{0}")
    else:
        device = torch.device("cpu")

    print("*******************************************")
    print(f" ****** running on device: {device} ******")
    print("*******************************************")
    return device


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"



class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, img_size, condi=False, embed_type=None, label_type="1d", num_classes=None):
        super(Discriminator, self).__init__()
        self.channels_img = channels_img
        self.features_d   = features_d

        self.num_classes  = num_classes
        self.img_size     = img_size
        self.condi        = condi
        self.embed_type   = embed_type
        self.label_type   = label_type

        conv2d_channels_img = self.channels_img

        if (self.condi is True) and (embed_type is not None): # it is a conditional GAN
            conv2d_channels_img = self.channels_img +1  # add the labels as another channel

            if embed_type=="torch":
                # embedding layer - for the Disc' it is the same size as the images channel (img_size*img_size)
                # Note: for the Gen' we will use a different size, as a hyperparameter.
                self.embed = nn.Embedding(self.num_classes, self.img_size*self.img_size)
            elif embed_type=="linear":
                self.embed = nn.Linear(self.num_classes, self.img_size*self.img_size)
            else:
                print(f"!ERROR! embed_type:'{embed_type}' NOT supported!")

        if self.img_size==128:
            self.disc = nn.Sequential(
                # input: N x channels_img x 128 x 128
                nn.Conv2d(conv2d_channels_img, features_d, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                # _block(in_channels, out_channels, kernel_size, stride, padding)
                self._block(features_d, features_d * 2, 4, 2, 1),
                self._block(features_d * 2, features_d * 4, 4, 2, 1),
                self._block(features_d * 4, features_d * 8, 4, 2, 1),
                self._block(features_d * 8, features_d * 16, 4, 2, 1),
                # After all _block img output is 4x4 (Conv2d below makes into 1x1)
                nn.Conv2d(features_d * 16, 1, kernel_size=4, stride=2, padding=0),
            )
        elif self.img_size==64:
            self.disc = nn.Sequential(
                # input: N x channels_img x 64 x 64
                nn.Conv2d(conv2d_channels_img, features_d, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                # _block(in_channels, out_channels, kernel_size, stride, padding)
                self._block(features_d, features_d * 2, 4, 2, 1),
                self._block(features_d * 2, features_d * 4, 4, 2, 1),
                self._block(features_d * 4, features_d * 8, 4, 2, 1),
                # After all _block img output is 4x4 (Conv2d below makes into 1x1)
                nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            )
        elif self.img_size==32:
            self.disc = nn.Sequential(
                # input: N x channels_img x 32 x 32
                nn.Conv2d(conv2d_channels_img, features_d, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                # _block(in_channels, out_channels, kernel_size, stride, padding)
                self._block(features_d, features_d * 2, 4, 2, 1),
                self._block(features_d * 2, features_d * 4, 4, 2, 1),
                # After all _block img output is 4x4 (Conv2d below makes into 1x1)
                nn.Conv2d(features_d * 4, 1, kernel_size=4, stride=2, padding=0),
            )
        else:
            print(f"Image size of:{self.img_size} is not supported")


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True), #doesn't norm across the batch
            nn.LeakyReLU(0.2))

    def forward(self, x, labels=None):
        """
        we assume labels (if not none) is a 1d label.
        if (self.embed_type=="linear") and (self.label_type=="one_hot")
            it means that we want to use the linear layer for the embbeding, and we need to convert the 1d label
            to a 1-hot-vector emedding, so it will fit to the input size of the linear layer.

        for all cases, after self.embed(labels), embedding are in size [batch, img_size*img_size]
        """
        if self.condi is True:
            if (self.embed_type=="linear") and (self.label_type=="one_hot"):
                labels = nn.functional.one_hot(labels, self.num_classes)
                labels=labels.float()

            embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
            x = torch.cat([x, embedding], dim=1)  # B x C x H x W

        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g,
                 img_size, condi, embed_size=None, embed_type=None, label_type="1d", num_classes=None):
        super(Generator, self).__init__()
        self.channels_noise = channels_noise
        self.channels_img   = channels_img
        self.features_g     = features_g

        self.num_classes    = num_classes
        self.img_size       = img_size
        self.embed_size     = embed_size
        self.condi          = condi
        self.embed_type     = embed_type
        self.label_type     = label_type

        block_in_size = channels_noise

        if (self.condi is True) and (self.embed_type is not None):  # it is a conditional GAN
            block_in_size = channels_noise + embed_size

            if embed_type == "torch":
                # embedding layer: for Gen' we conct' it to the noise in the latent space
                # so we take the embedding dim as a hyperparameter.
                # This is different from what we did for the Disc'.
                self.embed = nn.Embedding(self.num_classes, self.embed_size)
            elif embed_type=="linear":
                self.embed = nn.Linear(self.num_classes, self.embed_size)
            else:
                print(f"!ERROR! embed_type:'{embed_type}' NOT supported!")

        if self.img_size==128:
            self.net = nn.Sequential(
                # Input: N x channels_noise x 1 x 1
                self._block(block_in_size, features_g * 32, 4, 1, 0),  # img: 4x4
                self._block(features_g * 32, features_g * 16, 4, 2, 1),  # img: 8x8
                self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 16x16
                self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 32x32
                self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 64x64
                nn.ConvTranspose2d(features_g * 2, channels_img,
                                   kernel_size=4, stride=2, padding=1),
                nn.Tanh()) # Output: N x channels_img x 128 x 128
        elif self.img_size==64:
            self.net = nn.Sequential(
                # Input: N x channels_noise x 1 x 1
                self._block(block_in_size, features_g * 16, 4, 1, 0),  # img: 4x4
                self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
                self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
                self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
                nn.ConvTranspose2d(features_g * 2, channels_img,
                                   kernel_size=4, stride=2, padding=1),
                nn.Tanh()) # Output: N x channels_img x 64 x 64
        elif self.img_size==32:
            self.net = nn.Sequential(
                # Input: N x channels_noise x 1 x 1
                self._block(block_in_size, features_g * 8, 4, 1, 0),  # img: 4x4
                self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
                self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
                nn.ConvTranspose2d(features_g * 2, channels_img,
                                   kernel_size=4, stride=2, padding=1),
                nn.Tanh()) # Output: N x channels_img x 32 x 32
        else:
            print(f"Image size of:{self.img_size} is not supported")


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x, labels=None):
        """
        we assume labels (if not none) is a 1d label.
        if (self.embed_type=="linear") and (self.label_type=="one_hot")
            it means that we want to use the linear layer for the embbeding, and we need to convert the 1d label
            to a 1-hot-vector emedding, so it will fit to the input size of the linear layer.

        'embed_size' is the size of the embbeding for the generator, because it is NOT like the disc',
             we DONT add the embedding as another channel to the image, but we add it to the latent space.

        for all cases, after self.embed(labels), embedding are in size [batch, embed_size]
        """
        # latent vector z: B x noise_dim x 1 x 1

        if self.condi is True:
            if (self.embed_type=="linear") and (self.label_type=="one_hot"):
                labels = nn.functional.one_hot(labels, self.num_classes)
                labels=labels.float()

            # for 'embedding: unsqueeze to get shape also [B x noise_dim x1x1]
            embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
            # print(f"x:{x.shape}, embedding:{embedding.shape}")
            x = torch.cat([x, embedding], dim=1)

        return self.net(x)


