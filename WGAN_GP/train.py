"""
Training of WGAN-GP regular or conditional
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import Discriminator, Generator, initialize_weights

from torchvision import models

import os
import time
import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np

# loading images had some error, looked online to fix it:
import warnings
from PIL import Image
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#--------------------------------------------------------

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


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def get_soft_labels(batch, n_classes):
    # For now it support only 2 classes
    label_1 = torch.linspace(0, 1, batch).unsqueeze(1)
    label_2 = torch.linspace(1, 0, batch).unsqueeze(1)
    # print(f"labels:{label_1.shape}")
    # print(f"labels:{label_2.shape}")

    labels = torch.cat((label_1,label_2),1)
    # print(f"labels:{labels}")

    return labels

def test_label():
    # import itertools
    #
    # numbers = [0,0,0,1/3,1/3,1/3,2/3,2/3,2/3,1,1,1]
    # target = 1
    #
    # result = [seq for i in range(len(numbers), 0, -1)
    #           for seq in itertools.combinations(numbers, i)
    #           if sum(seq) == target]
    #
    # return result

    import itertools
    lin_factor = 3
    n_classes = 2

    lin = np.linspace(0, 1, lin_factor)
    a=[]
    for _ in range(n_classes):
        a.append(lin)
    # a = np.array([])
    # a = np.repeat(lin,n_classes)

    result = [seq for seq in list(itertools.product(*a)) if sum(seq) == 1]
    return result



def main(config):
    mnist_mean = (0.1307,)
    mnist_std = (0.3081,)

    device              = choose_cuda(cuda_num)

    condi               = config.condi
    embed_type          = config.embed_type
    label_type          = config.label_type

    data_name           = config.data_name
    channels_img        = config.channels_img
    num_classes         = config.num_classes
    num_samples         = config.num_samples

    lr                  = config.lr
    img_size            = config.img_size
    num_epochs          = config.num_epochs
    batch_size          = config.batch_size
    save_in_epoch       = config.save_in_epoch

    z_dim               = config.z_dim
    gen_embedding       = config.gen_embedding
    features_gen        = config.features_gen
    features_critic     = config.features_critic
    critic_iterations   = config.critic_iterations
    lambda_gp           = config.lambda_gp


    ###########################
    ###########################

    # Define transformation over the data
    # trans = transforms.Compose([transforms.Resize([img_size, img_size]),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize([0.5 for _ in range(channels_img)],
    #                                                       [0.5 for _ in range(channels_img)])])

    max_resize = np.maximum(config.classifier_img_size, img_size)
    trans_loading = transforms.Compose([transforms.Resize([max_resize, max_resize]), transforms.ToTensor(),
                                        transforms.Normalize([0.5 for _ in range(channels_img)],
                                                             [0.5 for _ in range(channels_img)])])

    trans_resize_GAN        = transforms.Compose([transforms.Resize([img_size, img_size])])

    if config.classifier_img_size is not None:
        trans_resize_classifier = transforms.Compose([transforms.Resize([config.classifier_img_size,
                                                                         config.classifier_img_size])])

    #  -------- Choosing the dataset ----------
    print(f"Choosing the dataset:{data_name}")
    # # Keep 1 of the following 'dataset' - for mnist, 'wikiart' or other datasets
    if data_name == "celebA":
        dataset = datasets.ImageFolder(root="/dsi/gannot-lab/datasets/Images_datasets/celcebA_kaggle/",
                                       transform=trans_loading)
    elif data_name == "mnist":
        dataset = datasets.MNIST(root="dataset/", transform=trans_loading, download=True)
    elif data_name == "wikiart":
        dataset = datasets.ImageFolder(root="/dsi/gannot-lab/datasets/wiki_art_splitted/dataset_gan",
                                       transform=trans_loading)
    elif data_name == "men_women":
        dataset = datasets.ImageFolder(root="/dsi/gannot-lab/datasets/Images_datasets/MenWomenFaces",
                                       transform=trans_loading)
    else:
        print(f"dataset: '{data_name}' NOT supported")
    # -----------------------------------

    # ----- Choose models name --------
    print(f"Choose models name")
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y%m%d")  # only date
    save_name = f"{dt_string}_data_{data_name}_condi_{condi}_imgSize_{img_size}" \
                f"_embed_type_{embed_type}_label_type_{label_type}" \
                f"_Z_DIM_{z_dim}_F_CRITIC_{features_critic}_F_GEN_{features_gen}_CRITIC_ITERATIONS_{critic_iterations}" \
                f"_lambda_gp_{lambda_gp}" \
                f"_epochs_{num_epochs}_batch_{batch_size}_lr_{lr}"
    # -----------------------------------

    # --- Paths ----
    # TODO: maybe change to save like this: (but than need to change other places in the code)
    # dir_fig_GAN     = f"{config.results}/{save_name}/figures"
    # dir_models_GAN  = f"{config.results}/{save_name}/trained_models"
    # dir_samples_GAN = f"{config.results}/{save_name}/GAN_samples"
    #--
    dir_fig_GAN     = config.dir_fig_GAN
    dir_models_GAN  = config.dir_models_GAN
    dir_samples_GAN = config.dir_samples_GAN
    # ---------------


    # load pre-trained classifer model
    if config.dir_pretrained_classifier is not None:
        # define the classifer model
        classifier = models.resnet18(pretrained=True)
        num_ftrs = classifier.fc.in_features
        classifier.fc = nn.Linear(num_ftrs, config.num_classes)
        classifier = classifier.to(device)
        checkpoint = torch.load(config.dir_pretrained_classifier, map_location=device)  # load the autovc model
        # classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.load_state_dict(checkpoint['model'])
        print(f"\nPre-trained classifer  model:'{config.dir_pretrained_classifier}' loaded\n")


    # ----------- loading the dataset ---------
    print(f"loading the dataset")
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # Original
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,  # my changes
                        num_workers=8, drop_last=True)
    # -----------------------------------

    # -------- Create G, D, models
    print(f"Create G, D, models")
    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    gen = Generator(channels_noise=z_dim, channels_img=channels_img, features_g=features_gen,
                 img_size=img_size, condi=condi, embed_size=gen_embedding,embed_type=embed_type,
                    label_type=label_type, num_classes=num_classes).to(device)

    critic = Discriminator(channels_img=channels_img, features_d=features_critic,
                           img_size=img_size, condi=condi, embed_type=embed_type, label_type=label_type,
                           num_classes=num_classes).to(device)

    initialize_weights(gen)
    initialize_weights(critic)

    # -------- initialize optimizer --------------
    print(f"initialize optimizer")
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))
    # -----------------------------------

    # ------- Create a fixed noise for plotting progress of an image ---
    print(f"Create a fixed noise")
    fixed_noise = torch.randn(num_samples, z_dim, 1, 1).to(device)
    # -----------------------------------

    gen.train()
    critic.train()


    # ------ Create Sampling folder -----
    samples_dir = f"{dir_samples_GAN}/{save_name}"
    print(f"Create Sampling folder:{samples_dir}")
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    # -----------------------------------

    # ------ Print some proprties before training --------
    print("\n#------ Proprties --------")
    print(f"dataset:{data_name}")
    print(f"Training condi' GAN?:{condi}")
    print(f"Epochs: {num_epochs}")
    print(f"batch size: {batch_size}")
    print(f"Number of batchses(=len(loader)): {len(loader)}")
    print(f"Samples from data will be saved in:{samples_dir}")
    print("#----------------------\n")


    print(f"start training with num_epochs:{num_epochs}")
    for epoch in range(num_epochs):
        G_loss_list = []
        D_loss_list = []
        start_time = time.monotonic()  # batch starting time
        for batch_idx, (real, labels) in enumerate(loader):
            real = real.to(device)
            cur_batch_size = real.shape[0]

            if config.dir_pretrained_classifier is not None:
                real_to_classifier = trans_resize_classifier(real)
                labels = classifier(real_to_classifier).to(device)

            if condi is False:
                # for a regular (not condi' GAN) we dont need labels.
                # we set them to None, so we can use the same code for the Condi'-GAN too
                labels = None

            # Now we can resize the loaded images for the GAN (after labels were set)
            real = trans_resize_GAN(real)

            # ------ Train Critic: max { E[critic(real)] - E[critic(fake)] } ------
            # equivalent to minimizing the negative of that
            loss_critic_list = []
            for _ in range(critic_iterations):
                noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
                # ----- Choose for condi / Not -----
                if condi is False:
                    fake        = gen(noise)
                    critic_real = critic(real).reshape(-1)
                    critic_fake = critic(fake).reshape(-1)
                else:
                    fake = gen(noise, labels)
                    critic_real = critic(real, labels).reshape(-1)
                    critic_fake = critic(fake, labels).reshape(-1)
                #------------

                gp = gradient_penalty(critic, real, fake,labels, device=device)
                loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake))
                               + lambda_gp * gp)
                loss_critic_list.append(loss_critic.detach().cpu().numpy())
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()
            D_loss_list.append(np.mean(loss_critic_list))
            # end training D
            # ---------------------------------

            # ------- Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)] -----
            # ----- Choose for condi / Not -----
            if condi is False:
                gen_fake = critic(fake).reshape(-1)
            else:
                gen_fake = critic(fake, labels).reshape(-1)
            # ------------
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            G_loss_list.append(loss_gen.data.item())
            # end training G

            ###### ----- Print losses occasionally and save samples --- ####
            save_each = int(len(loader) / save_in_epoch)  # saving every given number of epoch
            if batch_idx % save_each == 0 and batch_idx >= 0:
                end_time = time.monotonic()  # batch end time
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                # print(f'Epoch: {epoch + 1:02} | Time passed: {epoch_mins}m {epoch_secs}s')
                print(f"Epoch [{epoch + 1:02}/{num_epochs}] | Batch {batch_idx}/{len(loader)} "
                      f" | epoch passed time: {epoch_mins}m {epoch_secs}s  |  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}")

                with torch.no_grad():
                    # ----- Choose for condi / Not -----
                    if condi is False:
                        fake_fixed     = gen(fixed_noise)
                        fake_NotFixed  = gen(noise)
                    else:
                        fake_fixed    = gen(fixed_noise[:num_samples], labels[:num_samples])
                        fake_NotFixed = gen(noise[:num_samples], labels[:num_samples])

                        soft_label     = get_soft_labels(num_samples,2).to(device)
                        fake_softLabel = gen(noise,soft_label)
                    # ------------

                    # take out (up to) num_samples examples
                    img_grid_real = torchvision.utils.make_grid(real[:num_samples], normalize=True)
                    img_grid_fake_fixed = torchvision.utils.make_grid(fake_fixed[:num_samples], normalize=True)
                    img_grid_fake_NotFixed = torchvision.utils.make_grid(fake_NotFixed[:num_samples], normalize=True)
                    img_grid_fake_softLabel = torchvision.utils.make_grid(fake_softLabel[:num_samples], normalize=True)

                    save_path_real = f"{samples_dir}/epoch_{epoch}_batch_{batch_idx}_real.png"
                    save_path_fake_fixed = f"{samples_dir}/epoch_{epoch}_batch_{batch_idx}_fake_fixed.png"
                    save_path_fake_Notfixed = f"{samples_dir}/epoch_{epoch}_batch_{batch_idx}_fake_NotFixed.png"
                    save_path_fake_softLabel = f"{samples_dir}/epoch_{epoch}_batch_{batch_idx}_fake_softLabel.png"

                    torchvision.utils.save_image(torchvision.utils.make_grid(img_grid_real), save_path_real)
                    torchvision.utils.save_image(torchvision.utils.make_grid(img_grid_fake_fixed), save_path_fake_fixed)
                    torchvision.utils.save_image(torchvision.utils.make_grid(img_grid_fake_NotFixed),
                                                 save_path_fake_Notfixed)
                    torchvision.utils.save_image(torchvision.utils.make_grid(img_grid_fake_softLabel),
                                                 save_path_fake_softLabel)

        #### END OF Epoch ###

        ##### Saving models during training ######
        if (config.save_model_every is not None) \
                and (epoch + 1) % config.save_model_every == 0 \
                and (epoch + 1) < num_epochs:
            print("Start Saving model")
            model_checkpoint_dir = f"{dir_models_GAN}/{save_name}"
            G_model_dir = f"{model_checkpoint_dir}/G"
            D_model_dir = f"{model_checkpoint_dir}/D"
            if not os.path.exists(G_model_dir):
                os.makedirs(G_model_dir)
            if not os.path.exists(D_model_dir):
                os.makedirs(D_model_dir)

            G_model_name = f"{G_model_dir}/G_iter_{epoch + 1}_of_{num_epochs}"
            D_model_name = f"{D_model_dir}/D_iter_{epoch + 1}_of_{num_epochs}"

            if (config.checkpoint_model_every is not None) \
                    and (epoch + 1) % config.checkpoint_model_every == 0 \
                    and (epoch + 1) < num_epochs:
                G_model_name = f"{G_model_dir}/G_checkpoint_iter_{epoch + 1}_of_{num_epochs}"
                D_model_name = f"{D_model_dir}/D_checkpoint_iter_{epoch + 1}_of_{num_epochs}"

            G_list_of_files = os.listdir(G_model_dir)
            for file in G_list_of_files:
                if not file.startswith("G_checkpoint"):
                    if file.endswith(".ckpt"):
                        os.unlink(f"{G_model_dir}/{file}")

            D_list_of_files = os.listdir(D_model_dir)
            for file in D_list_of_files:
                if not file.startswith("D_checkpoint"):
                    if file.endswith(".ckpt"):
                        os.unlink(f"{D_model_dir}/{file}")

            torch.save({'last_iters': epoch,
                        'model': gen.state_dict(),
                        'optimizer': opt_gen.state_dict()},
                       f"{G_model_name}.ckpt")
            torch.save({'last_iters': epoch,
                        'model': critic.state_dict(),
                        'optimizer': opt_critic.state_dict()},
                       f"{D_model_name}.ckpt")
            print(f"Mid-model '{G_model_name}', '{D_model_name}' saved!")
        # -----------------
    # =========== end training loop, all epochs ended ============================


    # ---- saving Final model ---------
    print('start saving Final model...')
    model_checkpoint_dir = f"{dir_models_GAN}/{save_name}"
    G_model_dir = f"{model_checkpoint_dir}/G"
    D_model_dir = f"{model_checkpoint_dir}/D"
    if not os.path.exists(G_model_dir):
        os.makedirs(G_model_dir)
    if not os.path.exists(D_model_dir):
        os.makedirs(D_model_dir)

    G_model_name = f"{G_model_dir}/G_final_model"
    torch.save({
        'last_iters': epoch,
        'model': gen.state_dict(),
        'optimizer': opt_gen.state_dict()},
        f"{G_model_name}.ckpt")

    D_model_name = f"{D_model_dir}/D_final_model"
    torch.save({
        'last_iters': epoch,
        'model': critic.state_dict(),
        'optimizer': opt_critic.state_dict()},
        f"{D_model_name}.ckpt")
    # -------------


    # ------- Plot and save losses  -------------
    model_figures_dir = f"{dir_fig_GAN}/{save_name}"
    if not os.path.exists(model_figures_dir):
        os.makedirs(model_figures_dir)

    # Save plot - loss
    plt.figure()
    plt.plot(G_loss_list, linewidth=3, color='blue', label='Loss Generator')
    plt.plot(D_loss_list, linewidth=3, color='orange', label='Loss Discriminator')
    plt.legend()
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss - Generator vs Discriminator', fontsize=12)
    plt.grid(True)

    fig_name = f"GAN_Losses_Epcos_{num_epochs}"
    plt.savefig(f"{model_figures_dir}/{fig_name}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters etc.
    cuda_num = 3

    condi                    = True  # False / True   for regular / conditional
    embed_type               = "linear"  # None, torch , linear
    label_type               = "soft"     # "one_hot", "soft", "1d"

    data_name                = "men_women"  # celebA, men_women, wikiart,  mnist
    channels_img             = 3  # 1 (mnist),  3 (wikiart / RGB datasets),
    num_classes              = 2  # None, 10 (mnist), 6 (wikiart) , 2 (men_women)
    num_samples              = 32

    save_model_every        = 20  # None, or a number (backup saving, deleted when next is done)
    checkpoint_model_every  = 50  # None, or a number (checkpoint the model, saved for good)

    lr                      = 1e-4  # TODO: originaly 1e-4
    img_size                = 128  # TODO: originaly 64
    num_epochs              = 100
    batch_size              = 32  # TODO: originaly 64, 32(condi)
    save_in_epoch           = 2

    z_dim                   = 100  # TODO: originalyl 100
    gen_embedding           = 100  # TODO: None, 100
    features_gen            = 16  # TODO: originally 16
    features_critic         = 32  # TODO: originally 16
    critic_iterations       = 5    # TODO: originally 5
    lambda_gp               = 10   # TODO: originally 10

    # --- Paths ----
    dir_root                     = "/home/dsi/amiteli/Master/Courses/DGM/Final_Project"
    dir_fig_GAN                  = f"{dir_root}/figures/GAN"
    dir_models_GAN               = f"{dir_root}/trained_models/GAN"
    dir_samples_GAN              = f"{dir_root}/GAN_samples"
    dir_results                  = f"{dir_root}/GAN_results"

    # path for the pretrained classifier to make a soft labels, OR None to use the original labels
    # dir_pretrained_classifier    = f"{dir_root}/trained_models/Classifier/" \
    #                                f"classifier_ResNet18_resize256_epochs_50/" \
    #                                f"classifer_TotEpochs_50_batch_size_40_checkpoint_18_ValAcc_77.200.tar"

    dir_pretrained_classifier = f"{dir_root}/" \
                                f"results/" \
                                f"20220424_classifier_data_men_women_model_ResNet18_resize256_num_epochs_50_batch_train_300_lr_0.001" \
                                f"/model/classifer_TotEpochs_50_batch_size_300_checkpoint_28_ValAcc_78.167.tar"
    classifier_img_size          = 256  # None, or a resize factor
    # ---------------


    # Model configuration.
    parser.add_argument('--cuda_num', type=int, default=cuda_num)

    parser.add_argument('--save_model_every', type=int, default=save_model_every)
    parser.add_argument('--checkpoint_model_every', type=int, default=checkpoint_model_every)

    parser.add_argument('--condi', type=bool, default=condi)
    parser.add_argument('--embed_type', type=str, default=embed_type)
    parser.add_argument('--label_type', type=str, default=label_type)

    parser.add_argument('--dir_root', type=str, default=dir_root)
    parser.add_argument('--dir_fig_GAN', type=str, default=dir_fig_GAN)
    parser.add_argument('--dir_models_GAN', type=str, default=dir_models_GAN)
    parser.add_argument('--dir_samples_GAN', type=str, default=dir_samples_GAN)
    parser.add_argument('--dir_results', type=str, default=dir_results)

    parser.add_argument('--dir_pretrained_classifier', type=str, default=dir_pretrained_classifier)
    parser.add_argument('--classifier_img_size', type=int, default=classifier_img_size)

    parser.add_argument('--data_name', type=str, default=data_name)
    parser.add_argument('--channels_img', type=int, default=channels_img)
    parser.add_argument('--num_classes', type=int, default=num_classes)
    parser.add_argument('--num_samples', type=int, default=num_samples)

    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--img_size', type=int, default=img_size)
    parser.add_argument('--num_epochs', type=int, default=num_epochs)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--save_in_epoch', type=int, default=save_in_epoch)

    parser.add_argument('--z_dim', type=int, default=z_dim)
    parser.add_argument('--gen_embedding', type=int, default=gen_embedding)
    parser.add_argument('--features_gen', type=int, default=features_gen)
    parser.add_argument('--features_critic', type=int, default=features_critic)
    parser.add_argument('--critic_iterations', type=int, default=critic_iterations)
    parser.add_argument('--lambda_gp', type=float, default=lambda_gp)



    config = parser.parse_args()
    print(config)
    main(config)