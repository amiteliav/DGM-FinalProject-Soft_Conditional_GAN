import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset

from torchinfo import summary
from torchvision import datasets, models, transforms

import torchvision
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# from sklearn import decomposition
# from sklearn import manifold
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os

import copy
import random
import time
import datetime

import argparse

# loading images had some error, looked online to fix it:
import warnings
from PIL import Image
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#--------------------------------------------------------

# -- from our project ---
from project_utils import choose_cuda, print_model_summary
from project_utils import CelebA_dataset


# ----------
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
#--------------


class LeNet(nn.Module):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim,out_channels=6,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.fc_1 = nn.Linear(16 * 13 * 13, 120)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc_2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc_3 = nn.Linear(84, output_dim)

    def forward(self, x):
        # img size = [3,crop,crop]

        # x = [batch size, 3, 64, 64]
        x = self.conv1(x)
        # print(f"x.shape:{x.shape}")
        # x = [batch size, 6, 60, 60]
        x = F.max_pool2d(x, kernel_size=2)
        # print(f"x.shape:{x.shape}")
        # x = [batch size, 6, 30, 30]
        x = F.relu(x)
        x = self.conv2(x)
        # print(f"x.shape:{x.shape}")
        # x = [batch size, 16, 26, 26]
        x = F.max_pool2d(x, kernel_size=2)
        # print(f"x.shape:{x.shape}")
        # x = [batch size, 16, 13, 13]
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        # print(f"x.shape:{x.shape}")
        # x = [batch size, 16*13*13 = 2704]
        h = x

        x = self.fc_1(x)
        # print(f"x.shape:{x.shape}")
        # x = [batch size, 120]
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc_2(x)
        # print(f"x.shape:{x.shape}")
        # x = [batch size, 84]
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc_3(x)
        # print(f"x.shape:{x.shape}")
        # x = [batch size, output dim]
        return x, h


def plot_filter(images, filter):
    images = images = torch.cat([i.unsqueeze(0) for i in images], dim=0).cpu()
    filter = torch.FloatTensor(filter).unsqueeze(0).unsqueeze(0).cpu()

    n_images = images.shape[0]

    filtered_images = F.conv2d(images, filter)

    fig = plt.figure(figsize=(20, 5))

    for i in range(n_images):
        ax = fig.add_subplot(2, n_images, i + 1)
        ax.imshow(images[i].squeeze(0), cmap='bone')
        ax.set_title('Original')
        ax.axis('off')

        image = filtered_images[i].squeeze(0)

        ax = fig.add_subplot(2, n_images, n_images + i + 1)
        ax.imshow(image, cmap='bone')
        ax.set_title(f'Filtered')
        ax.axis('off');



######################################################
######################################################
######################################################
############################################
def choose_cuda(cuda_num):
    if cuda_num=="cpu":
        device = "cpu"
    elif torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if 0 <= cuda_num <= device_count - 1:  # devieces starts from '0'
            device = torch.device(f"cuda:{cuda_num}")
        else:
            print(f"Cuda Num:{cuda_num} NOT found, choosing cuda:0")
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


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

##########################



class Solver(object):
    def __init__(self,model, optimizer,loss,
                 dataset_train, dataset_test, dataloader_train, dataloader_test, config):
        """Initialize configurations"""
        self.dir_root               = config.dir_root
        self.dir_results            = config.dir_results
        self.dir_model_folder       = config.dir_model_folder
        self.dir_fig                = config.dir_fig
        self.dir_models             = config.dir_models
        self.save_name              = config.save_name

        self.num_classes        =config.num_classes

        # ----------------
        self.model = model
        self.opt   = optimizer
        self.loss  = loss
        #------------------

        self.dataset_train      = dataset_train
        self.dataset_test       = dataset_test
        self.loader_train       = dataloader_train
        self.loader_test        = dataloader_test

        self.device             = choose_cuda(config.cuda_num)
        self.num_epochs         = config.num_epochs
        self.num_to_test        = config.num_to_test
        self.batch_train        = config.batch_train
        self.batch_test         = config.batch_test


    def train_epoch(self):
        # print("start train")
        epoch_loss = 0
        epoch_acc = 0

        self.model.to(self.device)
        self.model.train()

        for batch_idx, (data, label) in enumerate(self.loader_train):
            # print(f"Batch index:{batch_idx}")
            data = data.to(self.device)
            label = label.to(self.device)

            # print(f"Train: data.shape:{data.shape}")

            self.opt.zero_grad()
            label_pred = self.model(data)

            # ############
            # print(f"label_pred:{label_pred.shape}")
            # print(f"label:{label.shape}")
            # # print(f"max label:{torch.max(label, 1)[1].shape}")
            # ############

            loss = self.loss(label_pred, label)
            acc = calculate_accuracy(label_pred, label)
            loss.backward()
            self.opt.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(self.dataset_train), epoch_acc / (batch_idx+1)


    def test_epoch(self):
        # print("start Test")
        epoch_loss = 0
        epoch_acc = 0

        self.model.to(self.device)
        self.model.eval()

        tot_test = 0
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(self.loader_test):
                # print(f"Batch index:{batch_idx}")
                # print(f"Test: data shape:{data.shape}")

                if (self.num_to_test is not None) and (batch_idx !=0) \
                        and (tot_test >= self.num_to_test):  # stop after given number of exapmles
                    print(f"Stop testing after:~{tot_test} examples, which were {batch_idx} batches, "
                          f"num_to_test:{self.num_to_test}")
                    break
                tot_test+= data.shape[0]

                data = data.to(self.device)
                label = label.to(self.device)

                label_pred = self.model(data)
                loss = self.loss(label_pred, label)
                acc = calculate_accuracy(label_pred, label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / tot_test, epoch_acc / (batch_idx+1)


    def train_solver(self):
        print(f"Start training, with num epochs:{self.num_epochs}")
        best_valid_acc = 0

        train_loss_all = []
        test_loss_all = []
        train_acc_all = []
        test_acc_all = []
        for epoch in range(self.num_epochs):
            start_time = time.monotonic()
            train_loss, train_acc = self.train_epoch()
            valid_loss, valid_acc = self.test_epoch()
            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_acc > best_valid_acc:  # saving only if validation accuracy is higher
                best_valid_acc = valid_acc  # update the best validation accuracy value
                # Saving the model
                print("Start Saving model")
                if not os.path.exists(self.dir_models):
                    os.makedirs(self.dir_models)

                filename = f"classifer_TotEpochs_{self.num_epochs}_batch_size_{self.batch_train}" \
                           f"_checkpoint_{epoch}_ValAcc_{valid_acc * 100:.3f}"
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': self.opt.state_dict(),
                    'num_epoch': self.num_epochs,
                    'batch_train': self.batch_train},  # end of parameters-to-be-saved list
                    f"{self.dir_models}/{filename}.tar")
                print('Finished saving model - Checkpoint Saved')
            # --- finish saving the best-acc-so-far model

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {(train_loss*1000):.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {(valid_loss*1000):.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

            train_loss_all.append(train_loss)
            test_loss_all.append(valid_loss)
            train_acc_all.append(train_acc)
            test_acc_all.append(valid_acc)
        ########### end training

        print(f"Best test accuracy during training:{best_valid_acc * 100:.2f}%\n")

        # -----------------------------------------------------------------------------

        # --------- Plot and save losses ----------
        # create folder for plots
        if not os.path.exists(self.dir_fig):
            os.makedirs(self.dir_fig)

        # Save plot - loss
        plt.figure()
        plt.plot(train_loss_all, linewidth=3, color='blue', label='Loss train')
        plt.plot(test_loss_all, linewidth=3, color='orange', label='Loss test')
        plt.legend()
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss - Train vs Test', fontsize=12)
        plt.grid(True)

        fig_name = f"Losses"
        plt.savefig(f"{self.dir_fig}/{fig_name}")

        # Save plot - acc
        plt.figure()
        plt.plot(train_acc_all, linewidth=3, color='blue', label='Acc train')
        plt.plot(test_acc_all, linewidth=3, color='orange', label='Acc test')
        plt.legend()
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Acc - Train vs Test', fontsize=12)
        plt.grid(True)

        fig_name = f"Accuracy"
        plt.savefig(f"{self.dir_fig}/{fig_name}")


def main(config):
    device = choose_cuda(config.cuda_num)

    # ------- Means & std ----------------------
    if config.data_name == "wikiart":
        mean_train = [0.5080, 0.4528, 0.3903]
        mean_test = [0.5048, 0.4495, 0.3874]
        std_train = [0.2174, 0.2061, 0.1941]
        std_test = [0.2174, 0.2051, 0.1922]
    elif config.data_name == "celebA" or config.data_name == "men_women":
        mean_train = [0.5080, 0.4528, 0.3903]
        mean_test = [0.5048, 0.4495, 0.3874]
        std_train = [0.2174, 0.2061, 0.1941]
        std_test = [0.2174, 0.2051, 0.1922]
    elif config.data_name == "mnist":
        mean_train = [0.5]
        mean_test = [0.5]
        std_train = [0.2174]
        std_test = [0.2174]
    else:
        mean_train = [0.5, 0.5, 0.5]
        mean_test = [0.5, 0.5, 0.5]
        std_train = [0.2174, 0.2174, 0.2174]
        std_test = [0.2174, 0.2174, 0.2174]
    # -------------------------------------------------

    # ---- Define transformations ------
    img_resize = [config.resize, config.resize]
    transforms_train = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize(img_resize),
                                           transforms.Normalize(mean=mean_train, std=std_train)])

    transforms_test = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize(img_resize),
                                          transforms.Normalize(mean=mean_test, std=std_test)])

    # -------------------------------------------


    #  -------- Choosing the dataset ----------
    print(f"Choosing the dataset:{config.data_name}")
    # # Keep 1 of the following 'dataset' - for mnist, 'wikiart' or other datasets
    if config.data_name == "wikiart" or config.data_name =="celebA":
        dataset_train = torchvision.datasets.ImageFolder(root=config.dir_train, transform=transforms_train)
        dataset_test = torchvision.datasets.ImageFolder(root=config.dir_test, transform=transforms_test)
    elif config.data_name == "mnist":
        # !NOTE!: we intentianlly switch the data for train and test,
        #         so we had less data for training, and more for the GAN. so train-> test, test->train
        dataset_train = datasets.MNIST(root="dataset/", transform=transforms_train, download=True, train=False)
        dataset_test = datasets.MNIST(root="dataset/", transform=transforms_test, download=True, train=True)
    elif config.data_name == "men_women":
        dataset_train = torchvision.datasets.ImageFolder(root=config.dir_train, transform=transforms_train)

        csv_file_path = "/dsi/gannot-lab/datasets/Images_datasets/celcebA_kaggle/list_attr_celeba.csv"
        dataset_test = CelebA_dataset(data_dir=config.dir_test, transform=transforms_test, csv_file = csv_file_path)

        # dataset_test = torchvision.datasets.CelebA(root=config.dir_test, split="test", target_type="attr",
        #                                            transform=transforms_test, download=False)
    else:
        print(f"dataset: '{config.data_name}' NOT supported")


    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_train,
                                               shuffle=True, num_workers=config.num_workers)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config.batch_test,
                                              shuffle=True, num_workers=config.num_workers)

    print(f'Number of training examples: {len(dataset_train)}')
    print(f'Number of test examples:     {len(dataset_test)}')

    num_batch_train = len(iter(dataloader_train))
    num_batch_test = len(iter(dataloader_test))
    print(f"num_batch_train:{num_batch_train}")
    print(f"num_batch_test:{num_batch_test}")
    # -------------------------------------------------------

    # ------------ Choosing a model archit' ----------------------
    if config.choose_model=="LeNet":  # using a model we have created
        print(f"Model chosen: LeNet")
        model = LeNet(input_dim=config.channels_img, output_dim=config.num_classes).to(device)
        loss = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters())
    elif config.choose_model=="ResNet18":  # using a pre-trained ResNet18 model
        print(f"Model chosen: ResNet18")
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features  # retrieve the feature size before the last layer
        model.fc = nn.Linear(num_ftrs, config.num_classes)  # choose the last layer: num_ftrs->num_classes
        model = model.to(device)
        loss = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.fc.parameters(), lr=config.lr, momentum=0.9)
    else:
        print(f"choose_model:{config.choose_model}, Not supported")
    # ----------------------------------------------------------------

    # ----- Create a solver -------
    solver = Solver(model=model, loss=loss,optimizer=optimizer,
                    dataset_train=dataset_train, dataset_test=dataset_test,
                    dataloader_train=dataloader_train, dataloader_test=dataloader_test,
                    config=config)

    solver.train_solver()
    # ------------------------------


if __name__ == '__main__':

    cuda_num = 3

    choose_model = "ResNet18"  # "ResNet18"  , LeNet
    resize = 256

    data_name           = "men_women"  # celebA, men_women, wikiart,  mnist
    channels_img        = 3  # 1 (mnist),  3 (wikiart / RGB datasets),
    num_classes         = 2  # None, 10 (mnist), 6 (wikiart) , 2 (men_women)

    num_epochs          = 50
    batch_train         = 300
    batch_test          = 300
    num_to_test         = 2000          # None, or number of examples to test
    lr                  = 0.001
    num_workers         = 8

    # --------------------------------------------------------------

    now = datetime.datetime.now()
    dt_string = now.strftime("%Y%m%d")  # only date
    save_name = f"{dt_string}_classifier_data_{data_name}_model_{choose_model}_resize{resize}" \
                f"_num_epochs_{num_epochs}_batch_train_{batch_train}_lr_{lr}"

    # ----- Folder for saving plots etc. ---------------------------------
    dir_root = "/home/dsi/amiteli/Master/Courses/DGM/Final_Project"
    dir_results = f"{dir_root}/results"
    dir_model_folder = f"{dir_results}/{save_name}"
    dir_fig = f"{dir_model_folder}/figures"
    dir_models = f"{dir_model_folder}/model"
    # -------------------------------------------------------


    # --- datastes Paths ----
    if data_name=="wikiart":
        dir_train = "/dsi/gannot-lab/datasets/wiki_art_splitted/dataset_classifier"
        dir_test  = "/dsi/gannot-lab/datasets/wiki_art_splitted/dataset_gan"
    elif data_name == "men_women":
        dir_train = "/dsi/gannot-lab/datasets/Images_datasets/MenWomenFaces"
        dir_test = "/dsi/gannot-lab/datasets/Images_datasets/celcebA_kaggle"
    elif data_name == "mnist":
        dir_train = None
        dir_test = None
        print(f"dataset:{data_name}, had it's built in dir.. ")
    else:
        print(f"Error! dataset: {data_name}, NOT supported")
    #-----------------------------------------------------------



    # ----------- Model configuration. ----------------
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda_num', type=int, default=cuda_num)

    parser.add_argument('--choose_model', type=str, default=choose_model)
    parser.add_argument('--resize', type=int, default=resize)

    parser.add_argument('--data_name', type=str, default=data_name)
    parser.add_argument('--channels_img', type=int, default=channels_img)
    parser.add_argument('--num_classes', type=int, default=num_classes)


    parser.add_argument('--num_epochs', type=int, default=num_epochs)
    parser.add_argument('--batch_train', type=int, default=batch_train)
    parser.add_argument('--batch_test', type=int, default=batch_test)
    parser.add_argument('--num_to_test', type=int, default=num_to_test)
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--num_workers', type=int, default=num_workers)

    parser.add_argument('--save_name', type=str, default=save_name)
    parser.add_argument('--dir_train', type=str, default=dir_train)
    parser.add_argument('--dir_test', type=str, default=dir_test)

    parser.add_argument('--dir_root', type=str, default=dir_root)
    parser.add_argument('--dir_results', type=str, default=dir_results)
    parser.add_argument('--dir_model_folder', type=str, default=dir_model_folder)
    parser.add_argument('--dir_fig', type=str, default=dir_fig)
    parser.add_argument('--dir_models', type=str, default=dir_models)

    # ----------------------


    config = parser.parse_args()
    print(config)
    print(" ")

    main(config)