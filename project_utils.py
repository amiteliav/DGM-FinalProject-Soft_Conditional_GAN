import os

import torch, torchvision
from torchinfo import summary
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision.io import read_image


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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


def print_model_summary(model, device,input_size = (20, 3, 64, 64)):
    model = model.to(device)
    # col_names = ["kernel_size", "input_size", "output_size", "num_params", "mult_adds"]
    col_names = ["input_size", "output_size", "num_params"]
    summary(model, input_size=input_size,col_names=col_names)


def norm_unnorm_example(path = "/dsi/gannot-lab/datasets/wiki_art_splitted/dataset_classifier"):
    batch_size = 4
    resize = [128, 128]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    mean_un = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
    std_un = [1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]

    trans_rezise = transforms.Compose([transforms.ToTensor(), transforms.Resize(resize)])

    trans_Norm = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=mean, std=std)])
    trans_UnNorm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean_un, std=std_un)])
    trans_NormAndResize = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=mean, std=std),
                                              transforms.Resize(resize)])

    demo_data_WithNorm = torchvision.datasets.ImageFolder(root=path, transform=trans_NormAndResize)
    demo_data_NoNorm = torchvision.datasets.ImageFolder(root=path, transform=trans_rezise)

    data_loader_WithNorm = torch.utils.data.DataLoader(demo_data_WithNorm, batch_size=batch_size, shuffle=True,
                                                       num_workers=2)
    data_loader_NoNorm = torch.utils.data.DataLoader(demo_data_NoNorm, batch_size=batch_size, shuffle=True,
                                                     num_workers=2)

    for step, (x, y) in enumerate(data_loader_WithNorm):
        # print(f"x.shape:{x.shape}")             # batch
        # print(f"x.shape:{x[0].shape}")          # 1 img from the batch
        # print(f"x.shape:{x[0][0].shape}")       # 1 channel from the img
        # print(f"x.shape:{x[0][0][0].shape}")    # 1 row/clo from 1 channel...
        img_WithNorm = x[0]
        break

    for step, (x, y) in enumerate(data_loader_NoNorm):
        # print(f"x.shape:{x.shape}")
        img_NoNorm = x[0]
        break

    img_WithNorm = np.array(img_WithNorm)
    img_NoNorm   = np.array(img_NoNorm)

    # Plot NoNorm img
    plt.imshow(img_NoNorm.transpose(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # Norm Img and plot it
    img_Norm = trans_Norm(img_NoNorm.transpose(1, 2, 0))
    img_Norm = np.array(img_Norm)
    plt.imshow(img_Norm.transpose(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # UnNorm the Img and plot
    img_unNorm = trans_UnNorm(img_Norm.transpose(1, 2, 0))
    img_unNorm = np.array(img_unNorm)
    plt.imshow(img_unNorm.transpose(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # Plot a pre-norm(normed with loader) Img
    plt.imshow(img_WithNorm.transpose(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # UnNorm the pre-norm Img and plot and plot
    img_unNorm = trans_UnNorm(img_WithNorm.transpose(1, 2, 0))
    img_unNorm = np.array(img_unNorm)
    plt.imshow(img_unNorm.transpose(1, 2, 0))
    plt.xticks([])
    plt.yticks([])
    plt.show()


def cal_mean_std(loader):
    nimages = 0
    mean = 0.
    std = 0.
    for batch, _ in loader:
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)

        # Compute mean and std here
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)

    # Final step
    mean /= nimages
    std /= nimages

    print(f"Number of Img:{nimages}")

    return mean, std


def calc_mean_std_all_dataset(img_size = [64, 64], batch = 100):
    """
    For:
    path_demo = "/home/dsi/amiteli/Master/Courses/DGM/Final_Project/Dataset/Demo"
    path_train = "/dsi/gannot-lab/datasets/wiki_art_splitted/dataset_classifier"
    path_test = "/dsi/gannot-lab/datasets/wiki_art_splitted/dataset_gan"

    running it over the data results with: (resize[64,64], batch=100)
    Demo:  mean_demo: tensor([0.5939, 0.5898, 0.5574]) , std_demo:tensor([0.1632, 0.1700, 0.1727])
    Train: mean_train:tensor([0.5080, 0.4528, 0.3903]) , std_train:tensor([0.2174, 0.2061, 0.1941])
    Test:  mean_demo:tensor([0.5048, 0.4495, 0.3874])  , std_demo:tensor([0.2174, 0.2051, 0.1922])
    """
    path_demo = "/home/dsi/amiteli/Master/Courses/DGM/Final_Project/Dataset/Demo"
    path_train = "/dsi/gannot-lab/datasets/wiki_art_splitted/dataset_classifier"
    path_test = "/dsi/gannot-lab/datasets/wiki_art_splitted/dataset_gan"

    # loading the datasets inorder to calc mean and std
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(img_size)])

    data_demo  = torchvision.datasets.ImageFolder(root=path_demo, transform=trans)
    data_train = torchvision.datasets.ImageFolder(root=path_train, transform=trans)
    data_test  = torchvision.datasets.ImageFolder(root=path_test, transform=trans)

    loader_demo  = torch.utils.data.DataLoader(data_demo,  batch_size=batch, shuffle=True, num_workers=2)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch, shuffle=True, num_workers=2)
    loader_test  = torch.utils.data.DataLoader(data_test,  batch_size=batch, shuffle=True, num_workers=2)

    print("Number of images for each data:")
    print(f'Number of demo examples:     {len(data_demo)} ')
    print(f'Number of training examples: {len(data_train)}')
    print(f'Number of test examples:     {len(data_test)} ')

    # -----------------------------------
    print("\nStart calc for demo")
    mean_demo, std_demo = cal_mean_std(loader_demo)
    print(f"mean_demo:{mean_demo}")
    print(f"std_demo:{std_demo}")

    print("\nStart calc for train")
    mean_train, std_train = cal_mean_std(loader_train)
    print(f"mean_train:{mean_train}")
    print(f"std_train:{std_train}")

    print("\nStart calc for test")
    mean_test, std_test = cal_mean_std(loader_test)
    print(f"mean_demo:{mean_test}")
    print(f"std_demo:{std_test}")



class CelebA_dataset(Dataset):
    def __init__(self, data_dir, transform=None, csv_file = None):
        self.data_dir       = data_dir

        if csv_file is not None:
            self.csv_file_dir   = csv_file
            self.csv_file_data      = pd.read_csv(self.csv_file_dir)
            print(f"loading csv file: {self.csv_file_dir}")

        self.transform = transform

    def __len__(self):
        return len(self.csv_file_data)

    def __getitem__(self, idx):
        while True:
            try:
                if idx>len(self.csv_file_data):
                    idx=0
                data_path = f"{self.data_dir}/img_align_celeba/{self.csv_file_data.iloc[idx, 0]}" # path are in col' 0 in csv
                image = Image.open(data_path).convert('RGB')
                break
            except:
                    idx+=1

        label = self.csv_file_data["Male"][idx]  # labels are under 'male'
        # print(f"labels:{label}")
        if label==-1:
            label = 0
        elif label==1:
            label = 1
        else:
            print(f"Error with labels, got label:{label}")

        image = transforms.ToTensor()(image)
        # print(f"dataset: image shape:{image.shape}")

        image = torch.permute(image, (1,2,0))
        image = image.numpy()
        if self.transform is not None:
            # print("dataset: start transformation")
            image = self.transform(image)

        return image,label


if __name__ == '__main__':

    #############
    # norm_unnorm_example()
    #############
    # calc_mean_std_all_dataset()
    #############

    img_resize = [64,64]
    mean_test = [0.5048, 0.4495, 0.3874]
    std_test = [0.2174, 0.2051, 0.1922]
    transforms_test = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize(img_resize),
                                          transforms.Normalize(mean=mean_test, std=std_test)])

    csv_file_path = "/dsi/gannot-lab/datasets/Images_datasets/celcebA_kaggle/list_attr_celeba.csv"
    dataset_test = CelebA_dataset(data_dir="/dsi/gannot-lab/datasets/Images_datasets/celcebA_kaggle",
                                  transform=transforms_test, csv_file=csv_file_path)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=4000,
                                                  shuffle=True, num_workers=8)

    for batch_idx, (data, label) in enumerate(dataloader_test):
        print(f"batch:{batch_idx}")
        x=1
    print("finish test")