import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import os


class ImageData(data.Dataset):
    """ image dataset
    img_root:    image root (root which contain images)
    label_root:  label root (root which contains labels)
    transform:   pre-process for image
    t_transform: pre-process for label
    filename:    MSRA-B use xxx.txt to recognize train-val-test data (only for MSRA-B)
    """

    def __init__(self, img_root, label_root, transform, t_transform):
            
        self.image_path = list(map(lambda x: os.path.join(img_root, x), os.listdir(img_root)))
        self.label_path = list(map(lambda x: os.path.join(label_root, x.split('/')[-1][:-3] + 'png'), self.image_path))
        self.transform = transform
        self.t_transform = t_transform

    def __getitem__(self, item):
        image = Image.open(self.image_path[item]).convert('RGB')
        label = Image.open(self.label_path[item]).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        if self.t_transform is not None:
            label = self.t_transform(label)
        return image, label

    def __len__(self):
        return len(self.image_path)


# get the dataloader (Note: without data augmentation)
def get_loader(img_root, label_root, img_size, batch_size, mode='train', num_thread=4, pin=True):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        t_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
        dataset = ImageData(img_root, label_root, transform, t_transform)
        ##split the dataset to train/test  L -100和100的两个数据集
        dataset, dataset_test = torch.utils.data.random_split(dataset, (dataset.__len__() -100, 100))

        data_loader_train = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_thread,
                                      pin_memory=pin)
        data_loader_test = data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_thread,
                                      pin_memory=pin)
                                      
        return data_loader_train, data_loader_test
    else:
        t_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))  # TODO: it maybe unnecessary
        ])
        dataset = ImageData(img_root, label_root, None, t_transform)
        return dataset

if __name__ == '__main__':      
    img_root = '/home/liuyang/Documents/data/MSRA/images'
    label_root = '/home/liuyang/Documents/data/MSRA/labels'
    data_loader, data_loader_test = get_loader(img_root, label_root, img_size = 224, batch_size = 8)
    print(data_loader_test)
    print(data_loader.__len__())
    print(data_loader_test.__len__())
    import pdb; pdb.set_trace()
    for  i , (img, label) in enumerate(data_loader_test):
        print(i)
        print(data_loader_test)
        print(img.shape)
        print(label.shape)