import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class ListDataset(Dataset):
    def __init__(self, list_path):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        # self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        annotation = self.img_files[index % len(self.img_files)].strip().split(' ')

        #---------
        #  Image
        #---------
        img = cv2.imread(annotation[0])
        img = img[:,:,::-1]
        img = np.asarray(img, 'float32')
        img = img.transpose((2, 0, 1))
        img = (img - 127.5) * 0.0078125
        input_img = torch.FloatTensor(img)

        #---------
        #  Label
        #---------
        label = int(annotation[1])
        return input_img, label
        # sample = {'input_img': input_img, 'label': label}
        # return sample

if __name__ == '__main__':

    train_path = '../data_preprocessing/mid_store/imglist_anno_12_train.txt'
    val_path = '../data_preprocessing/mid_store/imglist_anno_12_val.txt'
    batch_size = 8
    dataloaders = {'train': torch.utils.data.DataLoader(ListDataset(train_path), batch_size=batch_size, shuffle=True),
                   'val': torch.utils.data.DataLoader(ListDataset(val_path), batch_size=batch_size, shuffle=True)}
    dataset_sizes = {'train': len(ListDataset(train_path)), 'val': len(ListDataset(val_path))}

    for i_batch, sample_batched in enumerate(dataloaders['train']):

        images_batch, label_batch, bbox_batch, landmark_batch = sample_batched['input_img'], sample_batched[
            'label'], sample_batched['bbox_target'], sample_batched['landmark']

        print(i_batch, images_batch.shape, label_batch.shape, bbox_batch.shape, landmark_batch.shape)


