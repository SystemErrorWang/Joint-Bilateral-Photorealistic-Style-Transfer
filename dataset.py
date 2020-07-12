import os
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset




def load_simple_list(src_path):
    name_list = list()
    for name in os.listdir(src_path):
        path = os.path.join(src_path, name)
        name_list.append(path)
    name_list = [name for name in name_list if '.jpg' in name]
    return name_list


class StyleDataset(Dataset):
    def __init__(self, content_list, style_list, size):
        self.content_list = content_list
        self.style_list = style_list
        self.size = size

    def __getitem__(self, index):
        c_idx = np.random.randint(len(self.content_list))
        s_idx = np.random.randint(len(self.style_list))
        c_path = self.content_list[c_idx]
        s_path = self.style_list[s_idx]
        content = cv2.imread(c_path)[:, :, ::-1]
        style = cv2.imread(s_path)[:, :, ::-1]
        try:
            h0, w0, c0 = np.shape(content)
            dh0 = np.random.randint(h0-self.size)
            dw0 = np.random.randint(w0-self.size)
            content = content[dh0: dh0+self.size, dw0: dw0+self.size, :]
        except:
            content = cv2.resize(content, (self.size, self.size))

        try:
            h1, w1, c1 = np.shape(style)
            dh1 = np.random.randint(h1-self.size)
            dw1 = np.random.randint(w1-self.size)
            style = style[dh1: dh1+self.size, dw1: dw1+self.size, :]
        except:
            style = cv2.resize(style, (self.size, self.size))

        #content = cv2.resize(content, (self.size, self.size()))
        #style = cv2.resize(style, (self.size, self.size()))
        content = content.transpose((2, 0, 1))/255.0
        style = style.transpose((2, 0, 1))/255.0
        return content, style 

    def __len__(self):
        return 100000




    

def style_loader(content_folder, style_folder, size, batch_size):
    content_list = load_simple_list(content_folder)
    style_list = load_simple_list(style_folder)
    dataset = StyleDataset(content_list, style_list, size)
    num_workers = 8 if batch_size > 8 else batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    return dataloader
    

if __name__ == '__main__':
    pass
        
    

