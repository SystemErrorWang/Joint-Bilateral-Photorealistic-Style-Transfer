import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import utils
import torch
import network
import dataset
import argparse
import numpy as np

from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image





def train(args):

    model = network.BilateralNetwork(args.size).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    content_criterion = utils.ContentLoss().cuda()
    style_criterion = utils.StyleLoss().cuda()
    content_folder = '/home/tiger/dataset/image'
    style_folder = '/home/tiger/dataset/style'
    train_loader = dataset.style_loader(content_folder, style_folder,
                                         args.size, args.batch_size)
    num_batch = len(train_loader)
    for epoch in range(args.epoch):
        for idx, batch in tqdm(enumerate(train_loader), total=num_batch):
            total_iter = epoch*num_batch  + idx
            content = batch[0].float().cuda()
            style = batch[1].float().cuda()
            
            optimizer.zero_grad()

            output, coeffs = model(content, style)
            c_loss = content_criterion(output, content)
            s_loss = style_criterion(output, style)
            r_loss = utils.regularize(coeffs)

            total_loss = 0.5*c_loss + s_loss + 0.15*r_loss
            total_loss.backward()
             
            optimizer.step()

            if np.mod(total_iter+1, 50) == 0:
                print('{}, Epoch:{} Iter:{}, total loss: {}, c loss:{}, s loss:{}, r loss:{}'\
                    .format(args.save_dir, epoch, total_iter, total_loss.item(), 
                            c_loss.item(), s_loss.item(), r_loss.item()))
                
                
            if np.mod(total_iter+1, 500) == 0:
                
                if not os.path.exists(args.save_dir+'/images'):
                    os.mkdir(args.save_dir+'/images')
                
                save_image(content, args.save_dir+'/images/iter{}_content.jpg'.format(total_iter))
                save_image(style, args.save_dir+'/images/iter{}_style.jpg'.format(total_iter))
                save_image(output, args.save_dir+'/images/iter{}_output.jpg'.format(total_iter))
                
    
        torch.save(model.state_dict(), 
                    args.save_dir+'/model_epoch{}.pth'.format(epoch))






if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--size', default=256, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--save_dir', default='result1', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    
    train(args)
