<br><br><br>

# Joint Bilateral Learning for Real-time Universal Photorealistic Style Transfer
 [paper](https://arxiv.org/pdf/2004.10955.pdf) 

This is an unofficial implementation of paper “Joint Bilateral Learning for Real-time Universal Photorealistic Style Transfer”.

The bilateral grid upsampling part is borrowed from this repo: https://github.com/creotiv/hdrnet-pytorch.

You can train the model by simply change the dataset location in train.py. There maybe some minor problems,
but I this debugging them will not cost mush time.

I tried to train this project with my implementation, but failed to get results close to the paper shows.

Here are some problems I came across or I cannot make clear by reading the paper:

1. How to use the AdaIN? As far as I know, there are two kinds of AdaIN, one is Huang Xun's origianl implementation,
another is used in FUNIT/StyleGAN that learns a code containing mean std and assin them to network layers. In this paper, 
It seems to be the first kind, so I just use it in this repo.

2. Which layers are used to calculate the content loss and style loss? I didn't find detail description in the paper,
so I simply use the conv1_1, conv2_1, conv3_1, conv4_1 that the author used to extract features.

3. The description of the regularizer term is unclear, in which dimention did the author define the neighbors?

I cannot solve these problems, and the author didn't respond to my email, so I publish this code and hope there are someone 
interested in this work can discuss with me.
