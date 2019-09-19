#!/usr/bin/env python
# coding: utf-8

# # Transfer Learning
# 
# In this notebook, you'll learn how to use pre-trained networks to solved challenging problems in computer vision. Specifically, you'll use networks trained on [ImageNet](http://www.image-net.org/) [available from torchvision](http://pytorch.org/docs/0.3.0/torchvision/models.html). 
# 
# ImageNet is a massive dataset with over 1 million labeled images in 1000 categories. It's used to train deep neural networks using an architecture called convolutional layers. I'm not going to get into the details of convolutional networks here, but if you want to learn more about them, please [watch this](https://www.youtube.com/watch?v=2-Ol7ZB0MmU).
# 
# Once trained, these models work astonishingly well as feature detectors for images they weren't trained on. Using a pre-trained network on images not in the training set is called transfer learning. Here we'll use transfer learning to train a network that can classify our cat and dog photos with near perfect accuracy.
# 
# With `torchvision.models` you can download these pre-trained networks and use them in your applications. We'll include `models` in our imports now.

# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


# In[ ]:





# In[ ]:





# In[ ]:





# Most of the pretrained models require the input to be 224x224 images. Also, we'll need to match the normalization used when the models were trained. Each color channel was normalized separately, the means are `[0.485, 0.456, 0.406]` and the standard deviations are `[0.229, 0.224, 0.225]`.

# In[11]:


data_dir = 'Cat_Dog_data'

# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.CenterCrop((224, 224)), transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.CenterCrop((224, 224)), transforms.ToTensor()])
# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# We can load in a model such as [DenseNet](http://pytorch.org/docs/0.3.0/torchvision/models.html#id5). Let's print out the model architecture so we can see what's going on.

# In[4]:


model = models.densenet121(pretrained=True)
model


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# This model is built out of two main parts, the features and the classifier. The features part is a stack of convolutional layers and overall works as a feature detector that can be fed into a classifier. The classifier part is a single fully-connected layer `(classifier): Linear(in_features=1024, out_features=1000)`. This layer was trained on the ImageNet dataset, so it won't work for our specific problem. That means we need to replace the classifier, but the features will work perfectly on their own. In general, I think about pre-trained networks as amazingly good feature detectors that can be used as the input for simple feed-forward classifiers.

# In[12]:


# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# With our model built, we need to train the classifier. However, now we're using a **really deep** neural network. If you try to train this on a CPU like normal, it will take a long, long time. Instead, we're going to use the GPU to do the calculations. The linear algebra computations are done in parallel on the GPU leading to 100x increased training speeds. It's also possible to train on multiple GPUs, further decreasing training time.
# 
# PyTorch, along with pretty much every other deep learning framework, uses [CUDA](https://developer.nvidia.com/cuda-zone) to efficiently compute the forward and backwards passes on the GPU. In PyTorch, you move your model parameters and other tensors to the GPU memory using `model.to('cuda')`. You can move them back from the GPU with `model.to('cpu')` which you'll commonly do when you need to operate on the network output outside of PyTorch. As a demonstration of the increased speed, I'll compare how long it takes to perform a forward and backward pass with and without a GPU.

# In[14]:


import time


# In[ ]:





# In[3]:


for device in ['cpu', 'cuda']:

    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.to(device)

    for ii, (inputs, labels) in enumerate(trainloader):

        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        start = time.time()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if ii==3:
            break
        
    print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# You can write device agnostic code which will automatically use CUDA if it's enabled like so:
# ```python
# # at beginning of the script
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 
# ...
# 
# # then whenever you get a new Tensor or Module
# # this won't copy if they are already on the desired device
# input = data.to(device)
# model = MyModule(...).to(device)
# ```
# 
# From here, I'll let you finish training the model. The process is the same as before except now your model is much more powerful. You should get better than 95% accuracy easily.
# 
# >**Exercise:** Train a pretrained models to classify the cat and dog images. Continue with the DenseNet model, or try ResNet, it's also a good model to try out first. Make sure you are only training the classifier and the parameters for the features part are frozen.

# In[2]:


## TODO: Use a pretrained model to classify the cat and dog images
from torchvision import models
model = models.resnet50()


# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


model


# In[3]:


from torch import nn
for param in model.parameters():
    param.requires_grad = False
classifier = nn.Sequential(nn.Linear(2048, 1024),
                          nn.ReLU(),
                          nn.Dropout(p=0.3),
                          nn.Linear(1024, 40),
                          nn.ReLU(),
                          nn.Dropout(p=0.4),
                          nn.Linear(40, 2),
                          nn.LogSoftmax(dim=1))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


model.fc = classifier


# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


criterion = nn.NLLLoss()


# In[ ]:





# In[6]:


from torch import optim


# In[ ]:





# In[7]:


optimiser = optim.Adam(model.fc.parameters(), lr = 0.003)


# In[ ]:





# In[ ]:





# In[ ]:


model = model.cuda()
for i in range(5):
    for images, labels in trainloader:
        #model.cuda()
        images = images.cuda()
        labels = labels.cuda()
        optimiser.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimiser.step()
    else:
        for images, labels in testloader:
            images = images.cuda()
            labels = labels.cuda()
            
            output = model(images)
            top_p, top_class = output.topk(1, dim=1)
            accuracy = (top_class==labels.view(*top_class.shape))
            print(float(torch.sum(accuracy))/len(accuracy))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




