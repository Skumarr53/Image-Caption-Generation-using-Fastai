# Image Caption Generation

#### Implementation of ***Show, Attend and Tell*** paper

- [Image Caption Generation](#image-caption-generation)
  - [Demo](#demo)
  - [What's in this repo?](#whats-in-this-repo)
  - [Dataset Description](#dataset-description)
  - [Input files preparation](#input-files-preparation)
  - [Model architecture](#model-architecture)
    - [Encoder](#encoder)
    - [Attention Layer](#attention-layer)
    - [Decoder](#decoder)
  - [Training (using Fastai)](#training-using-fastai)
    - [Fastai utilities](#fastai-utilities)
    - [Training in Stages](#training-in-stages)
    - [Model intrpretation](#model-intrpretation)
  - [Technology used](#technology-used)
  - [Credits](#credits)
  - [Creator](#creator)


## Demo
![](snapshots/caption_gen.gif)

## What's in this repo?
* [main-Finalized.ipynb](main-Finalized.ipynb) - Notebook with all the preprocessing, data prepartion, and model building training steps.
* [modules/model.py](modules/model.py) - Pytorch implementation of model architecture.
* [modules/custom_callbacks.py](modules/custom_callbacks.py) - Fastai Callback utilities such as Teacher forcing, gradient clipping, loss and validation metric functions. 
* [web_app](web_app) - This directory contains model deployment setup files. 
 
## Dataset Description

https://www.kaggle.com/ming666/flicker8k-dataset

**Flickr8k** Dataset consisting of around 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events. The images were chosen from six different Flickr groups, and tend not to contain any well-known people or locations, but were manually selected to depict a variety of scenes and situations. 6000 are used for training, 1000 for test and 1000 for development.



## Input files preparation

#### 1. preparation of vocabulary dictionary.

The caption labels needs to be converted into numbers as a network does not accept strings as labels. we need a look-up dictionary and store word to numeric mappings in it. 

Along with it, caption lengths are also computed. Caption lengths are used for optimizing training (discussed in detail in the training part).


#### 2. Create Dataset class

In PyTorch, for Deep learning tasks, inputs are fed in batches because of memory constraints. To facilitate this we should create a class called **Dataset** that facilitates batch creation and loading.

The primary function of Dataset is stores the input paths. This class will be used by Pytorch's *DataLoader()* for loading images in batches.

#### 3. Create Dataloader object

The purpose of the **Dataloader** is to load a batch of inputs and labels pairs to be fed into the network.

It is always a good idea to sort by order of captions length for faster computation. On validation set, **SortSampler** funtion from *Fastai* is used which is built on top of PyTorch's **Sampler**. On the training set, **SortishSampler** that sorts data by order of length with a bit of randomness is used. The sampler return iterator of indices.


#### 4. Create Pad_collate function
Since the captions lengths are of different lengths, padding should be added for shorter captions to bring them to same length as PyTorch expects caption lengths to be of the same size. 

Funtion collect samples and return labels tensor with padding. This funtion is passed as an argment( ```collate_fn``` ) while creating ```DataLoader``` object.

## Model architecture

The network architecture consists of three components i.e encoder, Attention, and decoder. 

### Encoder

The encoder is a convolution neural network that takes in raw images as input and outputs extracted features as encoded images. The extractor produces **L** (no of output convolution layers) vectors each of **D**-dimension (no of pixels) corresponds to part of the image thus indicates **L** different features at different locations have been identified.

For the encoder part, I have used **Resnet-101** architecture pre-trained on **Imagenet**. Since Resnet is trained for classifying different objects last Linear layer outputs 1-d prbability tensor. But, our objective is to get feature images so we have to retain only convolution layers and drop the last feed-forward layers.

### Attention Layer

The attention model generates attention weights at every step based on previous step (**h[t-1]**) hidden state vector it receives from decoder. Hidden state carries information about context the caption that has been generated so far.  

### Decoder

The decoder is the one that generates captions (one word at a step) has LSTM network architecture. The decoder takes attention weighted hidden state which is an improvised version of decoder hidden state at step **t-1** that tells which part of the image should be focused to generate the next word.

The flow is depicted in the following image:
![](snapshots/model.png)

#### Model architecture dimensions
```py
embedding input dimension = 300 
attention dimension = 512
decoder dimension = 512
decoder dropout = 0.5
encoder output dimension = 2048
```


## Training (using Fastai)

As we are using pre-trained weights for the encoder which has been trained on the Imagenet dataset consisting of images of 1000's of different objects, that most likely includes objects found in our dataset. Therefore, the network need not require much of tuning. On the other hand, the decoder has to learn a lot as it starts language modeling from scratch.

So, it is better to train just decoder part (fine_tune off) for the first few epochs until we bring both of them to the same level then train the entire network for the next few epochs. In this way, we can save computational time involved in encoder's gradient computation while the decoder takes most of the updation in the initial few epochs.

Training decoder from scratch requires a lot of computation hence more time. Instead, we can use pre-trained word embeddings (word represent as a numeric vector) to train embedding layer output of which is passed into decoder along with the previous hidden state.


### Fastai utilities

Fastai is deep learning framework built on top of PyTorch with implementation of variuos state of the art methods. It provides a smooth API making it easier for most important deep learning applications. 

* **lr _finder** - It will do a mock training by going over a large range of learning rates, then plot them against the losses. We will pick a value a bit before the minimum, where the loss still improves.

![](snapshots/lr_find.png)

* **fit_one_cycle** - Method is implementation of one cycle policy. lr goes up to max and comes down for one cycle of passing through all mini-batches. In one fit cycle takes entire input and divides into batches of size 'bs'. then start with lr_min for the first batch increase gradually for next batches and when the batch number reaches 30 percent of total batches, lr reaches lr_max and then starts going down and reaches lr_min again at last batch.

    The original 1cycle policy has three steps:

    1. We progressively increase our learning rate from lr_max/div_factor to lr_max and at the same time, we progressively decrease our momentum from mom_max to mom_min.
    2. We do the exact opposite: we progressively decrease our learning rate from lr_max to lr_max/div_factor and at the same time, we progressively increase our momentum from mom_min to mom_max.
    3. We further decrease our learning rate from lr_max/div_factor to lr_max/(div_factor x 100) and we keep momentum steady at mom_max.


**Clipping gradients**:
*  Gradients can vanish because they are continuously multiplied by numbers less than one. This is called the vanishing gradient problem.

* It has little effect on learning, but if you have a "bad minibatch" that would cause gradients to explode for some reason, the clipping prevents that iteration from messing up your entire model.

**Early Stopping**

* The authors of *Show, Attend and Tell paper* observe that correlation between the loss and the BLEU score breaks down after a point, so they recommend to stop training early on when the BLEU score starts degrading or stops improving.

### Training in Stages

In the first stage, the model is trained with encoder part froze i.e only decoder weights allowed to be updated for faster training. The model was run with a batch of ```25``` images for 12 epochs using ```Adam()``` optimizer with a learning rate of ```4e-04```

**Results**:
epoch | train_loss | valid_loss | topK_accuracy | bleu_metric | time
------|------------|------------|---------------|-------------|-----
0 | 4.649515 | 4.511709 | 58.052895 | 0.106774 | 18:29
1 | 4.234053 | 4.231682 | 62.291264 | 0.125098 | 17:41
2 | 4.048578 | 4.089489 | 64.173981 | 0.136820 | 17:13
3 | 3.918362 | 4.001822 | 65.538071 | 0.142155 | 17:17
4 | 3.820599 | 3.946904 | 66.606972 | 0.147784 | 16:14
5 | 3.676066 | 3.904321 | 67.152397 | 0.140314 | 16:08
6 | 3.632400 | 3.884929 | 67.566093 | 0.145791 | 16:08
7 | 3.533431 | 3.860997 | 68.075752 | 0.154064 | 16:08
8 | 3.480697 | 3.852596 | 68.334770 | 0.151733 | 16:08
9 | 3.406797 | 3.853946 | 68.293274 | 0.150269 | 16:08

![](snapshots/loss_stage1.png)


In the second stage, the model is trained with the encoder part unfrozen condition. The model was run with batch of ```5``` images for 10 epochs using ```Adam()``` optimizer with ```1e-04``` learning rate adopting  ```one cycle policy``

**Results**:

epoch | train_loss | valid_loss | topK_accuracy | bleu_metric | time
------|------------|------------|---------------|-------------|-----
0 | 3.547406 | 3.914244 | 67.741348 | 0.134781 | 40:54
1 | 3.717416 | 3.972998 | 66.951462 | 0.142118 | 42:23
2 | 3.721014 | 3.950798 | 67.553833 | 0.150034 | 42:25
3 | 3.566937 | 3.928402 | 68.072418 | 0.155043 | 41:56
4 | 3.473794 | 3.910442 | 68.245857 | 0.163102 | 40:16
5 | 3.350647 | 3.915221 | 68.383591 | 0.161378 | 39:18


![](snapshots/loss_stage2.png)

**Evaluation Beam search**

**Beam search**: Involves selecting words with top ```k```(beam width) scores rather than a word with the best score at each step. Beam Search is useful for any language modeling problem because it finds the most optimal sequence.

![](snapshots/beam_search.png)

</br>

**Validation results**

Beam Size | Test BLEU-4
----------|-------------
1 |  21.8
3 |  23.46
5 |  23.9


### Model intrpretation

![](snapshots/eval.jpeg)


## Technology used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://www.clipartmax.com/png/middle/322-3225839_tnt-pytorch-machine-learning-logo.png" width=250>](https://pytorch.org/)
[<img target="_blank" src="https://buzz-prod-photos.global.ssl.fastly.net/img/87a50dce-a64d-4747-b152-30f2f13e80ef" width=150>](https://www.fast.ai/)
[<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=202>](https://flask.palletsprojects.com/en/1.1.x/) 
[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/HTML5_logo_and_wordmark.svg/120px-HTML5_logo_and_wordmark.svg.png" width=100>]()
[<img target="_blank" src="https://openjsf.org/wp-content/uploads/sites/84/2019/10/jquery-logo-vertical_large_square.png" width=100>](https://jquery.com/)


</br>

## Credits

1. [Show, Attend and Tell - paper (arxiv)](https://arxiv.org/abs/1502.03044)

2. [Illustrated Guide to LSTM's and GRU's - Medium](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21) 

2. [a-PyTorch-Tutorial-to-Image-Captioning - GitHub](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)

</br>

------
## Creator
[<img target="_blank" src="https://media-exp1.licdn.com/dms/image/C4D03AQG-6F3HHlCTVw/profile-displayphoto-shrink_200_200/0?e=1599091200&v=beta&t=WcZLox9lzVQqIDJ2-5DsEhNFvEE1zrZcvkmcepJ9QH8" width=150>](https://skumar-djangoblog.herokuapp.com/)
