# Image captioning with the Flickr8k Dataset

This project attempts to train an encoder-decoder architecture to predict captions for images using the Flickr8k dataset. The original dataset is not uploaded due to size, but there are many versions available on the Internet. For example, one can be found on Kaggle: https://www.kaggle.com/datasets/nunenuh/flickr8k

# The Flickr8k Dataset

The dataset comprises 8000 images of various dimensions. The metadata folder of the dataset contains the following files:
- `CrowdFlowerAnnotations.txt` and `ExpertAnnotations.txt`: represent the crowd and expert versions of annotations against the captions, which judge to which extent the captions actually describe the provided images.
- `Flickr_8k.trainImages.txt`, `Flickr_8k.devImages.txt` and `Flickr_8k.testImages.txt`: the image names in each of the train-validation-test datasets. There are 6000 train, 1000 validation, and 1000 test images in total.
- `Flickr8k.token.txt` and `Flickr8k.lemma.token.txt`: the captions. Each image contains 5 captions from 5 different humans. The "lemma" version represents the lemmatised captions.

<img width="716" alt="Screenshot 2024-06-01 at 1 15 03 PM" src="https://github.com/nicnl31/image-captioning-flickr8k/assets/86213993/825d001d-54e6-47fa-bcd4-f627f2769584">

# Preparation

The following steps are preformed to prepare the data for training:
- Resize all images to standardised `224x224x3`.
- A `Vocabulary` object is created, which represents the "dictionary" for a model. It is essentially a two-way bijective mapping between the set of individual tokens that make up the whole corpus and the set of corresponding indices. This corpus also includes special tokens for training and evaluation purposes, which are the beginning-of-sentence `<BEG>`, end-of-sentence `<END>`, padding `<PAD>`, and null `<UNK>`.
- 
