# Image captioning with the Flickr8k Dataset

This project attempts to train an encoder-decoder architecture to predict captions for images using the Flickr8k dataset. The original dataset is not uploaded due to size, but there are many versions available on the Internet. For example, one can be found on Kaggle: https://www.kaggle.com/datasets/nunenuh/flickr8k

# The Flickr8k Dataset

The dataset comprises 8000 images of various dimensions. The metadata folder of the dataset contains the following files:
- `CrowdFlowerAnnotations.txt` and `ExpertAnnotations.txt`: represent the crowd and expert versions of annotations against the captions, which judge to which extent the captions actually describe the provided images.
- `Flickr_8k.trainImages.txt`, `Flickr_8k.devImages.txt` and `Flickr_8k.testImages.txt`: the image names in each of the train-validation-test datasets. There are 6000 train, 1000 validation, and 1000 test images in total.
- `Flickr8k.token.txt` and `Flickr8k.lemma.token.txt`: the captions. Each image contains 5 captions from 5 different humans. The "lemma" version represents the lemmatised captions.

