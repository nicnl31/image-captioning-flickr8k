# Image captioning with the Flickr8k Dataset

This project attempts to train an encoder-decoder architecture to predict captions for images using the Flickr8k dataset. The original dataset is not uploaded due to size, but there are many versions available on the Internet. For example, one can be found on Kaggle: https://www.kaggle.com/datasets/nunenuh/flickr8k

# The Flickr8k Dataset

The dataset comprises 8000 images of various dimensions. The metadata folder of the dataset contains the following files:
- `CrowdFlowerAnnotations.txt` and `ExpertAnnotations.txt`: represent the crowd and expert versions of annotations against the captions, which judge to which extent the captions actually describe the provided images.
- `Flickr_8k.trainImages.txt`, `Flickr_8k.devImages.txt` and `Flickr_8k.testImages.txt`: the image names in each of the train-validation-test datasets. There are 6000 train, 1000 validation, and 1000 test images in total.
- `Flickr8k.token.txt` and `Flickr8k.lemma.token.txt`: the captions. Each image contains 5 captions from 5 different humans. The "lemma" version represents the lemmatised captions.

<img width="716" alt="Screenshot 2024-06-01 at 1 15 03 PM" src="https://github.com/nicnl31/image-captioning-flickr8k/assets/86213993/825d001d-54e6-47fa-bcd4-f627f2769584">

# Preparation

## Training preparation
The following steps are preformed to prepare the data for training:
- Resize all images to standardised `224x224x3`.
- A `Vocabulary` object is created, which represents the "dictionary" for a model. It is essentially a two-way bijective mapping between the set of individual tokens that make up the whole corpus and the set of corresponding indices. This corpus also includes special tokens for training and evaluation purposes, which are the beginning-of-sentence `<BEG>`, end-of-sentence `<END>`, padding `<PAD>`, and null `<UNK>`.
- For transfer learning: I use stratified random sampling to sample 40% of the dataset for faster training and model selection.

## Model selection
The `NASNetA-Large` is chosen to be the encoder component of the architecture. [1] For the decoder, the `LSTM` [2], [3] and the `GRU` [3] take turns to be tested, and the best decoder is chosen based on the architecture's performance in the validation loss.

The following are fixed hyperparameters:
- Embedding dimension: 256
- Hidden size dimension: 256

The following are chosen during transfer learning:
- Decoder: {`LSTM`, `GRU`}
- Number of layers: {1, 2}

## Training parameters
- Batch size: 128
- Total epochs: 300, with early stopping if validation loss does not improve for 10 epochs.
- Learning rate: `1e-4`, with decay rate `1e-1` if validation loss does not improve. Decay until the learning rate hits `1e-6`.
- Optimizer: `Adam`


## Evaluation parameters
- Loss function: Cross Entropy Loss as the objective function to minimise during the train-validation loop.
- Quantitative assessment of captions: the BLEU-n score, a popular metric for automatic machine translation evaluation [1], provides a simple and non-costly method to quantitatively assess the out-of-sample performance of the architecture. The idea of BLEU is to assess the similarity of the predictions and the ground truth in terms of the number of contiguous n-grams correctly seen in the predictions, compared to the ground truth. Four versions of the BLEU-n score, for each $n \in \{1, 2, 3, 4\}$, are used.

# Results

For the RNN component, the GRU with 1 layer is winning during transfer learning. Therefore, the `NASNetA-Large+GRU` is chosen for fine-tuning.

The `NASNetA-Large+GRU` achieves a train loss of `2.4760`, validation loss of `3.1421` in transfer learning, using 40% of the dataset. 

In fine-tuning, using the complete dataset, it achieves a train loss of `2.320` and validation loss of `2.8997`, which is a modest improvement and not much overfitting. 

The out-of-sample BLEU-1 to BLEU-4 metrics are `0.5065 0.3253 0.2015 0.1298`. This is a fairly impressive result given that the models are trained without any advanced features, such as attention. 

![pred1](https://github.com/nicnl31/image-captioning-flickr8k/assets/86213993/256bddbf-a452-46a5-8c6e-d693dc42cc7d)

## The good
- The captions are generated in a fairly human-natural tone when considered by themselves, which suggests that the model learned human speech well.
- A good generalisation capability is observed since there is not much overfitting. This can be seen in the narrow space between training and validation losses.
- The model recognises dogs and different types of humans (boy/girl, woman/man) fairly well in most situations. This is possibly due to NASNet's original ImageNet dataset that it was trained on.

## The rooms for improvement
- The characteristics of each object are still wrong in some instances, e.g. garment colours, surroundings, objects' actions.
- Some captions are found to be stuck in a loop, e.g. "a man in a black shirt and a woman in a black shirt and a woman in a black shirt and a woman in a black shirt and..."

# Future work

- Advanced language models: experiment with more advanced language models for the decoder, such as transformer architectures (BERT, GPT).
- Attention mechanism: Implementing attention mechanisms can allow the model to focus on specific parts of the image when generating each word in the caption.
- Fine-tuning: Experiment with different fine-tuning strategies for pre-trained models on the specific dataset, such as unfreezing more layers or adjusting learning rates for different parts of the network.
- Data augmentation: Use augmentation techniques to increase the diversity of the training data and reduce overfitting, like random cropping, flipping, blurring, changing colour hues, etc.

# References
[1] Barret Zoph, V. K. Vasudevan, J. Shlens, and Q. V. Le, “Learning Transferable Architectures for Scalable Image Recognition,” arXiv (Cornell University), Jul. 2017, doi: https://doi.org/10.48550/arxiv.1707.07012

[2] Staudemeyer, Ralf C and E. R. Morris, “Understanding LSTM -- a tutorial into Long Short-Term Memory Recurrent Neural Networks,” arXiv (Cornell University), Sep. 2019, doi: https://doi.org/10.48550/arxiv.1909.09586

[3] J. Chung, C. Gulcehre, K. Cho, and Y. Bengio, “Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling,” arXiv.org, Dec. 11, 2014. doi: https://doi.org/10.48550/arXiv.1412.3555. Available: http://arxiv.org/abs/1412.3555
