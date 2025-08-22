## Table of Contents

- [Demo](#demo)
- [Overview](#overview)
- [About the Dataset](#about-the-dataset)
- [Installation](#installation)
- [Deployement on Streamlit](#deployement-on-streamlit)
- [Directory Tree](#directory-tree)
- [Bug / Feature Request](#bug--feature-request)
- [Future Scope](#future-scope)

## Overview

This repository contains code for an image caption generation system using deep learning techniques. The system leverages a pretrained VGG16 model for feature extraction and a custom captioning model which was trained using LSTM for generating captions. The model is trained on the Flickr8k dataset using an attention mechanism to improve caption quality.

In addition to the base captioning model, the project integrates a Large Language Model (LLM) via the Together API (Mistral-7B-Instruct) to refine the generated captions, making them more natural, elegant, and human-like. This two-step approach improves the fluency and readability of captions.

- The key components of the project include:
- Image feature extraction using a pretrained VGG16 model
- Caption preprocessing and tokenization
- Custom captioning model architecture with attention mechanism
- Model training and evaluation
- Caption refinement using an LLM (Mistral via Together API) for enhanced natural language quality
- Streamlit app for interactive caption generation

## About the Dataset

The [Flickr8k dataset](https://www.kaggle.com/adityajn105/flickr8k) is used for training and evaluating the image captioning system. It consists of 8,091 images, each with five captions describing the content of the image. The dataset provides a diverse set of images with multiple captions per image, making it suitable for training caption generation models.

Download the dataset from [Kaggle](https://www.kaggle.com/adityajn105/flickr8k) and organize the files as follows:

- flickr8k
  - Images
    - (image files)
  - captions.txt

## Installation

This project is written in Python 3.10.12. If you don't have Python installed, you can download it from the [official website](https://www.python.org/downloads/). If you have an older version of Python, you can upgrade it using the pip package manager, which should be already installed if you have Python 2 >=2.7.9 or Python 3 >=3.4 on your system.
To install the required packages and libraries, you can use pip and the provided requirements.txt file.

## Directory Tree

```
|   app.py
|   image-captioner.ipynb
|   LICENSE.md
|   mymodel.h5
|   README.md
|   requirements.txt
|   tokenizer.pkl
```
