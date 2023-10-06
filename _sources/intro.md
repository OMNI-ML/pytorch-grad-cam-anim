# Welcome to CAManim! 

CAManim is a Python package for animating the output of Class Activation Maps (CAMs) from Convolutional Neural Networks (CNNs). CAMs are a way of visualising the features that a CNN has learned to recognise in an image. CAManim allows you to create animations of CAMs. In this case we are animating through the layers of a CNN model. Other potential use cases are possible, such as animating through the epochs of a CNN training process.

## Abstract

Deep neural networks have been widely adopted in numerous domains due to their high performance and accessibility to developers and application-specific end-users. Fundamental to image-based applications is the development of Convolutional Neural Networks (CNNs), which possess the ability to automatically extract features from data. However, comprehending these complex models and their learned representations, which typically comprise millions of parameters and numerous layers, remains a challenge for both developers and end-users. This challenge arises due to the absence of interpretable and transparent tools to make sense of black-box models. There exists a growing body of Explainable Artificial Intelligence (XAI) literature, including a collection of methods denoted Class Activation Maps (CAMs), that seek to demystify what representations the model learns from the data, how it informs a given prediction, and why it, at times, performs poorly in certain tasks. We propose a novel XAI visualization method denoted CAManim that seeks to simultaneously broaden and focus end-user understanding of CNN predictions by animating the CAM-based network activation maps through all layers, effectively depicting from end-to-end how a model progressively arrives at the final layer activation. Herein, we demonstrate that CAManim works with any CAM-based method and various CNN architectures. Beyond qualitative model assessments, we additionally propose a novel quantitative assessment that expands upon the Remove and Debias (ROAD) metric, pairing the qualitative end-to-end network visual explanations assessment with our novel quantitative ``yellow brick ROAD" assessment (ybROAD). This builds upon prior research to address the increasing demand for interpretable, robust, and transparent model assessment methodology, ultimately improving an end-user's trust in a given model's predictions. Examples and source code can be found at: https://omni-ml.github.io/pytorch-grad-cam-anim/


## CAManim videos

We have generated a number of CAManim videos for different CNN architectures and datasets. These videos are available on this website below.

```{tableofcontents}
```

## Try it out!

You can try out CAManim in a Google Colab notebook by clicking the button below:

mednist + densenet --> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OMNI-ML/pytorch-grad-cam-anim/blob/adapt-basecam-to-support-cam_anim/tutorials/CAManim_mednist_tutorial.ipynb)

Animating End-to-End Network Actication Maps --> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OMNI-ML/pytorch-grad-cam-anim/blob/adapt-basecam-to-support-cam_anim/tutorials/_CAManim_animating_end2end_activation_maps.ipynb)