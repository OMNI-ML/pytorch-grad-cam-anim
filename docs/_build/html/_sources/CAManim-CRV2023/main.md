---
abstract: |
  Deep neaural networks have been widely adopted in numerous domains due
  to their high performance and increasing accessibility to developers
  and their application-specific end-user. Fundamental to image-based
  applications is the development and refinement of Convolutional Neural
  Networks (CNNs), which possess the ability to automatically extract
  features from data. However, comprehending these complex models and
  their learned representations, which typically comprise millions of
  parameters and numerous layers, remains a challenge for both
  developers and end-users. This challenge arises, in part, due to the
  absence of interpretable and transparent tools to make sense of
  black-box models. There exists a growing body of Explainable
  Artificial Intelligence (XAI) literature, including a collection of
  methods denoted as Class Activation Map (CAM), that seeks to demystify
  what representations the model learns from the data, how it informs a
  given prediction, and why it, at times, performs poorly in certain
  tasks. This work proposes a novel XAI visualization method denoted
  CAManim that seeks to simultaneously broaden and focus end-user
  understanding of CNN predictions by animating the CAM-based network
  activation maps through all layers, effectively depicting from
  end-to-end how a model progressively arrives at the final layer
  activation. Herein, we demonstrate that CAManim works with any
  CAM-based method and various CNN architectures. Beyond the qualitative
  model assessment focus of this work, we additionally propose a novel
  quantitative assessment that expands upon the Remove and Debias (ROAD)
  metric that considers pairs the qualitative end-to-end network visual
  explanations assessment with our novel quantitative
  "yellow-brick-ROAD\" assessment (ybROAD). This builds upon prior
  research to address the increasing demand for interpretable, robust,
  and transparent model assessment methodology ultimately generating
  increasing trust in a given end-user upon a model's predictions.
author:
- "`{ekaczmarek, kdick}@cheo.on.ca`"
bibliography:
- references.bib
title: "CAManim: Animating End-to-End Network Activation Maps"
---

::: IEEEkeywords
explainable artificial intelligence; convolutional neural networks;
classification
:::

# Introduction

The popularization of deep learning in numerous domains of research has
led to the rapid adoption of these methodologies in disparate fields of
scientific research. Convolutional Neural Networks (CNNs) are a class of
deep learning models that use convolutions to extract image features,
achieving high performance in numerous computer vision applications.
However, due to the intrinsic network structure and the complexity of
features leveraged for model predictions, CNNs are, consequently, often
labeled as uninterpretable or 'black-box; models. Interpretability is
crucial for applications in high-criticality fields such as medicine,
where model decisions have the potential to cause excessive harm if
incorrect. In order to be deployed, models must be trustworthy both in
their class predictions and in the features used to make those
predictions. Therefore, there is a definitive impetus to develop
trustworthy explanations of model decisions.

There have been numerous methods proposed to improve the
interpretability of CNNs. Zeiler and Fergus initially investigated
network interpretability by using a deconvolutional network to identify
pixels activated in CNN feature maps [@zeiler2014visualizing]. Next,
gradient-based methods were used to develop saliency maps indicating
important image regions based on desired output class
[@simonyan2013deep; @springenberg2014striving; @sundararajan2017axiomatic].
Class Activation Maps are a group of methods that linearly combine
weighted feature activation maps from a given CNN layer
[@zhou16; @gradcam; @gradcampp; @Fu20; @Gildenblat21; @Draelos20; @Jiang21; @Wang20; @Desai20; @eigencam].
Typically, only the final layer(s) are visualised to confer
trustworthiness and describe what image features are used for model
predictions. However, this provides little detail on the learning
process of the model. In addition, selecting the correct final layer to
visualize from each CNN model is not straightforward and is often done
arbitrarily.

To better interpret how a given model considers a given image through
each of its layers, we can individually visualize the model's layer-wise
activation map. In a natural extension of this idea, these layer-wise
activation maps can be combined as individual frames of a video
animating the end-to-end network activation maps; a method we propose in
this article and denote CAManim. We develop local and global
normalization to understand learned network features on a layer-wise and
network-wise scale. We experiment and quantify layer-wise performance of
CAManim with numerous CNN models and CAM variations to show performance
in a variety of experimental conditions, including medical applications.

Our contributions are as follows:

-   We propose CAManim, a novel visualization method that creates
    activation maps for each layer in a given CNN. CAManim can be
    applied to any existing CAM or CNN.

-   We introduce local and global normalization to understand important
    learned features on a layer-wise and network-wise level.

-   We perform extensive experimentation to determine the computational
    time and complexity required to run CAManim.

-   We demonstrate the usefulness of CAManim across multiple CAM
    variations and CNN models, and in high criticality fields.

-   We quantitatively evaluate the performance of each CAM generated per
    model layer with a metric termed ybROAD, improving the understanding
    of how CNNs learn. This is further extended to selecting the most
    accurate feature map representation from all possible layers of a
    CNN.

# Related Work

The topic of explainable and trustworthy AI has been researched
extensively. Lipton *et al.* [@lipton2018mythos] emphasized the need for
interpretable and trustworthy networks. Ribeiro *et al.*
[@ribeiro2016should] conducted studies to assess if humans can place
trust in a classifier. Computationally, numerous methods have
investigated the improvement of CNN interpretation. In this section, we
provide an overview of proposed methods and how CAManim addresses a gap
in the current literature.

*Earliest Explainable AI Studies:* One of the earliest efforts to
interpret CNNs was made by Zeiler and Fergus[@zeiler2014visualizing]. In
this study, feature maps from convolutional layers are used as input to
a deconvolutional network to identify activated pixels in the original
image space. Simonyan *et al.* [@simonyan2013deep] approached network
explainability in two ways. First, they proposed class models, which are
images generated through gradient ascent that maximize the score for a
given class. Next, they produced class-specific saliency maps,
calculated using the gradient of the input image with respect to a given
class.

*Guided Backpropagation and Gradient-Based Methods:* Springenberg *et
al.* [@springenberg2014striving] extended Simonyan's work to Guided
Backpropagation, which excludes all negative gradients to improve
quality of saliency maps. These works are compared in
[@mahendran2016salient]. Despite calculating gradients with respect to
individual classes, Selvaraju *et al.* showed that the visualizations
produced by Guided Backpropagation are not class-discriminative (*i.e.*
there is little difference between images generated using different
class nodes)[@gradcam]. Sundarajan *et al.* [@sundararajan2017axiomatic]
proposed integrated gradients, calculated through the integral of the
gradient between a given image and baseline, to satisfy axioms of
sensitivity and implementation invariance. FullGrad is another
gradient-based method that is non-discriminative and uses the gradients
of bias layers to produce saliency maps [@Srinivas19].

*Gradient-Free Methods:* While gradient-based methods are quite popular
in the field of explainable AI, some studies argue that these methods
produce noisy visualizations due to gradient saturation
[@adebayo2018sanity; @kindermans2019reliability]. For this reason,
gradient-free methods have been investigated by a number of studies.
Zhou *et al.* [@zhou2014object] identified *K* images with the highest
activation at a given neuron in a convolutional layer and occludes
patches of each image to determine the object detected by the neuron.
Morcos *et al.* [@morcos2018importance] used an ablation analysis to
remove individual neurons or feature maps from a CNN and quantify the
effect on network performance. This study demonstrated that neurons with
high class selectivity (*i.e.* highly activated for a single class) may
indicate poor network generalizability. Zhou *et al.*
[@zhou2018revisiting] extended this work to show that ablating neurons
with high class selectivity may cause large differences in individual
class performance.

*Class Activation Maps:* A popular class of CNN visualizations are Class
Activation Maps (CAMs), which produce explainable visualizations through
a linearly weighted sum of feature maps at a given CNN layer [@zhou16].
The original CAM was proposed for a specific CNN model, consisting of
convolutional, global average pooling, and dense layers at the end of
the network. The dense layer weights were used to determine the weighted
importance of individual feature maps. However, this required a specific
CNN architecture and was not applicable to numerous high-performing
models. This led to the development of CNN model-agnostic CAM methods.

Gradient-based methods were the first variation of the original CAM
[@gradcam; @gradcampp; @Fu20; @Gildenblat21; @Draelos20; @Jiang21] .
These methods determine importance weights through calculating averaged
or elementwise gradients of the output of a class with respect to the
feature maps at the desired layer. As discussed previously, gradient
methods may produce noisy visualizations due to gradient saturation
[@adebayo2018sanity; @kindermans2019reliability; @Wang20; @Desai20; @eigencam];
as a result, perturbation CAM methods have been proposed
[@Wang20; @Desai20] . In this case, importance weights are calculated by
perturbing the original input image by the feature maps A and measuring
the change in prediction score. In addition, non-discriminative
approaches have been investigated to eliminate the reliance of
class-discriminative methods on correct class predictions. EigenCAM
produces its CAM visualization using the principal components of the
activations maps at the desired layer [@eigencam].

While most studies have developed saliency map and/or CAM formulations
for a single layer, LayerCAM demonstrated how aggregating feature maps
from multiple layers can refine the final CAM visualization to include
more fine-detailed information [@Jiang21]. Gildenblat extended this idea
across existing multiple CAM and saliency map methods [@Gildenblat21].
However, to the best of our knowledge, our study is the first to save
individual feature maps generated from every CNN layer and combine them
into an end-to-end network explanation.

# Methodology

In this section, we first recall the general formulation for Class
Activation Maps and outline notation preliminaries. Next, we explain the
generation of CAManim using CAMs from each layer of a CNN, depicted in
Figure [\[fig:overview\]](#fig:overview){reference-type="ref"
reference="fig:overview"}. The concepts of global and local
normalization are introduced, and the computational complexity of
CAManim is described. Lastly, we define the quantitative performance
metric for individual CAM visualizations, and propose ybROAD for
analyzing end-to-end layerwise CAManim.

## Individual CAM Formulation

The general formulation for any CAM method consists of taking a linearly
weighted sum of feature maps and is written as follows:

$$\label{eqn:cam-form}
L^c_{CAM(A^l)} = \sum\limits_{k}(\alpha_k^cA_k^l),  \textit{where }  A^l= f ^l(x)$$

For a given input image *x* and CNN model *f*, a CAM visualization *L*
can be generated through the weighted $\alpha$ summation of *k*
activation feature maps *A* at layer *l*. Class discriminative CAM
methods further define *L* per predicted class *c*. To exclude negative
activations, most CAM formulations are followed by a ReLU operation.

## End-to-End Layerwise Activation Maps

To formulate CAManim, CAM visualizations are first generated for every
differentiable layer *l* within a given CNN with a total number of
layers *N*:

$$\label{eqn:cam-form}
L^c_{CAManim} = L^c_{CAM(A^{l=0})} ... L^c_{CAM(A^{l=N})}$$

Each CAM visualization is subsequently saved as a PNG image *I* and
concatenated together to create the final CAManim video, as depicted
below:

$$\label{eqn:cam-form}
CAManim = \mathop{\mathrm{\scalerel*{\Vert}{\sum}}}_{l}^{N}I_{L^c_{CAM(A^{l})}}$$

## Global- vs. Local-Layer Normalization

CAManim can visualize the CAM activations of each layer using two
different types of normalization. Global normalization is performed
using the minimum and maximum activation value across all activations
generated, which is useful for visualizing which layer has the strongest
activation for a given class and provides network-level information.
Local normalization uses the minimum and maximum values of the
activations of each specific layer. Local normalization displays the
strongest activation of each individual layer and therefore provides
layer-wise information.

Figure [1](#fig:normalization){reference-type="ref"
reference="fig:normalization"} shows the difference between global and
local normalization for the first denseblock of DenseNet161
`\cite{}`{=latex}. The global normalization (right) displays an
attenuated version of the local normalization (left). This example
demonstrates that the layer-wise information detecting learning small
features, whereas the network-wise information shows that the
activations of this layer are much smaller than other layers within
DenseNet161.

<figure id="fig:normalization">
<img src="figures/Bear_Normalization.png" style="width:40.0%" />
<figcaption>Difference between local and global normalization for the
feature map generated from layer features.denseblock1 in
DenseNet161.</figcaption>
</figure>

## Model- and CAM-Specific Interpretation

## Computational Complexity

<figure id="fig:densenet-params">
<embed src="figures/densenet161_num-params.pdf" style="width:50.0%" />
<figcaption>Layer-wise Depiction of DenseNet161 parameters.</figcaption>
</figure>

## Quantitative Evaluation

To quantitatively evaluate the performance of each CAM visualization and
demonstrate the information gained through deeper layers in a CNN, we
calculate the Remove and Debias (ROAD) score [@Rong22]. This metric has
superior computational efficiency and prevents data leakage found with
other CAM performance metrics [@Rong22]. ROAD perturbs images through
noisy linear imputations, blurring regions of the image based on
neighbouring pixel values. The confidence increase (or decrease) *C* in
classification score using the perturbed image with the least relevant
pixels *LRP* (or most relevant pixels *MRP*) is then used to evaluate
the accuracy of a CAM visualization. Since the percentage of pixels
perturbed affects the ROAD performance, we evaluate ROAD at *p =* 20%,
40%, 60% and 80% pixel perturbation thresholds. As proposed by
Gildenblat [@Gildenblat21], we combine the least relevant pixel and most
relevant pixel scores for our final metric:

$$ROAD(L^c_{CAM(A^{l})}) = \sum\limits_{p} \frac{(C^p_{L_{LRP}} – C^p_{L_{MRP}})}{2}$$

A ROAD score is calculated for each CAM generated. Therefore, for *N*
differentiable layers in a CNN, there will be *N* ROAD scores calculated
for CAManim. We denote this series of ROAD values as the 'yellow brick
ROAD', or ybROAD for short.

$$\label{eqn:cam-form}
ybROAD = \mathop{\mathrm{\scalerel*{\Vert}{\sum}}}_{l}^{N}ROAD(L^c_{CAM(A^{l})})$$

The ybROAD metric can be used to analyze performance of an experiment
with given class, image, and CNN model over all layers of the network.
In this study, we identify the CNN layer that identifies features with
the largest impact on model performance through ybROAD\_max. The
ybROAD\_mean score is also calculated to summarize the overall ROAD
performance of a model.

::: figure*
![image](figures/conceptual-overview.pdf){width="\\textwidth"}
:::

# Results & Discussion

In this section, we first define the pre-trained models and datasets
used to evaluate CAManim in Section
[4.1](#sec:datamodel){reference-type="ref" reference="sec:datamodel"}.
Next, we demonstrate CAManim in high criticality fields using a ResNet50
model fine-tuned to perform breast cancer classification in Section
[4.2](#sec:bcresnet){reference-type="ref" reference="sec:bcresnet"}. We
then show example visualizations from CAManim for 10 different CAM
variations in Section [4.3](#sec:activationmaps){reference-type="ref"
reference="sec:activationmaps"} and discuss abnormal visualizations in
Section [4.4](#sec:visissues){reference-type="ref"
reference="sec:visissues"}. Lastly, we discuss the ybROAD performance of
CAManim in Section [4.5](#sec:ybROAD){reference-type="ref"
reference="sec:ybROAD"}.

## Pre-trained Models and Datasets {#sec:datamodel}

To evaluate CAManim, we use models from TorchVision pre-trained on the
2012 ImageNet-1K dataset. Specifically, results are shown for AlexNet
`\cite{}`{=latex}, ConvNext `\cite{}`{=latex}, DenseNet161
`\cite{}`{=latex}, EfficientNet-b7 `\cite{}`{=latex}, MaxViT-t
`\cite{}`{=latex}, and SqueezeNet `\cite{}`{=latex}. The CAManim videos
for an additional 14 models can be found here:
<https://omni-ml.github.io/pytorch-grad-cam-anim/>. All results in this
study (apart from the high criticality evaluation) are based on a
popular image used in CAM evaluations, preprocessed by resizing to 224 x
224 and normalizing the final image.

Next, we demonstrate the utility of CAManim in high criticality fields.
Specifically, we take a ResNet50 model pre-trained on the ImageNet
dataset, and fine-tune the model using the Kaggle breast ultrasound data
to classify malignant vs normal images `\cite{}`{=latex}. For
simplification, we call this network BC-ResNet50 (Breast Cancer
-ResNet50). This dataset consists of 133 normal images and 210 malignant
images, which are split into a 80-10-10% train-validation-test split.
Images are preprocessed to a size of 224 x 224 and augmentations are
applied to the training set. Preprocessing and training steps are
selected based on MONAI recommendations `\cite{}`{=latex}. After
fine-tuning the network, CAManim is run with an example test image of
the malignant class to understand how the CNN makes a correct prediction
of cancer.

## Case Study: End-to-End BC-ResNet50 Visualization for Malignant Tumour Prediction {#sec:bcresnet}

Figure [\[fig:mednet-viz\]](#fig:mednet-viz){reference-type="ref"
reference="fig:mednet-viz"} illustrates the layerwise activations that
BC-ResNet50 considers when determining the 'malignant' tumour.

::: figure*
![image](figures/mednet-10percentile_final.pdf){width="\\textwidth"}
:::

## Visualizing End-to-End Network Activation Maps {#sec:activationmaps}

We demonstrate the performance of CAManim on 10 different CAM methods,
including seven gradient-based methods (EigenGradCAM, GradCAM,
GradCAMElementWise, GradCAM++, HiResCAm, LayerCAM, and XGradCAM), two
perturbation methods (AblationCAM and ScoreCAM), a principal components
method (EigenCAM), and RandomCAM. RandomCAM generates random feature
activation maps from a uniform distribution between -1 and 1.

::: figure*
![image](figures/bear_one-model-all-cams_final.pdf){width="\\textwidth"}
:::

<figure id="fig:my_label">
<embed src="figures/bear_one-cam-all-models_final.pdf"
style="width:50.0%" />
<figcaption>Initial, middle, and final activation maps applying a single
CAM, HiResCAM, to various model architechtures.</figcaption>
</figure>

## Layer-Type Visualization Issues {#sec:visissues}

Certain differentiable layers may produce unanticipated CAM
visualizations, as depicted in Figure
[4](#fig:bad_layers){reference-type="ref" reference="fig:bad_layers"}.
In these layers, images are compressed to 1-dimensional (1D)
representation, preventing valid 2D features from being discovered
through CAM formulations. Instead, individual neurons that are highly
activated show up as vertical or horizontal lines over the image. While
these images are not informative, they are not incorrect; they simply
depict visualizations of 1D layers.

<figure id="fig:bad_layers">
<img src="figures/bad_layers.png" style="width:40.0%" />
<figcaption>Visualization of CAManim for fully connected and average
pooling layers.</figcaption>
</figure>

## ybROAD Quantitative Evaluation {#sec:ybROAD}

Figure X(YBROAD) displays the ybROAD for X trials of generating CAManim
for the bear image using ResNet152. Initially, the individual-layer ROAD
performance is very high ($\sim$`<!-- -->`{=html}0.45). At this point,
the CNN layer is activating many small regions throughout the image;
when each of these areas is perturbed, it is difficult to correctly
classify the image, and the ROAD score increases. As the network starts
learning larger features, less of the bear image is perturbed, and the
ROAD score decreases. Towards the end of the network, the ROAD score
increases again as the small activations are combined together to cover
the entire bear. This demonstrates how the ybROAD score can provide
additional information on how the network is learning.

# Conclusion

The conclusion goes here. this is more of the conclusion

# Acknowledgment {#acknowledgment .unnumbered}

The authors would like to thank\... more thanks here
