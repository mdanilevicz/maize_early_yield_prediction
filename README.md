# Crop early yield prediction using multimodal deep learning model

Deep learning model based on a spectral module (with resnet18 backbone for image data) and a DNN module (for extracting features of the tabular data comprising genotype information and crop management metadata).

Cite the paper [Maize Yield Prediction at an Early Developmental Stage Using Multispectral Images and Genotype Data for Preliminary Hybrid Selection](https://doi.org/10.3390/rs13193976) by Monica F. Danilevicz, Philipp E. Bayer, Farid Boussaid, Mohammed Bennamoun,and David Edwards.


## Model Goal

![Graphical abstract of the paper ](https://github.com/mdanilevicz/maize_early_yield_prediction/blob/main/image_md/GA.png)

## Model architecture

We applied a multimodal deep learning model for early crop yield prediction, using crop information in table format to train the first module and multispectral images from the plants growing in the field. Both modules with the multimodal framework are described in "multimodal_model.ipynb".
Each module in the frameworks outputs its own yield prediction, which is helpful to identify if there is any group that the modules has a higher difficulty to predict the yield. The dataset used to train the model was collected by researchers at TAMU University and are available at doi: 10.25739/4ext-5e97, 10.25739/96zv-7164 and 10.25739/d9gt-hv94.

![Multimodal model framework ](https://github.com/mdanilevicz/maize_early_yield_prediction/blob/main/image_md/multimodal.png)


Note: We are updating this repository, make a pull request if there is anything you think is missing after looking here and the paper.


