# Pneumonia-detection-using-deep-learning
```
Domain             : Computer Vision, Machine Learning
Sub-Domain         : Deep Learning, Image Recognition
Techniques         : Deep Convolutional Neural Network, ImageNet, Inception
Application        : Image Recognition, Image Classification, Medical Imaging
```
## Dataset
```
Dataset Name     : Chest X-Ray Images (Pneumonia)
Dataset Link     : [Chest X-Ray Images (Pneumonia) Dataset (Kaggle)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
                 : [Chest X-Ray Images (Pneumonia) Dataset (Original Dataset)](https://data.mendeley.com/datasets/rscbjbr9sj/2).
Original Paper   : [Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5).
                   (Daniel S. Kermany, Michael Goldbaum, Wenjia Cai, M. Anthony Lewis, Huimin Xia, Kang Zhang)
                   https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
```      

```
Dataset Details
Dataset Name            : Chest X-Ray Images (Pneumonia)
Number of Class         : 2
Number/Size of Images   : Total      : 5856 
                          Training   : 3516 
                          Validation : 1170 
                          Testing    : 1170

Model Parameters
Machine Learning Library: Keras
Optimizers              : Adam
Loss Function           : categorical_crossentropy

For Custom Deep Convolutional Neural Network : 
Training Parameters
Batch Size              : 64
Number of Epochs        : 45
Training Time           : about 20 - 25 minutes (using GBPU)

Output (Prediction/ Recognition / Classification Metrics)
Training set loss       : 0.1637
Training set accuracy   : 0.9332 (93.32%)
Validation set loss     : 0.1075
Validation set accuracy : 0.9162 (91.62%)
```

```
Detailed results
[confusion matrix]https://github.com/rohanchoudharyy/Pneumonia-detection-using-deep-learning/blob/master/confusion%20matrix.jpg

[Results]https://github.com/rohanchoudharyy/Pneumonia-detection-using-deep-learning/blob/master/results.jpg
