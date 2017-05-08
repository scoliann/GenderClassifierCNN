This script uses a Convolutional Neural Network to classify photos of men and women.

## Inspiration

I first learned about CNNs working through the [TensorFlow and deep earning, without a PhD](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0) tutorial on Codelabs.  This tutorial was useful, but demonstrated that there was quite a bit of overhead to creating good Convolutional Neural Networks from scratch in TensorFlow.  Later, I watched Siraj Raval's video [Build a TensorFlow Image Classifier in 5 Min](https://www.youtube.com/watch?v=QfNvhPx5Px8), and learned about Transfer Learning and TensorFlow's Inception model.  Interesting classification problems are abundant, and I knew that I had to experiment with Inception and Transfer Learning as it could prove to be a valuable tool in the future.

## About Inception and Transfer Learning

The Inception-v3 model is a CNN built by Google to compete in the ImageNet competition.  Inception-v3 is therefore natively trained to classify input images into one of 1,000 categories used in ImageNet.  Transfer Learning is a method by which one retrains part of a pre-existing machine learning model for a new purpose.  In this case, we will be retraining the bottleneck layer of Inception-v3. The "bottleneck" layer is the neural network layer before the final softmax layer in the CNN.  Only one layer of the Inception-v3 model needs to be retrained, as the previous layers have already learned useful, generalizable functions such as edge detection. 

## Acquiring Data

All data for this project was acquired using the Chrome extension [Fatkun Batch Download Image](https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/nnjjahlikiabnchcpehcpkdeckfgnohf?hl=en).  This extension lets one download all images displayed on a webpage, making it a versatile tool for building datasets from Google Images, or other image sources.

## The Dataset

For this project, I built a dataset using the aforementioned tool with the following specifications:
- The training set consisted of 1,000 photos of men’s faces and 1,000 photos of women’s faces.
- The test set consisted of 200 photos of men’s faces and 200 photos of women’s faces.

As some images that are freely accessible online may require payment or royalties, I have not included my dataset in this repository.

## Running the Script

To run the script genderClassification.py, one must first do the following:

1.  Create a dataset of men and women on which to test the CNN. 
2.  Place the photos of men in the `testData\male` folder, and the photos of women in the `testData\female` folder.

## Results

This Convolutional Neural Network achieved 86.5% accuracy in classifying images of male and female faces.  

The fundamental facial composition for men and women is the same:  We both have two eyes, one nose, one mouth, etc.  The number of shared attributes leads me to believe that gender-based facial classification is a difficult task, and as such, I am happy with the accuracy achieved.  This is an excellent example of how machine learning models can learn abstract concepts that would be difficult to program from the ground up.

## Potential Improvements

During the course of this project, the following occurred to me as potential improvements that may increase the accuracy of the CNN:

1.  Modify Inception-v3's retrain.py file to incorporate dropout.  This would allow one to retrain the bottleneck layer with an increased number of training steps without overfitting to the relatively small training dataset.
    - This was tested with a dropout of 0.75.  Unfortunately, the classification accuracy performed just as well or slightly worse.
2.  Increase the size of the training set.  500 images for each class is a relatively small size for a machine learning problem.  Perhaps adding more images before retraining would result in greater classification accuracy.
    - I tested this hypothesis with image sets five times larger and only achieved accuracies in the high 70's and low 80's.  Admittedly, my larger training sets were not as well cleaned as the initial training set.
3.  Apply Haar Cascades to directly parse out the area of each picture containing a face.  The images in the resulting dataset will be of faces only (ie. no rest of the body, background, etc.), and retrain Inception-v3 with this cleaned dataset.

## Resources

In addition to Siraj Raval's aforementioned [video](https://www.youtube.com/watch?v=QfNvhPx5Px8), the [TensorFlow for Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html?index=..%2F..%2Findex#0) tutorial on Codelabs is an extremely good resource for learning to work with TensorFlow's Inception model.







