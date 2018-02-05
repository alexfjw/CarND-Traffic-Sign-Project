# **Traffic Sign Recognition**

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./histogram.png
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./b1.png
[image4]: ./b2.png
[o1]: ./examples/o1.png "Traffic Sign 1"
[o2]: ./examples/o2.png "Traffic Sign 2"
[o3]: ./examples/o3.png "Traffic Sign 3"
[o4]: ./examples/o4.png "Traffic Sign 4"
[o5]: ./examples/o5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of samples from each class in the test set. The distriubtion is not uniform.
Also, no class has an exceedingly small number of samples.

![alt text][image1]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because reducing dimensionality of features will reduce the computational resources required.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Next, I conducted histogram equalization on the images. There were many images with different brightness values, for instance, the below.
Histogram equalization flattens the intensity histogram of images, resulting in brightness values for all images.

Dark image
![alt text][image3]

Bright image
![alt text][image4]

Finally, I normalized the image vectors, subtracting the mean and then dividing by the standard deviation, a common preprocessing step for image classification.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following:

Network:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| PreLU					|												|
| Convolution 3x3     	| 1x1 stride,  outputs 32x32x32 				|
| PreLU | |
| Convolution 3x3     	| 1x1 stride,  outputs 32x32x64 				|
| PreLU		|      									|
| Maxpool |  2x2 stride, 3x3 filter, outputs 16x16x64 |
| Convolution 3x3     	| 1x1 stride,  outputs 16x16x80 				|
| PreLU					|												|
| Convolution 3x3     	| 1x1 stride,  outputs 16x16x192 				|
| PreLU					|												|
| Maxpool |  2x2 stride, 3x3 filter, outputs 8x8x192 |
|	FC					|			outputs 2048						|
|	Dropout				|	Dropout rate: 0.5 during train, 0 during test 	|
|	FC (output) 		|			outputs num classes					|

 A PreLU is a popular non-linearity with a trainable parameter. Like the ReLU, it clips negative activations. Exactly how much to clip is decided by the trainable parameter.


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer with cross entropy loss as the loss function. I also used a training rate with exponential decay to help with training.
Finally, I pick the weights which give the best performance on the validation set, as a form of early stopping.

Optimizer = Adam
Starting learning rate = 0.001  
Decay parameters: Decay to 0.96*current_LR every 30,000 steps.  
Batch size = 256
Epoch = 15


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.967
* test set accuracy of 0.948

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?   
The first model chosen was LeNet, to get a good feel of how tensorflow works.  

* What were some problems with the initial architecture?  
It was unable to perform well enough, even after adjusting learning rates and performing more complex preprocessing.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.  

I initially attempted to use residual connections like in ResNet, but had littel to no progress. I then tried to fit more convolutional layers, and expand the width of each layer after consulting popular networks like AlexNet & ResNet. Many research papers had made mention of the importance of depth of layers and width of layers. Increasing the two was a logical choice stemming from this research.  


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][o1] ![alt text][o2] ![alt text][o3]
![alt text][o4] ![alt text][o5]

The third image might be difficult to classify because the image of animal crossing is heavily pixelized, such that it no longer is distinguishable humans.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Index  | Actual Label  | Prediction  |
| ------------- |:-------------:| :-----:|
| 1 |  road work | road work |
| 2 |  priority road     |   priority road |
| 3 |  wild animals crossing |  priority road |
| 4 |  yield |    yield |
| 5 |  no vehicles |  yield |

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a road work sign (probability of 0.79), and the image does contain a road work sign. 

The top five soft max probabilities were as follows:
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .79         			| road work   									|
| .02     				| double curve 										| 
| .00					| beware of ice/snow										|
| .00	      			| right of way at next intersection					 				|
| .00				    | right turn       							| 

For the second image, the model is relatively sure that this is a priority road sign (probability of ~1), and the image does contain a road work sign.

The top five soft max probabilities were as follows:
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| priority road   								 
| .00     				| roundabout mandatory							|
| .00					| stop											|
| .00	      			| no entry 					 				|
| .00				    | speed limit 50       							|

For the third image, the model is relatively sure that this is a priority road sign (probability of 0.89), and the image is an animal crossing. As mentioned above, this is a tricky image due to pixelation. 

The top five soft max probabilities were as follows:
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .89         			| priority road      							|
| .02     				| ahead only 										|
| .01					| no entry											|
| .00	      			| dangerous curve to right  					 				|
| .00				    | general caution       							|

For the fourth image, the model is relatively sure that this is a yield sign (probability of ~1), and the image is a yield sign. 

The top five soft max probabilities were as follows:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| yield    									|
| .00     				| speed limit 60 										|
| .00					| speed limit 50											|
| .00	      			| right turn ahead 					 				|
| .00				    | end of speed limit 80       							|

For the fifth image, the model is not very confident that this is a yield sign (probability of .65), and the image is a no vehicles sign. This could be due to the similar white center of both signs. Data augmentation is likely to solve this problem.

The top five soft max probabilities were as follows:

| Probability         	|     Prediction	        					|
 |:---------------------:|:---------------------------------------------:|
| .65         			| yield  									|
| .28     				| end of speed limit 80  							|
| .00					| no passing 										|
| .00	      			| speed limit 60   					 				|
| .00				    | right turn ahead       							|
