#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./images/barchart.png "Visualization"
[image2]: ./images/color_gray.png "Grayscaling"
[image3]: ./images/random_noise.jpg "Random Noise"
[image4]: ./images/2_speedlimit.png "Traffic Sign 1"
[image5]: ./images/12_1_PriorityRoad.png "Traffic Sign 2"
[image6]: ./images/13_1_Yield.png "Traffic Sign 3"
[image7]: ./images/13_2_yield.png "Traffic Sign 4"
[image8]: ./images/31_wildanimals.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/qthaole/CarND-TrafficSignClassification/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed over the labels.

![alt text][image1]
[image1]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because grayscaling reduces dimentionality and it's OK to do so beacause the color information should not have impact on the classification of traffic signs.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data. Normalization of input data helps make the training process more efficient. It makes the activation function work better and improves the optimization process (optima searching). Normalization also helps fit the fact that weights and bias are initialized with small values.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by shuffling the original training data first, then splitting it.

My final training set had 34208 number of images. My validation set and test set had 5000 and 12630 number of images.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the sixth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution       	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				 	|
| Convolution       	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flattening			| Outputs 400        							|
| Fully connected		| Outputs 120        							|
| RELU					|												|
| Fully connected		| Outputs 84        							|
| RELU					|												|
| Fully connected		| Outputs 43        							|

|						|												|
|						|												|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an EPOCH of 50, batch size of 128 and a learning rate of 0.001.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

I chose LeNet architecture to work with. It performs well in classifying the MNIST data, so I thought it should be a good start for the problem of traffic sign classification. I find that the architecture utilizes very well the concepts of weights and bias sharing, and pooling. Max pooling that is used in this implementaion of LeNet helps reduces computational cost by reducing the number of parameters to learn, while at the same time helps extract features of sub-regions.

However, I removed one fully-connected layer to make the model less heavy without compensating the overall acuracy.

My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.977
* test set accuracy of 0.895

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images I picked have different visual qualities (brightness, sharpness). Espcially, the yield sign that has small tree branches over could be difficult to predict.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

Here are the results of the prediction:
[12 13 14  3 31]
Correct labels:
[12, 13, 14, 2, 31]

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road      	| Priority road   								| 
| Yield     			| Yield 										|
| Stop					| Stop											|
| 50 km/h limit	      	| 60 km/h limit					 				|
| Wild animals crossing	| Wild animals crossing      					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is slightly less than, but still stays close to the accuracy on the test set of 0.895.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for displaying the top 5 softmax probabilities is located in the 18th cell of the Ipython notebook.

In most of the cases, the model is very sure of its predictions, with probability close to 1.

However, there is one numerical oddity in the softmax probabilities. That is, there are cases in which the sum of probabilities depasses 1. I think this is due to precision-related problem of calculations made on float numbers. Some numbers might accidentially be rounded to 1 and others to zero?

Details are presented below:

For the image of Priority road sign, the model is very sure that this is a Priority road sign (probability of 0.999). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999         			| Priority road   								| 
| .00000134     		| Yield 										|
| .0000000402			| No vehicles									|
| .0000000071      		| 50 km/h limit					 				|
| .0000000000691	    | Ahead only      								|


For the image of Yield sign:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Yield   										| 
| 0     				| 20 km/h limit 								|
| 0						| 30 km/h limit									|
| 0      				| 50 km/h limit					 				|
| 0	    				| 60 km/h limit      							|

For the image of Stop sign:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Stop   										| 
| 4.92779239e-14     	| No entry 										|
| 4.53347088e-19		| 30 km/h limit									|
| 3.80853238e-19      	| Roundabout mandatory					 		|
| 2.32363629e-20		| 80 km/h limit      							|

For the image of 50 km/h speed limit sign: The model is 87.8 percent sure that this is a 60 km/h limit sign while 12.9 percent sure that this is a 50 km/h limit sign. But it turned out to be 50 km/h limit sign instead.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.878         		| 60 km/h limit   								| 
| 0.129     			| 50 km/h limit 								|
| 2.32654904e-17		| 80 km/h limit									|
| 1.43869136e-25      	| 30 km/h limit					 				|
| 5.17113053e-26	    | No passing for vehicles over 3.5 metric tons  |

For the image of Wild animals crossing sign:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Wild animals crossing   						| 
| 4.14501333e-14     	| Dangerous curve to the left 					|
| 1.82951358e-14		| Double curve									|
| 3.09505442e-17      	| Slippery road					 				|
| 8.13392740e-20		| No passing for vehicles over 3.5 metric tons  |
