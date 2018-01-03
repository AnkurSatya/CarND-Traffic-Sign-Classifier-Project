
# **Traffic Sign Recognition** 

## **Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Images/Input2703.png "Visualization"
[image2]: ./Images/Input1238.png "Visualization"
[image]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)



[image1]: ./Images/Input2703.png "Visualization"
[image2]: ./Images/Input1238.png "Visualization"
[image3]: ./Images/class_distribution.png "Data Distribution"
[image4]: ./Images/Before_grayscaling.png "Before grayscaling"
[image5]: ./Images/After_grayscaling.png "After grayscaling"
[image6]: ./Images/After_normalization.png "Normalized image"
[image7]: ./Images/Before_augmentation.png "Before Augmentation"
[image8]: ./Images/After_augmentation.png "After Augmentation"

## Data Set Summary & Exploration

### 1. A summary of the dataset

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

### 2. Here is a visualization of the images in the dataset.

Two images randomly selected from the dataset are shown below:
!['Traffic sign showing speed limit=30'][image1]!['A caution Traffic sign'][image2]

### 3. Let's see the data distribution among different classes:
!['A bar graph representing data distribution'][image3]

As evident in the above bar graph, distribution is not at all equal. Initially, I ran the LeNet5 model on the same datatset and achieved an accuracy of 89% on the validation set. 

To rectify the problem, I started with finding the per class accuracy which revealed that the classes with lesser data were very poorly classified with an accuracy of 20-50%. 

So, with this dataset, it would be hard if not impossible to improve the accuracy. But, we have a tool with us- data augmentation. 

But before that we need to preprocess the data.

## Preprocessing

### 1. Grayscaling

Why grayscaling? I had this question too. My first instinctive answer to this was that colors would not be any helpful in classifying the traffic signs of which most are red.
To test this, I ran a quick test on the grayscaled original data set using the LeNet5. It did improve the accuracy, overall and class wise. But, alone, it's not very helpful. Augmentation is required.

Before Normalizing, let's  how a our grayscaled image looks.

####                                                              Before grayscaling
!["Before grayscaling"][image4]
####                                                              After grayscaling
!["A grayscaled image of traffic sign"][image5]

### 2. Normalizing

I rescaled the dataset within the range of [-1,1]. It is really helpful as it negates the effect of a large value of pixel intensity which might have created a bias during testing.

Let's see what happens after noramlization.
!['A normalized image'][image6]

The difference after normalizing an image is more evident in a high contrasting image.
Now, we are good to go for Augmentation.

## Augmentation

There are a lot of methods to augment data. The one which I used are:

* Add random brightness
* Randomly warping- it was done using affine transformation.
* Random scaling- it is a perspective transformation.
* Random translation.

Library used -OpenCV.

The mean data points in each class was around 800. So, I used these augmentation methods on those classes only which had less than 800 data points.
And a point worth noting is that all of the above described augmentation methods were used on an image.

Let's compare an original image and a augmented image.

Original Image             |  Augmented Image
:-------------------------:|:-------------------------:
![image7]                  |![image8]

The differences between the two- 

Augmented image is more blurred, warping is also evident. Additionally, it seems like a zoomed in version of the original and slightly shifted to the right and bottom which is noticable because of the addded padding on the left and the top.

After augmentation total number of data points were **46480** and every class had a minimum of **800** data points. The distribution seemed better after this.

## Architecture Design

My final model was inspired by the work of Pierre Sermanet and Yann LeCun on Traffic sign.

It consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
|1.Convolution 5x5    	| 1x1 stride, same padding, outputs 28x28x32 	|
| Batch Normalization   | Mean=0, variance=1, offset=0, variance=1      |
| Leaky ReLu			| alpha=0.01						     		|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| 2.Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x64    |
| Batch Normalization   | Mean=0, variance=1, offset=0, variance=1      |
| Leaky ReLu            | alpha=0.01                                    |
| Max Pooling           | 2x2 stride, outputs 5x5x64                    |
|3.Convolution 3x3      | 1x1 stride, outputs 3x3x128                   |
| Batch Normalization   | Mean=0, variance=1, offset=0, variance=1      |
| Leaky ReLu            | alpha=0.01                                    |
| Max Pooling- Layer1   | Filter-5x5, stride-3x3 ->Flattening of Layer1 |
| Max Pooling- Layer1   | Filter-3x3, stride-1x1 ->Flattening of Layer2 |
|4.Fully connected-Input| concatenation of Layer-1, 2 and 3, size-2240  |
|5.Fully connected-Hidden|size-400          							|
| Leaky ReLu			| alpha=0.01   									|
| Dropout               | keep_prob=0.20                                |
|6.Fully connected-Output|size-43 									    |

### Training the network

#### Weight Initialization

When training, the weight initialization is an important aspect. Poorly initialized weights can not only slow down the learning process but even stop the network from learning altogether( This can happen when the activation function being used is ReLu and the weights initialization resulted in the negative outputs of the layers.)

So, the method I used was Xavier Initialization which is very effective in deciding the distribution of the parameters initial weights. It takes into account the number of input nodes for a layer and accordingly, parameterize the variance of the distribution. 

***Var(W)=1/no. of input nodes***

#### Batch Normalization

After the convoltuion process is done, the outputs are fed to the activation function. The scale of the outputs often goes unnoticed prior feeding to the activation function which in case of sigmoid or tanh activation functions creates a problem. If the value is to higher, the gradient saturates and the learning of that parameter stops. 

Batch normalization deals with these problem easily. It normalizes the output from the previous layer to a defined scale and the learning process progresses normally.
It also has an additional benefit of tackling overfitting as it normalizes the outputs by taking into account the whole batch at a time. It proves out to be a good regularizer.

#### Activation Function

Initially, I used ReLu activation function. Just to play around, I checked how many neurons were being killed by it. The results were not devastating but I thought of using Leaky ReLu to seek out the differences. 
And I did find something for myself. The accuracy improved.

Hence, I used leaky relu with an alpha of 0.01.

#### Optimizer

I used the Adam Optimizer. It uses the momentum method and also, it decreases the learning rate as the learning progresses. It solves a big problem- how to decay the learning rate? 

The way it decreases the learning rate is evident in the formula written below:

x+=-learning_rate*momentum/(sqrt(cache)+1e-7)      1e-7 is for non-zero denominator
where, 
cache=decay_rate*cache+((1-decay_rate)*square(dx))

The hyperparamter for momentum method-0.9
The decay rate used- 0.99

#### Other Hyperparameters

Learning rate- 0.0008
Batch Size- 128
Epochs- 30

### Different Architectures and Final Result

My final model results were:
* training set accuracy of 99.99
* validation set accuracy of 98.1 
* test set accuracy of 96.6

Initially, I tried to improve the LeNet5 architecture by using methods like Xavier initializer, batch normalization and by tweaking the learning rate. It did help to improve the accuracy but the gain was not substantial.

The model was learning at steady pace and performing accordingly on the validation and the test set. But still it always reached a plateau.

The problem I realized was that the model was not able to learn the complex features of the traffic sign images. This model worked on the MNIST data but traffic sign data was more complex. So, the larger number of filters might help. 
I increased the number of filters and the accuracy starts improving.

Then to improve it further, I borrowed the idea, of branching out the outputs from a convolutional layer and feeding it to the input layer of the fully connected network, from the work of ***Pierre Sermanet and Yann LeCun***.

It pushed the accuracy further up.

##### Why did I choose this architecture?

In a typical feed-forward to the subsequent layers only architecture, the inputs to the fully connected network is the high level abstractions of the images. I think this is quite useful for the problems where classes are very different from each other. But in our case, it was not so. 
Traffic signs have very less differentiating features from each other. At an abstract level, they might even get lesser differentiating. So, the idea of feeding the outputs of the low level layers to the input of the fully connected network seems effective and useful in our case.

To ensure the model was working well even after a good test accuracy, I checked the classifier's classification probability on top 3 classes. The significant difference between the various probabilities was enough to justify the selection of the model.


[image9]: ./Images/Online_Images/bumpy_road.jpeg
[image10]: ./Images/Online_Images/general_caution.jpeg
[image11]: ./Images/Online_Images/no_entry.jpeg
[image12]: ./Images/Online_Images/stop.jpeg
[image13]: ./Images/Online_Images/wild_animal_crossing.jpeg

### Testing the Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image9] ![alt text][image10] ![alt text][image11] 
![alt text][image12] ![alt text][image13]

### Results

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Bumpy Road Ahead      | Bumpy Road Ahead 							    |   
| No Entry				| No Entry										|
| General Caution    	| General Caution					 			|
| Wild Animal Crossing	| Wild Animal Crossing  						|


The model was able to correctly guess all of the images which leads to an accuracy of 100%. This is actually better than the test accuracy which I did previously. But, it is not a fair comaparison. These images were clicked in a comparatively better surrounding conditions than those provided in the test dataset.

### Softmax Probabilities

The code for making predictions on my final model is located in the **17**th cell of the Jupyter notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.61), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .61        			| Stop sign   									| 
| .24     				| Speed Limit(50 Km/h)							|
| .08					| Speed Limit(30 Km/h)							|
| .03	      			| Speed Limit(30 Km/h)					 	    |
| .02				    | Double curve  					     		|


For the second image, the model is pretty sure that this is a general caution sign with a probability of 0.99 and the image is of a general caution sign. The top five softmax probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99        			| General Caution 								| 
| .00     	     		| Pedestrians							        |
| .00					| Go Ahead or right							    |
| .00	    			| Traffic Signals				 	            |
| .00				    | Dangerous curve to the right 	         		|

For the third image, the model is 100% sure that the sign is of wild animals crossing which in fact it is.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0       			| Wild Animals Crossing  					    | 
| .00     				| Speed Limit(60 Km/h)							|
| .00					| Bicycles crossing							    |
| .00	      			| Slippery road					 	            |
| .00				    | Road Work 					     		    |

For the fourth image, the model is 99% sure that the sign is of No entry which in fact it is.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99        			| No entry  									| 
| .00     				| Stop						                    |
| .00					| No passing							        |
| .00	      			| End of all speed and passing limits		    |
| .00				    | Go straight or right				     		|

For the fifth image, the model is 56% sure that the sign is of Bumpy road which in fact it is.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .56       			| Bumpy road 									| 
| .27     				| Bicycles crossing							    |
| .08					| No vehicles							        |
| .06	      			| Traffic signals					 	        |
| .01				    | Road work					     		        |






```python

```
