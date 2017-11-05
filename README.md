# Behavioral Cloning

---

The goals / steps of this project are the following:

 *Use deep neural networks and convolutional neural networks to clone driving behavior. 
 *Train, validate and test a model using Keras. The trained model should output a steering angle to an autonomous vehicle.
 *Use the simulator to test the performance of the model on two tracks
 
[//]: # (Image References)

[image1]: ./Figures/Training-Distribution.png
[image2]: ./Figures/Data-augmentation.png

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup 
Here is a link to my [project code](https://github.com/iyerhari5/P3-BehavioralCloning)

Data Set Summary & Exploration

Training data on Track 1 was provided by Udacity and I found that to be quite sufficient to train the model. On Track2, I created training data with
two laps around the track. The training data consists of left, center and right camera images as well as the streering angle for the car to drive
given the current images. All three images were used for the traning. The steering angles for the left and right cameras were adjusted by subtracing/adding
a small value from the recorded steering angle. The total size of this basic training data was 58155 samples. 


The image below shows the distribution of the steering angles in this first training dataset.
![alt text][image1]

As we can see the data consists mainly of low steering angles centered around zero which corresponds to straight driving. In order to make the
dataset more balanced, I decided to augment the data. The following methods were used for this.
 
 1. Horizontal fillping of the images. The corresponding steering angle was multiplied by -1.
 2. Random brightness changes
 3. Random shading changes. Parts of the image were made brighter/darker
 4. Shifting of the horizon using a perspective transformation in the vertical direction

The figure below shows an input image and the image after being transformed with steps 2,3 and 4 mentioned above.
![alt text][image2]
 
In order to make the dataset more balanced with respect to the steering angles 


### Model Architecture 

The original images in the data set are color images of size 32x32. Based on results reported in the literature, I decided to
convert the images to grayscale as the first step. This helps to reduce the dimensionality of the input space. The images are then
normalized by a simple transformation to center the data.

image = (image-128.0)/128.0

Here is an example of a traffic sign image before and after grayscaling and normalization.

![alt text][image4]

Data Augmentation

As can be noted, the training set contains only around 35K images. In order to make the traning more generalizable, I decided to 
augment the data with samples generated from the training set itself. For this I implemented functions to add translation, rotation, zooming
and perspective projection on the images.

Here is an example of an original image and 4 more images generated with the described transformations from the original image.

![alt text][image5]

The augmented dataset hence should be more robust to differences in the pose of the camera, centering and rotation in the images 
presented to the neural network.



My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   			    	| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x20 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x20 				|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x36 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x36 					|
| Fully connected		| outputs 120        							|
| RELU					|												|
| Fully connected		| outputs 84        							|
| RELU					|												|
| Fully connected		| outputs 43        							|

### Training

To train the model, I used an Adam Optimizer. The training was done with 20 Epochs  and a batch size of 128. In order for the model to
generalize better, I used dropouts in the two fully connected layers before the output layer. The drop out probability was set to 0.5 during
the training.

My final model results were:
* training set accuracy of   :99.7%
* validation set accuracy of :98.3%
* test set accuracy of       :96.8%


The initial architecture I started with was the LeNet architecture. That gave around 94% validation accuracy without any data augnmentation. 
With the data augmnetaiton, the validation accuracy improved by ~2%. Fur further improvements, I added more complexity to the model by
increasing the number of features in the first and second convolutional layers. This resulted in increasing the validation set accuracy to ~98%

The model seems to generalize reasonably well giving ~97% accuracy on the test set.

### Testing  Model on New Images

Here are five German traffic signs that I found on the web that seem reasonably similar to images in the traning set.
	
![alt text][image6] 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h)							| 
| Speed limit (70km/h)  | Speed limit (70km/h)							| 
| Speed limit (80km/h)  | Speed limit (80km/h)							| 
| Go straight or right  | Go straight or right							|
| Slippery Road			| Wild animals Crossing  						|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 
This seems comparable to the 96.8% accuracy achieved on the test set.

Next we look at how confident the model was in making the predictions. For the first image, the model is very  
sure that this is a speed limit 30 km/h sign (probability of 1.0)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (30km/h)	 						| 
| ~0    				| Speed limit (50km/h)							|
| ~0					| Speed limit (70km/h)							|
| ~0	      			| Speed limit (20km/h)							|
| ~0				    | Yield   										|

The next four images also the model is very sure about the prediction with the most probable class having probability of ~1.0


### Visualizing the Neural Network 

The output of the first convolutional layer was visualized with the first traffic sign image from the web as the input. As can be seen from the
figure below, the layer seems to be activating on the edges of the speed limit letters and the circular outline.

![alt text][image7]