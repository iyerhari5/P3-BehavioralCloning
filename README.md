# Behavioral Cloning

---

The goals / steps of this project are the following:

 *Use deep neural networks and convolutional neural networks to clone driving behavior. 
 *Train, validate and test a model using Keras. The trained model should output a steering angle to an autonomous vehicle.
 *Use the simulator to test the performance of the model on two tracks
 
[//]: # (Image References)

[image1]: ./Figures/Training-Distribution.png
[image2]: ./Figures/Data-augmentation.png
[image3]: ./Figures/Training-Distribution-After.png
[image4]: ./Figures/9-layer-ConvNet-model.png


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup 
Here is a link to my [project code](https://github.com/iyerhari5/P3-BehavioralCloning)

Data Set Summary & Exploration

Training data on Track 1 was provided by Udacity and I found that to be quite sufficient to train the model. On Track2, I created training data with
two laps around the track. The training data consists of left, center and right camera images as well as the streering angle for the car to drive
given the current images. All three images were used for the traning. The steering angles for the left and right cameras were adjusted by subtracing/adding
a small value from the recorded steering angle. The original images size of 160x320x3 was converted to a size of 66x200x3.  Also the image format
was converted to YUV color space. The total size of this basic training dataset was 58155 samples. 


The image below shows the distribution of the steering angles in this first training dataset.
![alt text][image1]

As we can see the data consists mainly of low steering angles centered around zero which corresponds to straight driving. In order to make the
dataset more balanced, I decided to augment the data. The following methods were used for this.
 
 1. Horizontal flipping of the images. The corresponding steering angle was multiplied by -1.
 2. Random brightness changes
 3. Random shading changes. Parts of the image were made brighter/darker
 4. Shifting of the horizon using a perspective transformation in the vertical direction

The figure below shows an input image and the image after being transformed with steps 2,3 and 4 mentioned above.
![alt text][image2]
 
In order to make the dataset more balanced with respect to the steering angles, the transformation above were applied only if the absolute 
value of the original image steering angle was greater than 0.2 degrees. The figure below shows the distribution of the training dataset
after the above data augmentation steps.
![alt text][image3]


### Model Architecture 

The model used is the model from the NVIDIA autonomous vehicle group. The figure below shows the architecture.
![alt text][image4]

The image intensites are normalized in the first layer of the keras model. Also as mentioned previously the RGB images were converted to YUV space
as recommended by NVIDIA for use with the selected model.

Inorder for the model to generalize better, I used the L2 normalization for the convolutional layers. Droputs in the fully connected layers was tried,
but the performance seemed to degrade. So it was not used for the final model.


### Training

To train the model, I used an Adam Optimizer. The mean squared error(MSE) was used as the loss function and training was done for 5 epochs. Callbacks
were implemented to save the model only when the MSE on the validation set improved from the previously saved verison. An early stopping callback 
was also implemented if the MSE did not improve. However for the small number of epochs used here, early termination never occured.



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