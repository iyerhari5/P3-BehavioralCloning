# Behavioral Cloning

---

The goals / steps of this project are the following:
* Use deep neural networks and convolutional neural networks to clone driving behavior. 
* Train, validate and test a model using Keras. The trained model should output a steering angle to an autonomous vehicle.
* Use the simulator to test the performance of the model on two tracks.
 
[//]: # (Image References)

[image1]: ./Figures/Training-Distribution.png
[image2]: ./Figures/Data-augmentation.png
[image3]: ./Figures/Training-Distribution-After.png
[image4]: ./Figures/9-layer-ConvNet-model.png
[image5]: ./Figures/run1.gif
[image6]: ./Figures/run2.gif


## Writeup 
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

### Testing  Model on the test tracks

The results of the model on both tracks are shown below. 

Track 1:

![alt text][image5]

Track 2:

![alt text][image6]


### Testing for generalization

In order to test how well the model can generalize, I used only the training data from Track 2 to create the model. The only change was that I increased the data augmnetation to create more samples from the 
input data. The results were quite satisfactory and the model was able to perform quite well on Track 1 even though it had never seen the track before
in the training!

