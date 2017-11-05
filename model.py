import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

def preprocess_image(img):
    # original shape: 160x320x3, input shape for neural net: 66x200x3
    new_img = img[50:140,:,:]
    # apply gaussian blurring
    new_img = cv2.GaussianBlur(new_img, (3,3), 0)
    # scale to 66x200x3 
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    # convert to YUV color space
	#Images from cv2 imread  come in RGB format
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img
    

def get_training_data_info(excelFile,imgPath):   
    images = []
    measurements = []   
    image_paths = []
    #with open('./Training_uda/driving_log.csv') as csvfile:
    with open(excelFile) as csvfile: 
        reader = csv.reader(csvfile)
        for row in reader:
            steering_center = float(row[3])

            # create adjusted steering measurements for the side camera images
            correction = 0.05 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # read in images from center, left and right cameras
            path = imgPath #"./Training_uda/IMG/" # fill in the path to your training IMG directory
            filename  = row[0].split('/')[-1]
            image_paths.extend([path + filename])
            
            filename  = row[1].split('/')[-1]
            image_paths.extend([path + filename])
            
            filename  = row[2].split('/')[-1]
            image_paths.extend([path + filename])
            
            # add images and angles to data set
            measurements.extend([steering_center, steering_left, steering_right])
             
    return image_paths, measurements


def get_training_data(image_paths,measurements):
    images = []
    for filename in image_paths:
        image = cv2.imread( filename,1)
        image = preprocess_image(image)
        images.extend([image])
        
    X_train = np.array(images)
    y_train = np.array(measurements)
    
    return X_train, y_train
	
def augment_vertshift_shading(img, angle):
    new_img = img.astype(float)
    # random brightness - add a random value to the image
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:,:,0] + value) > 255 
    if value <= 0:
        mask = (new_img[:,:,0] + value) < 0
    new_img[:,:,0] += np.where(mask, 0, value)
   
    # random shadow - full height, random left/right side, random darkening
    h,w = new_img.shape[0:2]
    mid = np.random.randint(0,w)
    factor = np.random.uniform(0.3,0.8)
    if np.random.rand() > .5:
        new_img[:,0:mid,0] *= factor
    else:
        new_img[:,mid:w,0] *= factor
    
    # shift the horizon up/down 
    h,w,_ = new_img.shape
    horizon = 2*h/5
    v_shift = np.random.randint(-h/4,h/4)
    pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
    pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    new_img = cv2.warpPerspective(new_img,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
    
    return (new_img.astype(np.uint8), angle)


def augment_angle(img, angle):
    ''' 
    method for adding random distortion to dataset images, including random brightness adjust, and a random
    vertical shift of the horizon position
    '''
    new_img = img.astype(float)
   
    # randomly shift horizontal
    h,w,_ = new_img.shape
    horizon = 2*h/5
    h_shift = np.random.randint(-w/4,w/4)
    pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
    pts2 = np.float32([[0+h_shift,horizon],[w+h_shift,horizon],[0,h],[w,h]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    new_img = cv2.warpPerspective(new_img,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
    if h_shift>0:
        angle = angle+abs(h_shift)/10
    else:
        angle = angle-abs(h_shift)/10
        
    return (new_img.astype(np.uint8), angle)
	
#Load the data

images = []
angles = [] 
image_paths = []
  
image_paths, angles  = get_training_data_info('./Training_uda/driving_log.csv','./Training_uda/IMG/')
image_paths1, measurements1  = get_training_data_info('./Training_track2/driving_log.csv','./Training_track2/IMG/')
image_paths.extend(image_paths1)
angles.extend(measurements1)
print("Training data sample size:",np.shape(image_paths))

#Load the data
images, measurements = get_training_data(image_paths, angles)

aug_images, aug_measurements = [], []
for image, measurement in zip(images, measurements):
    aug_images.append(image)
    aug_measurements.append(measurement)
    
    if abs(measurement)>0.2:
        aug_images.append(cv2.flip(image,1))
        aug_measurements.append(measurement*-1.0)

        new_img, angle = augment_vertshift_shading(image, measurement)
        aug_images.append(new_img)
        aug_measurements.append(measurement)
        
X_train = np.array(aug_images)
y_train = np.array(aug_measurements)
print("Training data sample size after augmentation:",X_train.shape[0])


#Shuffle the data
X_train, y_train = shuffle(X_train, y_train)


#define the model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Dropout
from keras.layers import Lambda
from keras.layers import Convolution2D
from keras.regularizers import l2
from keras.layers.advanced_activations import ELU

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66,200,3)))
model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu',W_regularizer=l2(0.001), subsample=(2,2)))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu',W_regularizer=l2(0.001), subsample=(2,2)))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu',W_regularizer=l2(0.001), subsample=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu',W_regularizer=l2(0.001), subsample=(1,1)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu',W_regularizer=l2(0.001), subsample=(1,1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))


#Train the model
from keras.callbacks import ModelCheckpoint, EarlyStopping

#model.compile(loss='mse',optimizer='adam')
#model.fit(X_train,y_train,validation_split=0.2,shuffle = True, nb_epoch=5)

model.compile('adam', 'mean_squared_error', ['mean_squared_error'])
checkpoint = ModelCheckpoint("model.h5", monitor='val_mean_squared_error', verbose=1,
                                  save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=4,
                                verbose=1, mode='min')
model.fit(X_train, y_train, batch_size=64, nb_epoch=5, verbose=1,
                      callbacks=[checkpoint, early_stop], validation_split=0.15, shuffle=True)

