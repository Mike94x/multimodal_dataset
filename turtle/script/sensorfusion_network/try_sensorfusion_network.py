#!/usr/bin/env python3
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from pprint import pprint
# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import gc



# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)
if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

#     tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1500)])


# def generator(subset):
#   i = 0
#   DIR = f"dataset_network/{subset}"
#   image_files = os.listdir(DIR+"/image")
#   depth_files = os.listdir(DIR+"/deptmap")
#   groundtruth_files = os.listdir(DIR+"/groundtruth")
#   image_files.sort()
#   depth_files.sort()
#   groundtruth_files.sort()
#   batch_x = [list(), list()] # I have two input: image + feature vector
#   batch_y = list() # output
#   for i in range (10):#(len(image_files)):
#     img_name = image_files[i]
#     image_file_path = f'{DIR}/image/{img_name}'
#     depth_name = depth_files[i]
#     depth_file_path = f'{DIR}/deptmap/{depth_name}'
#     groundtruth_name = groundtruth_files[i]
#     groundtruth_file_path = f'{DIR}/groundtruth/{groundtruth_name}'
#     i += 1
#     image = cv2.imread(image_file_path,cv2.IMREAD_UNCHANGED)
#     depth = cv2.imread(depth_file_path,cv2.IMREAD_UNCHANGED)
#     groundtruth = cv2.imread(groundtruth_file_path,cv2.IMREAD_UNCHANGED)
#     scale_percent = 60 # percent of original size
#     width = int(image.shape[1] * scale_percent / 100)
#     height = int(image.shape[0] * scale_percent / 100)
#     dim = (width, height)
#     image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) # resize image
#     depth = cv2.resize(depth, dim, interpolation = cv2.INTER_AREA) # resize depthmap
#     groundtruth = cv2.resize(groundtruth, dim, interpolation = cv2.INTER_AREA) # resize groundtruth
#     batch_x[0].append(image)
#     batch_x[1].append(depth)
#     batch_y.append(groundtruth)

#   batch_x[0] = tf.convert_to_tensor(batch_x[0])  # convert each list to array
#   batch_x[1] = tf.convert_to_tensor(batch_x[1])
#   batch_y = tf.convert_to_tensor(batch_y)
#   print(" GENERATION SUCCESSFUL")
#   return batch_x, batch_y


def generator(subset):
  DIR = f"dataset_network/{subset}"
  image_files = os.listdir(DIR+"/image")
  depth_files = os.listdir(DIR+"/deptmap")
  groundtruth_files = os.listdir(DIR+"/groundtruth")
  image_files.sort()
  depth_files.sort()
  groundtruth_files.sort()
  batch_x = [list(), list()] # I have two input: image + feature vector
  batch_y = list() # output
  for i in range (200):#(len(image_files)):
    img_name = image_files[i]
    image_file_path = f'{DIR}/image/{img_name}'
    depth_name = depth_files[i]
    depth_file_path = f'{DIR}/deptmap/{depth_name}'
    groundtruth_name = groundtruth_files[i]
    groundtruth_file_path = f'{DIR}/groundtruth/{groundtruth_name}'
    image = cv2.imread(image_file_path,cv2.IMREAD_UNCHANGED)
    depth = np.expand_dims(cv2.imread(depth_file_path,cv2.IMREAD_UNCHANGED),axis=-1)
    groundtruth = np.expand_dims(cv2.imread(groundtruth_file_path,cv2.IMREAD_UNCHANGED),axis=-1)

    # scale_percent = 60 # percent of original size
    # width = int(image.shape[1] * scale_percent / 100)
    # height = int(image.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) # resize image
    # depth = cv2.resize(depth, dim, interpolation = cv2.INTER_AREA) # resize depthmap
    # groundtruth = cv2.resize(groundtruth, dim, interpolation = cv2.INTER_AREA) # resize groundtruth
    batch_x[0].append(image)
    batch_x[1].append(depth)
    batch_y.append(groundtruth)

  batch_x[0] = np.array(batch_x[0])  # convert each list to array
  batch_x[1] = np.array(batch_x[1])
  batch_y = np.array(batch_y)
  print(" GENERATION SUCCESSFUL")
  return batch_x, batch_y



def display(display_list):
  plt.figure(figsize=(15, 15))
  title = ['Input Image','Input Depthmap', 'True Depthmap', 'Predicted Depthmap']
  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

def Train_model():
  
    # info,BATCH_SIZE,train_batches,STEPS_PER_EPOCH,test_batches=load_dataset()
    pointcloud_size=200 #200 punti per pointcloud (standardizzata)
    dropout_rate = 0.5
    # Image size that we are going to use
    IMG_SIZE = [480,640]
    # IMG_SIZE = [64,64]
    # Our images are RGB (3 channels)
    N_CHANNELS = 3
    # N_CLASSES=37
    input_size_img = (IMG_SIZE[0], IMG_SIZE[1], N_CHANNELS)
    input_size_depth = (IMG_SIZE[0], IMG_SIZE[1], 1)

    initializer = 'he_normal'

    # Ramo input immagini
    inputs1 = tf.keras.layers.Input(name='input_1',shape=input_size_img)
    conv_enc_11 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs1)
    # conv_enc_11 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding='same', kernel_initializer=initializer)(conv_enc_11)
    max_pool_12 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_enc_11)
    conv_enc_12 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_12)
    # conv_enc_12 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_12)
    max_pool_13 = tf.keras.layers.MaxPooling2D()(conv_enc_12)



    # Ramo input depthmap
    inputs2 = tf.keras.layers.Input(name='input_2',shape=input_size_depth)

    conv_enc_21 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs2)
    # conv_enc_21 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding='same', kernel_initializer=initializer)(conv_enc_21)
    max_pool_22 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_enc_21)

    conv_enc_22 = tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_22)

    # conv_enc_22 = tf.keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_22)
    
    max_pool_23 = tf.keras.layers.MaxPooling2D()(conv_enc_22)
   
    # SENSOR FUSION

    conc=tf.keras.layers.Concatenate()([max_pool_13, max_pool_23])
    # conv_enc_31 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conc)
    # conv_enc_31 = tf.keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_31)
    # max_pool_31 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_enc_31)
    # upsamp_32 = tf.keras.layers.UpSampling2D(size = (2,2))(max_pool_31)
    # conv_enc_32 = tf.keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(upsamp_32)
    # conv_enc_32 = tf.keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_32)
    upsamp_33 = tf.keras.layers.UpSampling2D(size = (2,2))(conc)
    conv_enc_33 = tf.keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(upsamp_33)
    # conv_enc_33 = tf.keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(upsamp_33)
    upsamp_43 = tf.keras.layers.UpSampling2D(size = (2,2))(conv_enc_33)

    conv_enc_43 = tf.keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(upsamp_43)
    # conv_enc_43 = tf.keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_43)
    # OUTPUT
    output = tf.keras.layers.Conv2D(1, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_43)

    # model
    model = keras.Model(inputs=[inputs1,inputs2], outputs=output, name="sensor_fusion")
    # model = keras.Model(inputs=inputs1, outputs=output, name="sensor_fusion")

    model.compile(optimizer='adam',
                    loss=tf.keras.losses.MeanAbsoluteError(),
                    metrics=['accuracy'])

    # model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True) 
    exit()
    length=len(os.listdir('/home/mike/catkin_ws/src/turtle/script/dataset_network/train/image'))
    batch_size=4
    step_epoch = 20 #length // batch_size
    # VAL_SUBSPLITS = 5
    # VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS
    
    checkpoint = 'checkpoint/cp.ckpt'
    checkpoint_path=os.path.join(os.path.dirname(__file__), checkpoint)
    if False :#os.path.isfile(os.path.dirname(__file__)+'/checkpoint/checkpoint'):
        model.load_weights(checkpoint_path).expect_partial()
        print("-------------------Assegno i pesi, non faccio training-------------------")
    else:

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)
        EPOCHS=20#int(input('Quante epoche?: '))
        x, y = generator('train')
        # dataset = generator('train')
        # print(" INPUT SHAPE: ", x.shape)
        x_val,y_val=generator('validation')
        gc.collect()
        model_history = model.fit(x, y,
                                validation_data=(x_val,y_val),
                                batch_size=batch_size, 
                                epochs=EPOCHS,
                                steps_per_epoch=step_epoch,
                                callbacks=[cp_callback]
                                )

    
    # loss = model_history.history['loss']
    # val_loss = model_history.history['val_loss']

    # plt.figure()
    # plt.plot(model_history.epoch, loss, 'r', label='Training loss')
    # plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss Value')
    # plt.ylim([0, 1])
    # plt.legend()
    # plt.show()
 
    return  model

if __name__ == '__main__':
  
  path_depth= '/home/mike/catkin_ws/src/turtle/script/dataset_network/train/deptmap/'
  path_image= '/home/mike/catkin_ws/src/turtle/script/dataset_network/train/image/'
  path1='/home/mike/catkin_ws/src/turtle/script/dataset_network/train/image'
  path2='/home/mike/catkin_ws/src/turtle/script/dataset_network/train/deptmap'
  path3='/home/mike/catkin_ws/src/turtle/script/dataset_network/train/groundtruth'
  # data_image=load_dataset(path1,img_size,'rgb')
  # data_depth=load_dataset(path2,img_size,'grayscale')
  # data_groundtruth=load_dataset(path3,img_size,'grayscale')
  # data=load_dataset('train')
  # print(data_depth)
  # train_data=create_train_data(path_image,path_depth,img_size)
  # print(len(train_data))
  # print(len(train_data[0])) 
  # print(np.shape(train_data[0][0])) #immagine (480, 640, 3)
  # print(np.shape(train_data[0][1])) #depthmap (480, 640)
  # exit()
  model=Train_model()
  img_name="/image/image50.png"
  deptmap_name="/deptmap/depthmap50.png"
  groundtruth_name="/groundtruth/groundtruth50.png"
  img_size=[480,640]
  directory = "/home/mike/catkin_ws/src/turtle/script/dataset_network/validation"
  img = tf.keras.utils.load_img((directory+img_name), target_size=(img_size[0], img_size[1]))
  img_array = tf.keras.utils.img_to_array(img)
  depth = tf.keras.utils.load_img((directory+deptmap_name), target_size=(img_size[0], img_size[1]), color_mode='grayscale',)
  depth_array = tf.keras.utils.img_to_array(depth)
  groundtruth = tf.keras.utils.load_img((directory+groundtruth_name), target_size=(img_size[0], img_size[1]), color_mode='grayscale',)
  groundtruth_array = tf.keras.utils.img_to_array(groundtruth)
  # plt.figure()
  # plt.imshow(tf.keras.utils.array_to_img(img_array))
  # plt.figure()
  # plt.imshow(tf.keras.utils.array_to_img(depth_array))
  # plt.show()
  img_array_with_batch = tf.expand_dims(img_array, 0)# Create a batch
  depth_array_with_batch = tf.expand_dims(depth_array, 0)# Create a batch
  predictions = model.predict([img_array_with_batch,depth_array_with_batch]) 
  p=tf.argmax(predictions)
  plt.figure()
  plt.title("Immagine")
  plt.imshow(tf.keras.utils.array_to_img(img_array))
  plt.figure()
  plt.title("Deptmap")
  plt.imshow(tf.keras.utils.array_to_img(depth_array))
  plt.figure()
  plt.title("Groundtruth")
  plt.imshow(tf.keras.utils.array_to_img(groundtruth_array))
  plt.figure()
  plt.title("Pred Deptmap")
  plt.imshow(tf.keras.utils.array_to_img(p))
  plt.show()
  # display([img_array, create_mask(predictions)])
  # create_depthmap(model)














