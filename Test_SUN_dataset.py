import tensorflow as tf
import tensorflow_datasets as tfds #Import used to load sun397 dataset
from keras import layers, models
from keras.applications import VGG19
 
def preprocess_data(sample): #Preprocces function due to fact that not all images in sun397 are of same size
    image = sample['image']
    label = sample['label']
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0  
    return image, label

external_drive_path = 'D:\\' #Loads model and sun397 into my external drive since sun397 is 36 GB

(ds_train, ds_validation, ds_test), ds_info = tfds.load( 
    #Used to load and split sun397 into training, validation, and testing sections
    name='sun397',
    split=['train', 'validation', 'test'],
    shuffle_files=True,
    with_info=True,
    as_supervised=False,
    data_dir=external_drive_path
)

#Alters the images in the splits so that they all have the same input shape
ds_train = ds_train.map(preprocess_data) 
ds_validation = ds_validation.map(preprocess_data)
ds_test = ds_test.map(preprocess_data)

batch_size = 32 #Controls the batch size for each of the shapes
ds_train = ds_train.batch(batch_size)
ds_validation = ds_validation.batch(batch_size)
ds_test = ds_test.batch(batch_size)

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False


#Creation of Model. Used Conv2D since the internet recommended that I use it for 2D image classification
model = models.Sequential()
model.add(base_model)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.01))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(397, activation='softmax')) 
#Used softmax at recommendation from device and internet since this is a multi-classification problem

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#Used sparse_categorical_crossentropy since mse resulted in a loss rate of around 50k

model.fit(ds_train,
          epochs=3, #Number of epochs
          validation_data=ds_validation)

model.save('D:\\sun397_model') #Saves the model
