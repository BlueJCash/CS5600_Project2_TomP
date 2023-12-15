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

#Alteres the images in teh splits so that they all have the same input shape
ds_train = ds_train.map(preprocess_data) 
ds_validation = ds_validation.map(preprocess_data)
ds_test = ds_test.map(preprocess_data)

batch_size = 32 #Controls the batch size for each of the shapes
ds_train = ds_train.batch(batch_size)
ds_validation = ds_validation.batch(batch_size)
ds_test = ds_test.batch(batch_size)

loaded_model = tf.keras.models.load_model('D:\\sun397_model')

loaded_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

loaded_model.fit(ds_train,
          epochs=2, #Number of epochs
          validation_data=ds_validation)

loaded_model.save('D:\\updated_sun397_model')