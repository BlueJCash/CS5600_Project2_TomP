import tensorflow as tf
import tensorflow_datasets as tfds

# Load the test dataset
external_drive_path = 'D:\\'

(ds_train, ds_validation, ds_test), ds_info = tfds.load(
    name='sun397',
    split=['train', 'validation', 'test'],
    shuffle_files=True,
    with_info=True,
    as_supervised=False,
    data_dir=external_drive_path
)

def preprocess_data(sample):
    image = sample['image']
    label = sample['label']
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0  
    return image, label

ds_test = ds_test.map(preprocess_data)
ds_test = ds_test.batch(8)

# Load the saved model
loaded_model = tf.keras.models.load_model('sun397_model')

# Evaluate the loaded model on the test dataset
evaluation = loaded_model.evaluate(ds_test)
print("Test Accuracy: {:.2f}%".format(evaluation[1] * 100))
 
