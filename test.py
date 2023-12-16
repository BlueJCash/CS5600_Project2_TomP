import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Loads the dataset. Change the path to match the location of the sample dataset
external_drive_path = 'D:\\sun397_testset'

datagen = ImageDataGenerator(rescale=1./255)

test_gen = datagen.flow_from_directory(
    external_drive_path,
    target_size=(244,244),
    batch_size=8,
    class_mode='sparse'
)

# Loads the saved model
loaded_model = tf.keras.models.load_model('sun397_model')

# Evaluates the loaded model on the test dataset
evaluation = loaded_model.evaluate(test_gen)
print("Test Accuracy: {:.2f}%".format(evaluation[1] * 100))
 
