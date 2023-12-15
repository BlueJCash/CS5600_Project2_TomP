import os
import tensorflow as tf
import tensorflow_datasets as tfds

def make_subdirectory(data):
    for sample in data:
        image_pic = sample['image'] #Image from ds_test
        label = sample['label']
        label_str = str(label.numpy()) #For clarity since actual label is confusing
        label_dir = os.path.join(test_directory, label_str)
        
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
            
        label_im_count = len(os.listdir(label_dir)) #Counts the current number of images in directory
        
        if label_im_count < 10:
            im_path = os.path.join(label_dir, f'image_{label_im_count}.jpg') #So that images don't continously overwrite
            image_pic = tf.image.encode_jpeg(image_pic)
            tf.io.write_file(im_path,image_pic)
            


external_drive_path = 'D:\\'

# Load the SUN397 dataset and get category information
(ds_train, ds_validation, ds_test), ds_info = tfds.load(
    name='sun397',
    split=['train', 'validation', 'test'],
    shuffle_files=True,
    with_info=True,
    as_supervised=False,
    data_dir=external_drive_path
)

test_directory = 'sun397_testset'

make_subdirectory(ds_test)
        
    

