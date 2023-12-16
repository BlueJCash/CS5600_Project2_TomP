# CS5600_Project2_TomP
Tom Prouty's GitHub Repository for the Final Project of CS5600

__IMPORTANT: Do not run the Python files containing tfds. Running these files yourself will cause the system to try and download
the SUN397 dataset onto your computer, which is only possible if you have an available 36 GB minimum of space in memory.__

The goal of this project was to create a multi-category image classification ConvNet. To do this the TensorFlow
SUN397 dataset, Scene UNderstanding, was used for training, validation, and testing. This dataset contains approximately 108,000 
highly detailed images that have been sorted into 397 categories with  the dataset's full file size being around 36 
GB of memory. Additionally, the minimum number of images in each category of SUN397 is 100.

The model created as a part of this project is contained in the sun397_model. The Python program used to originally
generate the model is Test_SUN_dataset.py which used a VGG19 pre-trained model imported from Keras as a starting
point for training. The Python file cont_training.py was used to perform further fitting on the model, this was 
primarily implemented due to hardware constraints as training the model failed if too many epochs were used
at once. Finally, the sun397_model was evaluated for general accuracy using the Python file test.py.

To evaluate the model please make sure that the model is in the same directory as the sample dataset. 
Also, change the path in test.py to the path of the directory you are using to store the model and sample data.

__The following contains links to my Google Drive containing the sun397_model and the sample data set
the sun397_model is 84.1 MB and the sample dataset is 204 MB. So, room in memory will be needed for 
downloading and executing the model's evaluation.__

Model Download (84MB): https://drive.google.com/drive/folders/1boRyJEK9iOAfS7GTsODbyy8lnCw4HhiR?usp=sharing

Sample Dataset Download (204MB): https://drive.google.com/drive/folders/18NLE57fyTKxhaB9iAGoMg6egxe5zHA4g?usp=sharing



