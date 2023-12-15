# CS5600_Project2_TomP
Tom Prouty's GitHub Repository for the Final Project of CS5600

__IMPORTANT: Do not run the Python files containing tfds. Running the files yourself will cause the system to try and download
the SUN397 dataset onto your computer, which is only possible if you have an available 36 GB minimum of space in memory.__

The goal of this project was to create a multi-category image classification ConvNet. To do this the TensorFlow
SUN397 dataset, Scene UNderstanding, created by Princeton University. This dataset contains approximately 108,000 
highly detailed images that have been sorted into 397 categories with  the dataset's full size being around 36 
GB of memory. Additionally, the minimum number of images in each category of SUN397 is 100, which is considered
low for image classification purposes from what papers I have read. The SUN397 dataset contains an interesting 
structure, but more detail regarding that structure is explained in the report.

The model created as a part of this project is contained in the sun397_model. The Python program used to originally
generate the model is Test_SUN_dataset.py which used a VGG19 pre-trained model imported from Keras as a starting
point for training. The Python file cont_training.py was used to perform further fitting on the model, this was 
primarily implemented due to hardware constraints as the model would fail to train if too many epochs were used
at once. Finally, the sun397_model was evaluated for general accuracy using the Python file test.py.

__The following contains links to my Google Drive containing the sun397_model and the sample data set
the sun397_model is 84.1 MB and the sample dataset is 204 MB. So, room in memory will be needed for 
downloading and execution of the model's evaluation__

Model Download:
Sample Dataset Download: 

