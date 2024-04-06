from keras.preprocessing.image import ImageDataGenerator
import os

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K 

import keras
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from pathlib import Path

import src.utils.batchgenerator as dg
import src.utils.CustomLogger.custom_logger as cl
logger = cl.CustomLogger()

### What attributes should BasePreProcessor have for MRI, CT, and X-ray?
    ### Attributes for MRI, CT, and X-ray:
    ### MRI is 3D while X-ray and CT are 2D
    ### MRI has 3 channels while X-ray and CT have 1 channel
    ### Example: MRI has a target size of (256, 256, 3) while X-ray and CT have a target size of (256, 256, 1)
### What common methods should BasePreProcessor have for MRI, CT, and X-ray?
### What methods should be abstract and implemented in the subclasses?
### What methods should be concrete and implemented in the base class?
### What methods should be static and implemented in the base class?
### What methods should be class methods and implemented in the base class?
### What methods should be instance methods and implemented in the base class?
### What methods should be private and implemented in the base class?
### What methods should be public and implemented in the base class?
### What methods should be protected and implemented in the base class?

class BasePreProcessor:
    def __init__(self, img_dir:str) -> None:
        self.logger = logger.custlogger(loglevel='DEBUG')
        self.logger.info("Initializing the Medical Image Preprocessor Class")
        self.img_dir = img_dir
        self.data_generator = None
        self.kwargs = {
        'data_frames': {
            'train_df': None,
            'valid_df': None,
            'test_df': None
        },
        'batch_size': None,  # You can set this later if needed
        'dim': (320, 320, 16),
        'num_channels': None,  # You can set this later if needed
        'num_classes': None,
        'shuffle': False,
        'x_col': None,
        'y_cols': None,
        'train_generator': None,
        'test_generator': None,
        'valid_generator': None,
        'labels': None,
        'pos_weights': None,
        'neg_weights': None,
        # New keys from MriPp module
        'orig_x': 240,
        'orig_y': 240,
        'orig_z': 155,
        'output_x': 160,
        'output_y': 160,
        'output_z': 16,
        'max_tries': 1000,
        'background_threshold': 0.95,
        'model_2d_or_3d': None
        }


    ## Checks already in GeneralData Generator
    def check_for_img_dir(self, img_dir):
        """
        Check if the image directory exists.

        Args:
            img_dir (str): Path to the image directory.
        
        Returns: 
            bool: True if the image directory exists, False otherwise.
        """
        if not os.path.exists(path=img_dir):
            raise ValueError("Please provide a valid image directory")
        else:
            return True
        
    def check_for_file_type(self, img_dir, file_type):
        """
        Check if the image directory contains the specified file type.

        Args:
            img_dir (str): Path to the image directory.
            file_type (str): File type to check for.
        
        Returns:  
            bool: True if the file type exists in the image directory, False otherwise.
        """
        for files in os.listdir(img_dir):
            if files.endswith(file_type):
                return True
            else:
                raise ValueError("Please provide a valid file type") 
            

    def load_image(self, file_path):
        """
        Load an image from a file path.

        Args:
            file_path (str): Path to the image file.

        Returns:
            np.ndarray: Loaded image as a NumPy array.
        """
        raise NotImplementedError("Subclasses must implement the 'load_image' method.")


        
    def initialize_data_generator(self, **kwargs):
        """
        Initialize the data generator.

        Args:
            **kwargs: Additional keyword arguments to be passed to the data generator.

        Returns:
            None
        """
        self.data_generator = dg.GeneralDataGenerator(img_dir=self.img_dir, **kwargs)
        pass

    def train(self):
        """
        Train the model using the data generator.

        Returns:
            None
        """
        if self.data_generator is None:
            raise ValueError("Data generator is not initialized. Call initialize_data_generator method first.")
        
        # Use the data generator for training
        # Example:
        # for epoch in range(num_epochs):
        #     for batch in self.data_generator:
        #         # Train the model using the batch data
        pass
    
    
    def compute_class_sens_spec(self, pred, label, class_num):
        """
        Compute sensitivity and specificity for a particular example
        for a given class.

        Args:
            pred (np.array): binary arrary of predictions, shape is
                            (num classes, height, width, depth).
            label (np.array): binary array of labels, shape is
                            (num classes, height, width, depth).
            class_num (int): number between 0 - (num_classes -1) which says
                            which prediction class to compute statistics
                            for.

        Returns:
            sensitivity (float): for a given class_num.
            specificity (float): for a given class_num.
        """

        # extract sub-array for specified class
        class_pred = pred[class_num]
        class_label = label[class_num]

        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        
        # compute:
        summed_arr = class_pred.flatten() + class_label.flatten()
        subtract_arr = class_pred.flatten() - class_label.flatten()

        # true positives
        tp = np.count_nonzero(summed_arr == 2)
        # true negatives
        tn = np.count_nonzero(summed_arr == 0)
        
        #false positives
        fp = np.count_nonzero(subtract_arr == 1)
        
        # false negatives
        fn = np.count_nonzero(subtract_arr == -1)

        # compute sensitivity and specificity
        sensitivity = tp/(tp + fn)
        specificity = tn/(tn + fp)

        ### END CODE HERE ###
        return sensitivity, specificity
    
    def update_kwargs(self):
        """
        Update the keyword arguments.

        Args:
            **kwargs: Additional keyword arguments to be updated.

        Returns:
            None
        """
        NotImplementedError("Subclasses must implement the 'update_kwargs' method.")


