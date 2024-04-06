
import h5py
import keras
import numpy as np
import tensorflow as tf
from IPython.display import Image
from keras import backend as K

import pandas as pd
import nibabel as nib
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical



from pathlib import Path
import src.utils.CustomLogger.custom_logger as cl
import src.utils.basepreprocessor as bp

logger = cl.CustomLogger()

class MriPP(bp.BasePreProcessor):
    def __init__(self, img_dir: str, **kwargs) -> None:
        super().__init__(img_dir)
        self.logger = logger.custlogger(loglevel='DEBUG')
        self.logger.info("Initializing the Medical Image Subsection Extraction BASE Class")

        
        # Initialize default values for kwargs
        self.orig_x = kwargs.get('orig_x', 240)
        self.orig_y = kwargs.get('orig_y', 240)
        self.orig_z = kwargs.get('orig_z', 155)
        self.output_x = kwargs.get('output_x', 160)
        self.output_y = kwargs.get('output_y', 160)
        self.output_z = kwargs.get('output_z', 16)
        self.num_classes = kwargs.get('num_classes', 4)
        self.max_tries = kwargs.get('max_tries', 1000)
        self.background_threshold = kwargs.get('background_threshold', 0.95)
    
    def load_image(self, image_nifty_file, label_nifty_file):
        """
        load the image and label file, get the image content and return a numpy array for each

        Args:
            image_nifty_file (str): path to the image file
            label_nifty_file (str): path to the label file

        Returns:
            image (np.array): image content
            label (np.array): label content
        """
    # load the image and label file, get the image content and return a numpy array for each
        image = np.array(nib.load(image_nifty_file).get_fdata())
        label = np.array(nib.load(label_nifty_file).get_fdata())
        
        return image, label
    

    def get_sub_volume(self, image, label, 
                    orig_x = 240, orig_y = 240, orig_z = 155, 
                    output_x = 160, output_y = 160, output_z = 16,
                    num_classes = 4, max_tries = 1000, 
                    background_threshold=0.95):
        """
        Extract random sub-volume from original images.

        Args:
            image (np.array): original image, 
                of shape (orig_x, orig_y, orig_z, num_channels)
            label (np.array): original label. 
                labels coded using discrete values rather than
                a separate dimension, 
                so this is of shape (orig_x, orig_y, orig_z)
            orig_x (int): x_dim of input image
            orig_y (int): y_dim of input image
            orig_z (int): z_dim of input image
            output_x (int): desired x_dim of output
            output_y (int): desired y_dim of output
            output_z (int): desired z_dim of output
            num_classes (int): number of class labels
            max_tries (int): maximum trials to do when sampling
            background_threshold (float): limit on the fraction 
                of the sample which can be the background

        returns:
            X (np.array): sample of original image of dimension 
                (num_channels, output_x, output_y, output_z)
            y (np.array): labels which correspond to X, of dimension 
                (num_classes, output_x, output_y, output_z)
        """
        # Initialize features and labels with `None`
        X = None
        y = None

        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        
        tries = 0
        
        while tries < max_tries:
            # randomly sample sub-volume by sampling the corner voxel
            # hint: make sure to leave enough room for the output dimensions!
            # do not remove/delete the '0's
            
            start_x = np.random.randint(orig_x - output_x + 1)
            start_y = np.random.randint(orig_y - output_y + 1)
            start_z = np.random.randint(orig_z - output_z + 1)
            
            # extract relevant area of label
            y = label[start_x: start_x + output_x,
                    start_y: start_y + output_y,
                    start_z: start_z + output_z]
            
            # One-hot encode the categories.
            # This adds a 4th dimension, 'num_classes'
            # (output_x, output_y, output_z, num_classes)
            y = keras.utils.to_categorical(y, num_classes=num_classes)

            # compute the background ratio (this has been implemented for you)
            bgrd_ratio = np.sum(y[:, :, :, 0])/(output_x * output_y * output_z)

            # increment tries counter
            tries += 1

            # if background ratio is below the desired threshold,
            # use that sub-volume.
            # otherwise continue the loop and try another random sub-volume
            if bgrd_ratio < background_threshold:

                # make copy of the sub-volume
                X = np.copy(image[start_x: start_x + output_x,
                                start_y: start_y + output_y,
                                start_z: start_z + output_z, :])
                
                # change dimension of X
                # from (x_dim, y_dim, z_dim, num_channels)
                # to (num_channels, x_dim, y_dim, z_dim)

                X = np.moveaxis(X, -1, 0)
                # change dimension of y
                # from (x_dim, y_dim, z_dim, num_classes)
                # to (num_classes, x_dim, y_dim, z_dim)
                y = np.moveaxis(y, -1, 0)
                ### END CODE HERE ###
                
                # take a subset of y that excludes the background class
                # in the 'num_classes' dimension
                y = y[1:, :, :, :]
        
                return X, y

        # if we've tried max_tries number of samples
        # Give up in order to avoid looping forever.
        print(f"Tried {tries} times to find a sub-volume. Giving up...")

    def standardize_img(self, image):
        """
        Standardize mean and standard deviation 
            of each channel and z_dimension.

        Args:
            image (np.array): input image, 
                shape (num_channels, dim_x, dim_y, dim_z)

        Returns:
            standardized_image (np.array): standardized version of input image
        """
        
        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        
        # initialize to array of zeros, with same shape as the image
        standardized_image = np.zeros(image.shape)

        # iterate over channels
        for c in range(image.shape[0]):
            # iterate over the `z` dimension
            for z in range(image.shape[3]):
                # get a slice of the image 
                # at channel c and z-th dimension `z`
                image_slice = image[c,:,:,z]

                # subtract the mean from image_slice
                centered = image_slice - np.mean(image_slice)
                
                # divide by the standard deviation (only if it is different from zero)
                if np.std(centered) != 0:
                    centered_scaled = None

                    # update  the slice of standardized image
                    # with the scaled centered and scaled image
                centered_scaled = centered/np.std(centered)
                standardized_image[c, :, :, z] = centered_scaled

        ### END CODE HERE ###

        return standardized_image
    
    def single_class_dice_coefficient(self, y_true, y_pred, axis=(0, 1, 2), 
                                  epsilon=0.00001):
        """
        Compute dice coefficient for single class.

        Args:
            y_true (Tensorflow tensor): tensor of ground truth values for single class.
                                        shape: (x_dim, y_dim, z_dim)
            y_pred (Tensorflow tensor): tensor of predictions for single class.
                                        shape: (x_dim, y_dim, z_dim)
            axis (tuple): spatial axes to sum over when computing numerator and
                        denominator of dice coefficient.
                        Hint: pass this as the 'axis' argument to the K.sum function.
            epsilon (float): small constant added to numerator and denominator to
                            avoid divide by 0 errors.
        Returns:
            dice_coefficient (float): computed value of dice coefficient.     
        """

        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        
        dice_numerator = 2*(K.sum(y_pred*y_true, axis=axis)) + epsilon
        dice_denominator = K.sum(y_pred, axis=axis) + K.sum(y_true, axis=axis) + epsilon
        dice_coefficient = dice_numerator/dice_denominator
        
        ### END CODE HERE ###

        return dice_coefficient

    def dice_coefficient(self, y_true, y_pred, axis=(1, 2, 3), 
                        epsilon=0.00001):
        """
        Compute mean dice coefficient over all abnormality classes.

        Args:
            y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                        shape: (num_classes, x_dim, y_dim, z_dim)
            y_pred (Tensorflow tensor): tensor of predictions for all classes.
                                        shape: (num_classes, x_dim, y_dim, z_dim)
            axis (tuple): spatial axes to sum over when computing numerator and
                        denominator of dice coefficient.
                        Hint: pass this as the 'axis' argument to the K.sum function.
            epsilon (float): small constant add to numerator and denominator to
                            avoid divide by 0 errors.
        Returns:
            dice_coefficient (float): computed value of dice coefficient.     
        """

        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        
        dice_numerator = 2*(K.sum(y_pred*y_true, axis=axis)) + epsilon
        dice_denominator = K.sum(y_pred, axis=axis) + K.sum(y_true, axis=axis) + epsilon
        dice_coefficient = K.mean(dice_numerator/dice_denominator)
        
        ### END CODE HERE ###

        return dice_coefficient
    
    def soft_dice_loss(self, y_true, y_pred, axis=(1, 2, 3), 
                   epsilon=0.00001):
        """
        Compute mean soft dice loss over all abnormality classes.

        Args:
            y_true (Tensorflow tensor): tensor of ground truth values for all classes.
                                        shape: (num_classes, x_dim, y_dim, z_dim)
            y_pred (Tensorflow tensor): tensor of soft predictions for all classes.
                                        shape: (num_classes, x_dim, y_dim, z_dim)
            axis (tuple): spatial axes to sum over when computing numerator and
                        denominator in formula for dice loss.
                        Hint: pass this as the 'axis' argument to the K.sum function.
            epsilon (float): small constant added to numerator and denominator to
                            avoid divide by 0 errors.
        Returns:
            dice_loss (float): computed value of dice loss.     
        """

        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        dice_numerator = 2*(K.sum(y_pred*y_true, axis=axis)) + epsilon
        dice_denominator = K.sum(y_pred**2, axis=axis) + K.sum(y_true**2, axis=axis) + epsilon
        dice_loss = 1 - K.mean(dice_numerator/dice_denominator)

        ### END CODE HERE ###

        return dice_loss
    
    def get_sens_spec_df(self, pred, label):
        patch_metrics = pd.DataFrame(
            columns = ['Edema', 
                    'Non-Enhancing Tumor', 
                    'Enhancing Tumor'], 
            index = ['Sensitivity',
                    'Specificity'])
        
        for i, class_name in enumerate(patch_metrics.columns):
            sens, spec = self.compute_class_sens_spec(pred, label, i)
            patch_metrics.loc['Sensitivity', class_name] = round(sens,4)
            patch_metrics.loc['Specificity', class_name] = round(spec,4)

        return patch_metrics
    

    @property
    def kwargs(self):
        # Return kwargs for 3D model
        return {
            'orig_x': self.orig_x,
            'orig_y': self.orig_y,
            'orig_z': self.orig_z,
            'output_x': self.output_x,
            'output_y': self.output_y,
            'output_z': self.output_z,
            'num_classes': self.num_classes,
            'max_tries': self.max_tries,
            'background_threshold': self.background_threshold
        }

    @kwargs.setter
    def kwargs(self, value):
        if isinstance(value, dict):
            # Update kwargs with new values
            self.orig_x = value.get('orig_x', self.orig_x)
            self.orig_y = value.get('orig_y', self.orig_y)
            self.orig_z = value.get('orig_z', self.orig_z)
            self.output_x = value.get('output_x', self.output_x)
            self.output_y = value.get('output_y', self.output_y)
            self.output_z = value.get('output_z', self.output_z)
            self.num_classes = value.get('num_classes', self.num_classes)
            self.max_tries = value.get('max_tries', self.max_tries)
            self.background_threshold = value.get('background_threshold', self.background_threshold)
        else:
            raise ValueError("kwargs must be a dictionary")