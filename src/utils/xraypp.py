import os
from pathlib import Path

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


import src.utils.CustomLogger.custom_logger as cl
import src.utils.basepreprocessor as bp
import src.utils.batchgenerator as dg
import src.utils.computestats as cs
logger = cl.CustomLogger()

class XrayPP(bp.BasePreProcessor):
    def __init__(self,  
                 train_df:pd.DataFrame,
                 valid_df:pd.DataFrame,
                 test_df:pd.DataFrame,
                 labels:list,
                 img_dir:str,
                 target_w:int=320,
                 target_h:int=320, 
                 ) -> None:
        super().__init__(img_dir=Path(img_dir))
        self.img_dir = Path(self.img_dir)
        self.train_df = train_df 
        self.valid_df = valid_df
        self.test_df = test_df
        self.labels = labels
        self.target_w = target_w
        self.target_h = target_h 
        self.cs = cs.ComputeStats()
        self.logger = logger.custlogger(loglevel='DEBUG')
        self.logger.info(msg="Initializing the X-ray image preprocessing class")
        self.logger.info(msg=f"Image Directory: {self.img_dir}")

        
        ## error out if empty dataframe or empty img directory is provided 
        if self.train_df is None:
            raise ValueError("Please provide a valid training dataframe")
        
        if self.test_df is None:
            raise ValueError("Please provide a valid testing dataframe")
        
        if self.valid_df is None:
            raise ValueError("Please provide a valid validation dataframe")
        
        if self.img_dir is None:
            raise ValueError("Plose provide a valid image directory")

        if self.labels is None:
            raise ValueError("Please provide a valid list of labels")
        
        self.accession_columns = {'Image_file_name': 'Image',
                                  'Patient_id': 'PatientId',
                                  }
        self.no_of_rand_imgs = 9

        self._image_generator = None 
        self._all_img_files = None
        self._set_of_rand_imgs = None
        self._pos_counts = None
        self._train_generator = None 
        self._test_generator = None
        self._valid_generator = None


        # Update kwargs_2d based on initialization parameters
        self._kwargs_2d = {
            # 'img_dir': self.img_dir,
            'data_frames': {'train_df': self.train_df,
                            'valid_df': self.valid_df,
                            'test_df': self.test_df},
            'batch_size': None,  # You can set this later if needed
            'dim': (self.target_w, self.target_h, 16),
            'num_channels': None,  # You can set this later if needed
            'num_classes': len(self.labels),
            'shuffle': False,
            'x_col': 'Image',
            'y_cols': self.labels,
            'model_2d_or_3d': '2D', 
            'labels': self.labels
        }

    @property
    def kwargs_2d(self):
        return self._kwargs_2d

    @kwargs_2d.setter
    def kwargs_2d(self, value):
        if isinstance(value, dict):
            self._kwargs_2d.update(value)
        else:
            raise ValueError("kwargs_2d must be a dictionary")

    @property
    def all_image_files(self):
        if self._all_img_files is None:
            self._all_img_files = os.listdir(path=self.img_dir)
            return self._all_img_files
        
    @property
    def pos_counts(self):
        if self._pos_counts is None:
            columns = list(self.train_df.keys())
            columns.remove(self.accession_columns['Image_file_name'])
            columns.remove(self.accession_columns['Patient_id'])
            count_dict = {}
            for column in columns:
                count_dict[column] = self.train_df[column].sum()
            return count_dict
        
    @property
    def set_of_rand_imgs(self):
        if self._set_of_rand_imgs is None:
            random_images = [np.random.choice(self.all_image_files) for i in range(self.no_of_rand_imgs)]
            return random_images 
    
    @property
    def image_generator(self):
        if self._image_generator is None:
            image_generator = ImageDataGenerator(
                samplewise_center=True, #Set each sample mean to 0.
                samplewise_std_normalization= True # Divide each input by its standard deviation
                )
            return image_generator
    
    ### Lot of these Checks already in GeneralData Generator
    ### Do not repeat yourself
    def check_for_img_dir(self, img_dir):
        if not os.path.exists(path=img_dir):
            raise ValueError("Please provide a valid image directory")
        else:
            return True
        
    def check_for_file_type(self, img_dir, file_type):
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

    ### This section contains all the functions for data exploration and visualization   
    ### could also externally give a set of 9 images to visualize right ?
    ### would help to explore any specific images if someone wanted to ?
    def make_example_image(self, batch_size, num_channels):
        generator = self.get_generator(batch_size=batch_size, num_channels=num_channels)
        sns.set_style(style="white")
        generated_image, label = generator.__getitem__(index=0)
        plt.imshow(X=generated_image[0], cmap='gray')
        plt.colorbar()
        plt.title(label='Raw Chest X Ray Image')
        print(f"The dimensions of the image are {generated_image.shape[1]} pixels width and {generated_image.shape[2]} pixels height")
        print(f"The maximum pixel value is {generated_image.max():.4f} and the minimum is {generated_image.min():.4f}")
        print(f"The mean value of the pixels is {generated_image.mean():.4f} and the standard deviation is {generated_image.std():.4f}")
        return generated_image, label 
    
    def compare_raw_generated_image(self, raw_image, generated_image):
        # Include a histogram of the distribution of the pixels
        sns.set()
        plt.figure(figsize=(10, 7))

        # Plot histogram for original iamge
        sns.distplot(raw_image.ravel(), 
                    label=f'Original Image: mean {np.mean(raw_image):.4f} - Standard Deviation {np.std(raw_image):.4f} \n '
                    f'Min pixel value {np.min(raw_image):.4} - Max pixel value {np.max(raw_image):.4}',
                    color='blue', 
                    kde=False)

        # Plot histogram for generated image
        sns.distplot(generated_image[0].ravel(), 
                    label=f'Generated Image: mean {np.mean(generated_image[0]):.4f} - Standard Deviation {np.std(generated_image[0]):.4f} \n'
                    f'Min pixel value {np.min(generated_image[0]):.4} - Max pixel value {np.max(generated_image[0]):.4}', 
                    color='red', 
                    kde=False)

        # Place legends
        plt.legend()
        plt.title('Distribution of Pixel Intensities in the Image')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('# Pixel')
        plt.show()

    ### This section contains all the functions for data preprocessing and augmentation
    ### components 
    ### 1. Data Generator
    ### 2. Data Leakage Check
    ### 3. Data Augmentation
    def check_for_leakage(self, df1, df2, patient_col):
        """
        Return True if there any patients are in both df1 and df2.

        Args:
            df1 (dataframe): dataframe describing first dataset
            df2 (dataframe): dataframe describing second dataset
            patient_col (str): string name of column with patient IDs
        
        Returns:
            leakage (bool): True if there is leakage, otherwise False
        """

        df1_patients_unique = set(df1[patient_col])
        df2_patients_unique = set(df2[patient_col])
        
        patients_in_both_groups = list(df1_patients_unique & df2_patients_unique)

        # leakage contains true if there is patient overlap, otherwise false.
        leakage = True if len(patients_in_both_groups) > 0 else False # boolean (true if there is at least 1 patient in both groups)
            
        return leakage

    def run_checks(self):
        self.logger.info("Running Data Checks")

        self.logger.info("Checking for total number of images in directory")
        print("Total number of images in the directory: {}".format(len(self.all_image_files)))


        self.logger.info("Counting number of images in each dataset")
        print("Number of training images: {}".format(len(self.train_df)))
        print("Number of validation images: {}".format(len(self.valid_df)))
        print("Number of test images: {}".format(len(self.test_df)))
        
        # self.logger.info("Checking example image in each generator")
        # x, y = self.train_generator.__getitem__(0)
        # plt.imshow(x[0]);
        # x, y = self.test_generator.__getitem__(0)
        # plt.imshow(x[0]);
        # x, y = self.valid_generator.__getitem__(0)
        # plt.imshow(x[0]);

        self.logger.info("Checking for Data Leakage")
        leakage_checks = [self.check_for_leakage(df1=self.train_df, df2=self.valid_df, patient_col='PatientId'),
                            self.check_for_leakage(df1=self.train_df, df2=self.test_df, patient_col='PatientId'),
                            self.check_for_leakage(df1=self.valid_df, df2=self.test_df, patient_col='PatientId')]
        print("leakage between train and valid: {}".format(self.check_for_leakage(df1=self.train_df, df2=self.valid_df, patient_col='PatientId')))
        print("leakage between train and test: {}".format(self.check_for_leakage(df1=self.train_df, df2=self.test_df, patient_col='PatientId')))
        print("leakage between valid and test: {}".format(self.check_for_leakage(df1=self.valid_df, df2=self.test_df, patient_col='PatientId')))
        if any(leakage_checks):
            raise Exception('Data Leakage Detected. Please check your datasets again')
        
        self.logger.info("All data checks passed")


    def set_batch_size(self, batch_size):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer")
        self._kwargs_2d['batch_size'] = batch_size

    def set_num_channels(self, num_channels):
        if not isinstance(num_channels, int) or num_channels <= 0:
            raise ValueError("Number of channels must be a positive integer")
        self._kwargs_2d['num_channels'] = num_channels

    def get_generator(self, batch_size, num_channels):
        self.logger.info("Initializing the Data Generator For Training, Testing and Validation generators")
        self.set_batch_size(batch_size=batch_size)
        self.set_num_channels(num_channels=num_channels)
        self.initialize_data_generator(**self.kwargs_2d)
        return self.data_generator
    
    def update_kwargs(self):
        self.logger.info("Updating the kwargs for 2D model")
        kwargs_updated = self.kwargs | self.kwargs_2d
        self.kwargs = kwargs_updated
        return self.kwargs
    
    

        
    
    