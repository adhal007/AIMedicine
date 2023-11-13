from keras.preprocessing.image import ImageDataGenerator
import os

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils.CustomLogger.custom_logger as cl
from pathlib import Path

logger = cl.customLogger(__name__, log_level='INFO')
class MedicalImagePreprocess:
    def __init__(self,  
                 train_df:pd.DataFrame,
                 valid_df:pd.DataFrame,
                 test_df:pd.DataFrame,
                 labels:list,
                 target_w:int=320,
                 target_h:int=320, 
                 img_dir:str) -> None:
        
        self.img_dir = Path(img_dir)
        self.train_df = train_df 
        self.valid_df = valid_df
        self.test_df = test_df
        self.labels = labels
        self.target_w = target_w
        self.target_h = target_h
        self.logger = logger
        self.logger.info("Initializing the Medical Image Preprocessing Class")
        self.logger.info(f"Image Directory: {self.img_dir}")
        
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
        self._test_valid_generators = None 

    @property
    def all_image_files(self):
        if self._all_img_files is None:
            self._all_img_files = os.listdir(self.img_dir)
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



    ### This section contains all the functions for data exploration and visualization   
    ## could also externally give a set of 9 images to visualize right ?
    ## would help to explore any specific images if someone wanted to ?
    def check_for_img_dir(self, img_dir):
        if not os.path.exists(path=img_dir):
            raise ValueError("Please provide a valid image directory")
        else:
            return True
        
    def visualize_rand_imgs(self):
        print('Display Random Images')
        plt.figure(figsize=(20,10))
        for i in range(self.no_of_rand_imgs):
            plt.subplot(3,3, i+1)
            img = plt.imread(fname=os.path.join(self.img_dir, self.set_of_rand_imgs[i]))
            plt.imshow(img, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
    

    
    def single_image_visualize(self, sample_image):
        raw_image = plt.imread(fname=os.path.join(self.img_dir, sample_image))
        plt.imshow(X=raw_image, cmap='gray')
        ## this is what segments and shows the quantification of the image
        plt.colorbar()
        plt.title(label='Raw Chest X Ray Image')
        print(f"The dimensions of the image are {raw_image.shape[0]} pixels width and {raw_image.shape[1]} pixels height, one single color channel")
        print(f"The maximum pixel value is {raw_image.max():.4f} and the minimum is {raw_image.min():.4f}")
        print(f"The mean value of the pixels is {raw_image.mean():.4f} and the standard deviation is {raw_image.std():.4f}")

    def investigate_pixel_dist(self, sample_image):
        raw_image = plt.imread(fname=os.path.join(self.img_dir, sample_image))       
        sns.distplot(a=raw_image.ravel(), 
             label=f'Pixel Mean {np.mean(a=raw_image):.4f} & Standard Deviation {np.std(a=raw_image):.4f}', kde=False)
        plt.legend(loc='upper center')
        plt.title(label='Distribution of Pixel Intensities in the Image')
        plt.xlabel(xlabel='Pixel Intensity')
        plt.ylabel(ylabel='# Pixels in Image')

     
    def visualize_pos_counts(self):
        plt.figure(figsize=(10, 7))
        plt.bar(x=list(self.pos_counts.keys()), height=list(self.pos_counts.values()))
        plt.ylabel(ylabel='Number of Patients')
        plt.xticks(rotation=45)
        plt.title(label='Distribution of Positive Labels in Training Data')
        plt.show()

    ### This section contains all the functions for data preprocessing and augmentation
    ### components 
    ### 1. Data Generator
    ### 2. Data Leakage Check
    ### 3. Data Augmentation

    # This is the default generator    
    def get_generator(self, directory="data/nih/images-small/"):
        # Flow from directory with specified batch size and target image size
        default_generator = self.image_generator.flow_from_dataframe(
                dataframe=self.train_df,
                directory=directory,
                x_col="Image", # features
                # Let's say we build a model for mass detection
                y_col= ['Mass'], # labels
                class_mode="raw", # 'Mass' column should be in train_df
                batch_size= 1, # images per batch
                shuffle=False, # shuffle the rows or not
                target_size=(320,320) # width and height of output image
        )
        return default_generator
    
    def get_train_generator(self, df, image_dir, x_col, y_cols, shuffle=True, batch_size=8, seed=1):
        """
        Return generator for training set, normalizing using batch
        statistics.

        Args:
        df (dataframe): dataframe specifying training data.
        image_dir (str): directory where image files are held.
        x_col (str): name of column in df that holds filenames.
        y_cols (list): list of strings that hold y labels for images.
        batch_size (int): images per batch to be fed into model during training.
        seed (int): random seed.
        target_w (int): final width of input images.
        target_h (int): final height of input images.
        
        Returns:
            train_generator (DataFrameIterator): iterator over training set
        """        
        print("getting train generator...") 
        # normalize images
        image_generator = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization= True)
        
        # flow from directory with specified batch size
        # and target image size
        generator = image_generator.flow_from_dataframe(
                dataframe=df,
                directory=image_dir,
                x_col=x_col,
                y_col=y_cols,
                class_mode="raw",
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
                target_size=(self.target_w,self.target_h))
        
        return generator

    def get_test_and_valid_generator(self, valid_df, test_df, train_df, image_dir, x_col, y_cols, sample_size=100, batch_size=8, seed=1):
        """
        Return generator for validation set and test set using 
        normalization statistics from training set.

        Args:
        valid_df (dataframe): dataframe specifying validation data.
        test_df (dataframe): dataframe specifying test data.
        train_df (dataframe): dataframe specifying training data.
        image_dir (str): directory where image files are held.
        x_col (str): name of column in df that holds filenames.
        y_cols (list): list of strings that hold y labels for images.
        sample_size (int): size of sample to use for normalization statistics.
        batch_size (int): images per batch to be fed into model during training.
        seed (int): random seed.
        target_w (int): final width of input images.
        target_h (int): final height of input images.
        
        Returns:
            test_generator (DataFrameIterator) and valid_generator: iterators over test set and validation set respectively
        """
        print("getting train and valid generators...")
        # get generator to sample dataset
        raw_train_generator = ImageDataGenerator().flow_from_dataframe(
            dataframe=train_df, 
            directory=self.img_dir, 
            x_col="Image", 
            y_col=y_cols, 
            class_mode="raw", 
            batch_size=sample_size, 
            shuffle=True, 
            target_size=(self.target_w, self.target_h))
        
        # get data sample
        batch = raw_train_generator.next()
        data_sample = batch[0]

        # use sample to fit mean and std for test set generator
        image_generator = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization= True)
        
        # fit generator to sample from training data
        image_generator.fit(x=data_sample)

        # get test generator
        valid_generator = image_generator.flow_from_dataframe(
                dataframe=valid_df,
                directory=image_dir,
                x_col=x_col,
                y_col=y_cols,
                class_mode="raw",
                batch_size=batch_size,
                shuffle=False,
                seed=seed,
                target_size=(self.target_w,self.target_h))

        test_generator = image_generator.flow_from_dataframe(
                dataframe=test_df,
                directory=image_dir,
                x_col=x_col,
                y_col=y_cols,
                class_mode="raw",
                batch_size=batch_size,
                shuffle=False,
                seed=seed,
                target_size=(self.target_w,self.target_h))
        return valid_generator, test_generator
    
    @property 
    def train_generator(self):
        if self._train_generator is None:
            self._train_generator = self.get_train_generator(df=self.train_df,
                                                            image_dir=self.img_dir,
                                                            x_col="Image",
                                                            y_cols=self.labels,
                                                            shuffle=True,
                                                            batch_size=8,
                                                            seed=1,
                                                            target_w = 320,
                                                            target_h = 320)
        return self._train_generator

    @property
    def test_valid_generators(self):
        if self._test_valid_generators is None:
            self._valid_generator, self._test_generator = self.get_test_and_valid_generator(valid_df=self.valid_df,
                                                                                            test_df=self.test_df,
                                                                                            train_df=self.train_df,
                                                                                            image_dir=self.img_dir,
                                                                                            x_col="Image",
                                                                                            y_cols=self.labels,
                                                                                            sample_size=100,
                                                                                            batch_size=8,
                                                                                            seed=1)
        return self._valid_generator, self._test_generator

    
    def make_example_image(self, directory=None):
        generator = self.get_generator(directory=directory)
        sns.set_style(style="white")
        generated_image, label = generator.__getitem__(0)
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
        self.logger.info("Checking for Image Directory")
        self.check_for_img_dir(self.img_dir)


        self.logger.info("Checking for total number of images in directory")
        print("Total number of images in the directory: {}".format(len(self.all_image_files)))


        self.logger.info("Counting number of images in each dataset")
        print("Number of training images: {}".format(len(self.train_df)))
        print("Number of validation images: {}".format(len(self.valid_df)))
        print("Number of test images: {}".format(len(self.test_df)))
        
        self.logger.info("Checking example image in each generator")
        x, y = self.train_generator.__getitem__(0)
        plt.imshow(x[0]);
        x, y = self.test_generator.__getitem__(0)
        plt.imshow(x[0]);
        x, y = self.valid_generator.__getitem__(0)
        plt.imshow(x[0]);

        self.logger.info("Checking for Data Leakage")
        leakage_checks = [self.check_for_leakage(df1=self.train_df, df2=self.valid_df, patient_col='PatientId'),
                            self.check_for_leakage(df1=self.train_df, df2=self.test_df, patient_col='PatientId'),
                            self.check_for_leakage(df1=self.valid_df2, df2=self.test_df, patient_col='PatientId')]
        print("leakage between train and valid: {}".format(self.check_for_leakage(df1=self.train_df, df2=self.valid_df, patient_col='PatientId')))
        print("leakage between train and test: {}".format(self.check_for_leakage(df1=self.train_df, df2=self.test_df, patient_col='PatientId')))
        print("leakage between valid and test: {}".format(self.check_for_leakage(df1=self.valid_df, df2=self.test_df, patient_col='PatientId')))
        if any(leakage_checks):
            raise Exception('Data Leakage Detected. Please check your datasets again')
        

        self.logger.info("All data checks passed")


class MedicalImageEDA(MedicalImagePreprocess):
    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, labels: list, target_w: int = 320, target_h: int = 320, img_dir: str) -> None:
        super().__init__(train_df, valid_df, test_df, labels, target_w, target_h, img_dir)

    def visualize_class_imbalance_training(self):
        train_generator = self.train_generator
        labels = train_generator.labels
        plt.xticks(rotation=90)
        plt.bar(x=labels, height=np.mean(train_generator.labels, axis=0))
        plt.title("Frequency of Each Class")
        plt.show()
    
    def compute_class_freqs(self, labels):
        """
        Compute positive and negative frequences for each class.

        Args:
            labels (np.array): matrix of labels, size (num_examples, num_classes)
        Returns:
            positive_frequencies (np.array): array of positive frequences for each
                                            class, size (num_classes)
            negative_frequencies (np.array): array of negative frequences for each
                                            class, size (num_classes)
        """
        
        # total number of patients (rows)
        N = labels.shape[0]
        
        positive_frequencies = np.sum(labels, axis=0)/N
        negative_frequencies = 1 - positive_frequencies
        return positive_frequencies, negative_frequencies
    
    def visualize_class_imabalance(self):
        train_generator = self.train_generator
        labels = train_generator.labels
        pos_freq, neg_freq= self.compute_class_freqs(labels=train_generator.labels)
        data = pd.DataFrame(data={"Class": labels, "Label": "Positive", "Value": pos_freq})
        data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(iterable=neg_freq)], ignore_index=True)
        plt.xticks(rotation=90)
        f = sns.barplot(x="Class", y="Value", hue="Label" ,data=data)