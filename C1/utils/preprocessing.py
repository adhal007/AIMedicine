from keras.preprocessing.image import ImageDataGenerator
import os

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MedicalImagePreprocess:
    def __init__(self,  train_df:pd.DataFrame, img_dir=None) -> None:
        
        self.img_dir = None 
        self.train_df = train_df 

        ## error out if empty dataframe or empty img directory is provided 
        if self.train_df is None:
            raise ValueError("Please provide a valid training dataframe")
        if self.img_dir is None:
            raise ValueError("Plose provide a valid image directory")


        self.accession_columns = {'Image_file_name': 'Image',
                                  'Patient_id': 'PatientId',
                                  }
        self.no_of_rand_imgs = 9

        self._image_generator = None 
        self._all_img_files = None
        self._set_of_rand_imgs = None
        self._pos_counts = None

    @property
    def all_image_files(self):
        if self._all_img_files is None:
            return self.train_df['Image'].values

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
        
    ## could also externally give a set of 9 images to visualize right ?
    ## would help to explore any specific images if someone wanted to ?
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
    
    def make_example_image(self, directory="data/nih/images-small/"):
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
    