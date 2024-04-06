import numpy as np
import h5py
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from IPython.display import Image

### Single interface data generator class ############################################################
######################################################################################################
######################################################################################################
class GeneralDataGenerator(keras.utils.Sequence):
    def __init__(self,
                 sample_list=None,
                 base_dir=None,
                 data_frames=None,
                 img_dir=None,
                 x_col=None,
                 y_cols=None,
                 batch_size=1,
                 shuffle=True,
                 dim=(160, 160, 16),
                 num_channels=4,
                 num_classes=3,
                 verbose=1,
                 orig_x=240,
                 orig_y=240,
                 orig_z=155,
                 output_x=160,
                 output_y=160,
                 output_z=16,
                 max_tries=1000,
                 background_threshold=0.95,
                 model_2d_or_3d=None,
                 labels=None):
        self.sample_list = sample_list
        self.base_dir = base_dir
        self.img_dir = img_dir
        self.x_col = x_col
        self.y_cols = y_cols
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dim = dim
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.verbose = verbose
        self.orig_x = orig_x
        self.orig_y = orig_y
        self.orig_z = orig_z
        self.output_x = output_x
        self.output_y = output_y
        self.output_z = output_z
        self.max_tries = max_tries
        self.background_threshold = background_threshold
        self.model_2d_or_3d = model_2d_or_3d
        self.labels=labels

        if data_frames is not None:
            self._train_df = data_frames.get("train_df")
            self._valid_df = data_frames.get("valid_df")
            self._test_df = data_frames.get("test_df")
        else:
            self._train_df = None
            self._valid_df = None
            self._test_df = None

        if self.sample_list and self.base_dir:
            self.on_epoch_end()
        elif self.train_df is not None and self.train_df.shape[0] > 0 and self.img_dir is not None and self.x_col is not None and self.y_cols is not None:
            self._train_generator = self.get_train_generator()
            self._test_valid_generators = self.get_test_and_valid_generator()
    
    @property
    def train_df(self):
        return self._train_df
    
    @train_df.setter
    def train_df(self, df):
        self._train_df = df
        
    @property
    def valid_df(self):
        return self._valid_df
    
    @valid_df.setter
    def valid_df(self, df):
        self._valid_df = df
        
    @property
    def test_df(self):
        return self._test_df
    
    @test_df.setter
    def test_df(self, df):
        self._test_df = df

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.sample_list:
            self.indexes = np.arange(len(self.sample_list))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.sample_list:
            return int(np.floor(len(self.sample_list) / self.batch_size))
        elif self.train_df:
            return int(np.floor(len(self.train_df) / self.batch_size))

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'

        # Initialization
        if len(self.dim) == 3:  # 3D images
            X_shape = (self.batch_size, self.num_channels, *self.dim)
            y_shape = (self.batch_size, self.num_classes, *self.dim)
        elif len(self.dim) == 2:  # 2D images
            X_shape = (self.batch_size, *self.dim, self.num_channels)
            y_shape = (self.batch_size, *self.dim, self.num_classes)

        X = np.zeros(X_shape, dtype=np.float64)
        y = np.zeros(y_shape, dtype=np.float64)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if self.verbose == 1 and self.base_dir:
                print("Training on: %s" % (self.base_dir + ID))
            elif self.verbose == 1 and self.img_dir:
                print("Training on: %s" % (self.img_dir + ID))
            if self.base_dir:
                with h5py.File(self.base_dir + ID, 'r') as f:
                    X[i] = np.array(f.get("x"))
                    # remove the background class
                    y[i] = np.moveaxis(np.array(f.get("y")), 3, 0)[1:]
            elif self.img_dir:
                # Assuming RGB images for 2D case
                image = Image.open(self.img_dir + ID)
                X[i] = np.array(image)
                # Assuming single class for 2D case
                y[i] = np.array(Image.open(self.img_dir + ID))

        return X, y

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if self.sample_list:
            indexes = self.indexes[
                      index * self.batch_size: (index + 1) * self.batch_size]
            # Find list of IDs
            sample_list_temp = [self.sample_list[k] for k in indexes]
            # Generate data
            X, y = self.__data_generation(sample_list_temp)
        elif self.train_df:
            X, y = self._train_generator.next()

        return X, y

    def get_train_generator(self):
        'Return generator for training set'
        if self.train_df is not None and self.train_df.shape[0] > 0 and self.img_dir is not None and self.x_col is not None and self.y_cols is not None:
            image_generator = ImageDataGenerator(
                samplewise_center=True,
                samplewise_std_normalization=True)

            generator = image_generator.flow_from_dataframe(
                dataframe=self.train_df,
                directory=self.img_dir,
                x_col=self.x_col,
                y_col=self.y_cols,
                class_mode="raw",
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                seed=1,
                target_size=(self.dim[0], self.dim[1]))

            return generator

    def get_test_and_valid_generator(self):
        'Return generators for test and validation sets'
        if self.train_df is not None and self.train_df.shape[0] > 0 and self.img_dir is not None and self.x_col is not None and self.y_cols is not None:
            raw_train_generator = ImageDataGenerator().flow_from_dataframe(
                dataframe=self.train_df,
                directory=self.img_dir,
                x_col=self.x_col,
                y_col=self.y_cols,
                class_mode="raw",
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                target_size=(self.dim[0], self.dim[1]))

            batch = raw_train_generator.next()
            data_sample = batch[0]

            image_generator = ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True)

            image_generator.fit(x=data_sample)
        
        if self.valid_df is not None and self.valid_df.shape[0] > 0 and self.img_dir is not None and self.x_col is not None and self.y_cols is not None:
            valid_generator = image_generator.flow_from_dataframe(
                dataframe=self.valid_df,
                directory=self.img_dir,
                x_col=self.x_col,
                y_col=self.y_cols,
                class_mode="raw",
                batch_size=self.batch_size,
                shuffle=False,
                seed=1,
                target_size=(self.dim[0], self.dim[1]))
        
        if self.test_df is not None and self.test_df.shape[0] > 0 and self.img_dir is not None and self.x_col is not None and self.y_cols is not None:
            test_generator = image_generator.flow_from_dataframe(
                dataframe=self.test_df,
                directory=self.img_dir,
                x_col=self.x_col,
                y_col=self.y_cols,
                class_mode="raw",
                batch_size=self.batch_size,
                shuffle=False,
                seed=1,
                target_size=(self.dim[0], self.dim[1]))

        return valid_generator, test_generator


