

from keras.layers import concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical

from keras import backend as K

from keras.layers import Input
from keras.models import Model
from keras.layers import (
    Activation,
    Conv3D,
    Conv3DTranspose,
    MaxPooling3D,
    UpSampling3D,
)


from pathlib import Path
import src.utils.CustomLogger.custom_logger as cl
import src.utils.basepreprocessor as bp
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

from keras.models import load_model

import src.utils.computestats as cs
import keras.utils as image
# set_verbosity(INFO)
logger = cl.CustomLogger()


class DLModels:
    def __init__(self, model_2d_dir:str, pre_trained_wts:str, model_3d_dir:str) -> None:
        self.logger = logger.custlogger(loglevel='DEBUG')
        self.logger.info("Initializing the Deep Learning Model Class")
        self.model_2d_dir = model_2d_dir
        self.model_3d_dir = model_3d_dir
        self.pre_trained_wts = pre_trained_wts
        self.cs = cs.ComputeStats()
        # self.model_2d_kwargs ={
        #     'test_generator': None,
        #     'labels': None,
        #     'pos_weights': None,
        #     'neg_weights': None
        # }
        # self.img_dir = img_dir
        # self.base_preprocessor = bp.BasePreProcessor(img_dir)

        self._model_2d_kwargs ={
            'test_generator': None,
            'labels': None,
            'pos_weights': None,
            'neg_weights': None
        }

    @property
    def model_2d_kwargs(self):
        return self._model_2d_kwargs
    
    @model_2d_kwargs.setter
    def model_2d_kwargs(self, value):
        if isinstance(value, dict):
            self._model_2d_kwargs = value
        else:
            raise ValueError("model_2d_kwargs must be a dictionary")
        
    
    def create_convolution_block(self, input_layer, n_filters, batch_normalization=False,
                                kernel=(3, 3, 3), activation=None,
                                padding='same', strides=(1, 1, 1),
                                instance_normalization=False):
        """
        :param strides:
        :param input_layer:
        :param n_filters:
        :param batch_normalization:
        :param kernel:
        :param activation: Keras activation layer to use. (default is 'relu')
        :param padding:
        :return:
        """
        layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(
            input_layer)
        if activation is None:
            return Activation('relu')(layer)
        else:
            return activation()(layer)


    def get_up_convolution(self, n_filters, pool_size, kernel_size=(2, 2, 2),
                        strides=(2, 2, 2),
                        deconvolution=False):
        if deconvolution:
            return Conv3DTranspose(filters=n_filters, kernel_size=kernel_size,
                                strides=strides)
        else:
            return UpSampling3D(size=pool_size)


    def unet_model_3d(self, loss_function, input_shape=(4, 160, 160, 16),
                    pool_size=(2, 2, 2), n_labels=3,
                    initial_learning_rate=0.00001,
                    deconvolution=False, depth=4, n_base_filters=32,
                    include_label_wise_dice_coefficients=False, metrics=[],
                    batch_normalization=False, activation_name="sigmoid"):
        """
        Builds the 3D UNet Keras model.f
        :param metrics: List metrics to be calculated during model training (default is dice coefficient).
        :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
        coefficient for each label as metric.
        :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
        layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
        to train the model.
        :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
        layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
        :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
        divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
        :param pool_size: Pool size for the max pooling operations.
        :param n_labels: Number of binary labels that the model is learning.
        :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
        :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
        increases the amount memory required during training.
        :return: Untrained 3D UNet Model
        """
        inputs = Input(input_shape)
        current_layer = inputs
        levels = list()

        # add levels with max pooling
        for layer_depth in range(depth):
            layer1 = self.create_convolution_block(input_layer=current_layer,
                                            n_filters=n_base_filters * (
                                                    2 ** layer_depth),
                                            batch_normalization=batch_normalization)
            layer2 = self.create_convolution_block(input_layer=layer1,
                                            n_filters=n_base_filters * (
                                                    2 ** layer_depth) * 2,
                                            batch_normalization=batch_normalization)
            if layer_depth < depth - 1:
                current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
                levels.append([layer1, layer2, current_layer])
            else:
                current_layer = layer2
                levels.append([layer1, layer2])

        # add levels with up-convolution or up-sampling
        for layer_depth in range(depth - 2, -1, -1):
            up_convolution = self.get_up_convolution(pool_size=pool_size,
                                                deconvolution=deconvolution,
                                                n_filters=
                                                current_layer.shape[1])(
                current_layer)
            concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
            current_layer = self.create_convolution_block(
                n_filters=levels[layer_depth][1].shape[1],
                input_layer=concat, batch_normalization=batch_normalization)
            current_layer = self.create_convolution_block(
                n_filters=levels[layer_depth][1].shape[1],
                input_layer=current_layer,
                batch_normalization=batch_normalization)

        final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
        act = Activation(activation_name)(final_convolution)
        model = Model(inputs=inputs, outputs=act)

        if not isinstance(metrics, list):
            metrics = [metrics]

        model.compile(optimizer=Adam(lr=initial_learning_rate), loss=loss_function,
                    metrics=metrics)
        return model
    
    def weighted_dense_net_model_2d(self, **kwargs):
        """
        generate a weighted 2D densenet model

        Args:
            **kwargs: keyword arguments to pass to the model
            kwargs['labels']: list of labels
            kwargs['pos_weights']: positive weights
            kwargs['neg_weights']: negative weights
        
        Returns:
            model: a 2D densenet model
        """
        # print("Received kwargs:", kwargs)

        # Extracting kwargs or providing default values
        labels = kwargs.get('labels', None)
        pos_weights = kwargs.get('pos_weights', None)
        neg_weights = kwargs.get('neg_weights', None)

        # Ensure that all required arguments are provided
        if labels is None or pos_weights is None or neg_weights is None:
            raise ValueError("Please provide all required arguments: labels, pos_weights, and neg_weights")        
        base_model = DenseNet121(weights=self.model_2d_dir, include_top=False)

        x = base_model.output

        # add a global spatial average pooling layer
        x = GlobalAveragePooling2D()(x)

        # and a logistic layer
        predictions = Dense(len(labels), activation="sigmoid")(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer='adam', loss=self.cs.get_weighted_loss(pos_weights, neg_weights))
        
        return model
    
    def predict_2d(self, **kwargs):
        print("Received kwargs:", kwargs)
    
        # Extracting kwargs or providing default values
        labels = kwargs.get('labels', None)
        pos_weights = kwargs.get('pos_weights', None)
        neg_weights = kwargs.get('neg_weights', None)
        test_generator = kwargs.get('test_generator', None)
        
        # Ensure that all required arguments are provided
        if labels is None or pos_weights is None or neg_weights is None or test_generator is None:
            raise ValueError("Please provide all required arguments: labels, pos_weights, neg_weights, and test_generator")
        
        model = self.weighted_dense_net_model_2d(**kwargs)
        if self.pre_trained_wts is not None:
            model.load_weights(self.pre_trained_wts)
        else:
            raise ValueError("Please provide a valid path to the pre-trained weights")
        predicted_vals = model.predict_generator(test_generator, steps=len(test_generator))
        return predicted_vals
    

    
