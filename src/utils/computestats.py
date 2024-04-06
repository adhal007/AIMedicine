
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.models import load_model
import keras.utils as image

# import util
# from public_tests import *
# from test_utils import *

import tensorflow as tf
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K

from sklearn.metrics import roc_auc_score, roc_curve

import src.utils.CustomLogger.custom_logger as cl
logger = cl.CustomLogger()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ComputeStats:
    def __init__(self) -> None:
        self.logger = logger.custlogger(loglevel='DEBUG')
        self.logger.info("Initializing the Compute Stats Class")
        # self.img_dir = img_dir
        # self.base_preprocessor = bp.BasePreProcessor(img_dir)

    ## Testing these 2 functions will be the next steps
    # UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
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
        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        
        # total number of patients (rows)
        N = labels.shape[0]
        
        positive_frequencies = np.sum(labels, axis=0)/N
        negative_frequencies = 1 - positive_frequencies

        ### END CODE HERE ###
        return positive_frequencies, negative_frequencies

    # UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
 
    
    def get_weighted_loss(self, pos_weights, neg_weights, epsilon=1e-7):
        """
        Return weighted loss function given negative weights and positive weights.

        Args:
        pos_weights (np.array): array of positive weights for each class, size (num_classes)
        neg_weights (np.array): array of negative weights for each class, size (num_classes)
        
        Returns:
        weighted_loss (function): weighted loss function
        """
        def weighted_loss(y_true, y_pred):
            """
            Return weighted loss value. 

            Args:
                y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
                y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
            Returns:
                loss (float): overall scalar loss summed across all classes
            """
            # initialize loss to zero
            loss = 0.0
            
            ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
            for i in range(len(pos_weights)):
                y_t = y_true[i]
                y_hat = y_pred[i]
                w_p = pos_weights[i]
                w_n = neg_weights[i]
                log_f_pos =  K.log(y_hat)
                log_f_neg = K.log(1-y_hat)
                
                # for each class, add average weighted loss for that class 
                loss += -1*(w_p * y_t * log_f_pos) + (w_n * (1-y_t) *log_f_neg)
            loss = K.mean(loss)
            return loss

            ### END CODE HERE ###
        return weighted_loss
    
    def calc_pos_neg_freq(self, labels):
        """
        compute the positive and negative frequencies based on the labels

        Args:
        labels (np.array): array of labels, size (num_examples, num_classes)

        Returns:
        freq_pos (np.array): array of positive frequences for each class, size (num_classes)
        """
        freq_pos, freq_neg = self.compute_class_freqs(labels=labels)
        return freq_pos, freq_neg
    
    def calc_pos_neg_weights(self, freq_pos, freq_neg):
        """
        compute the positive and negative weights based on the positive and negative frequencies

        Args:
        freq_pos (np.array): array of positive frequences for each class, size (num_classes)

        freq_neg (np.array): array of negative frequences for each class, size (num_classes)

        Returns:
        dict: dictionary with pos_weights, neg_weights, pos_contribution, neg_contribution

        """
        pos_weights = freq_neg
        neg_weights = freq_pos
        pos_contribution = freq_pos * pos_weights 
        neg_contribution = freq_neg * neg_weights
        return {'pos_weights':pos_weights, 
                'neg_weights':neg_weights,
                'pos_contribution':pos_contribution,
                'neg_contribution':neg_contribution}
    
    def get_mean_std_per_batch(self, image_dir, df, H=320, W=320):
        sample_data = []
        for img in df.sample(100)["Image"].values:
            image_path = os.path.join(image_dir, img)
            sample_data.append(
                np.array(image.load_img(image_path, target_size=(H, W))))

        mean = np.mean(sample_data, axis=(0, 1, 2, 3))
        std = np.std(sample_data, axis=(0, 1, 2, 3), ddof=1)
        return mean, std


    def load_image(self, img, image_dir, df, preprocess=True, H=320, W=320):
        """Load and preprocess image."""
        mean, std = self.get_mean_std_per_batch(image_dir, df, H=H, W=W)
        img_path = os.path.join(image_dir, img)
        x = image.load_img(img_path, target_size=(H, W))
        if preprocess:
            x -= mean
            x /= std
            x = np.expand_dims(x, axis=0)
        return x    
    
    def grad_cam(self, input_model, image, cls, layer_name, H=320, W=320):
        """GradCAM method for visualizing input saliency."""

        conv_output = input_model.get_layer(layer_name).output
        x_tensor = tf.convert_to_tensor(conv_output, dtype=tf.float32)
        with tf.GradientTape() as t:
            t.watch(x_tensor)
            output = input_model(x_tensor)

        result = output
        grads = t.gradient(result, x_tensor)
        # grads = g.gradient(y_c, conv_output)
            # grads = grads[0]
        # grads = K.gradients(y_c, conv_output)[0]

        gradient_function = K.function([input_model.input], [conv_output, grads])

        output, grads_val = gradient_function([image])
        output, grads_val = output[0, :], grads_val[0, :, :, :]

        weights = np.mean(grads_val, axis=(0, 1))
        cam = np.dot(output, weights)

        # Process CAM
        cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        return cam


    def compute_gradcam(self, model, img, image_dir, df, labels, selected_labels, layer_name='bn',
                    W = 320, H=320):
        
        preprocessed_input = self.load_image(img, image_dir, df)
        predictions = model.predict(preprocessed_input)
        
        ##############################
        print("Loading original image")
        plt.figure(figsize=(15, 10))
        plt.subplot(151)
        plt.title("Original")
        plt.axis('off')
        plt.imshow(self.load_image(img, image_dir, df, preprocess=False), cmap='gray')
        ##############################
        
        
        layer_name='bn'
        conv_output = model.get_layer(layer_name).output
        gradModel = Model(
                    inputs=[model.inputs],
                    outputs=[conv_output,model.output])
        
        j = 1
        for i in range(len(labels)):
            if labels[i] in selected_labels:
                print(f"Generating gradcam for class {labels[i]}")

                cls = 0 # specific class output probability
                with tf.GradientTape() as tape:
                    (convOutputs, pred) = gradModel(preprocessed_input)
                    loss = pred[:, cls]
                # use automatic differentiation to compute the gradients
                grads = tape.gradient(loss, convOutputs)
                
                output, grads_val = convOutputs[0, :], grads[0, :, :, :] #no need of batch information

                weights = np.mean(grads_val, axis=(0, 1))
                cam = np.dot(output, weights)

                # Process CAM
                cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
                cam = np.maximum(cam, 0)
                gradcam = cam / cam.max()
                
                ###############################
                plt.subplot(151 + j)
                plt.title(f"{labels[i]}: p={predictions[0][i]:.3f}")
                plt.axis('off')
                plt.imshow(self.load_image(img, image_dir, df, preprocess=False),cmap='gray')
                
                #value = np.array(min(0.5, predictions[0][i])).reshape(1,1)
                value = min(0.5, predictions[0][i])
                value = np.repeat(value,W*H).reshape(W,H)
                plt.imshow(gradcam, cmap='jet', alpha=value)
                j += 1
        pass 

                

    def get_roc_curve(self, labels, predicted_vals, generator):
        auc_roc_vals = []
        for i in range(len(labels)):
            try:
                gt = generator.labels[:, i]
                pred = predicted_vals[:, i]
                auc_roc = roc_auc_score(gt, pred)
                auc_roc_vals.append(auc_roc)
                fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
                plt.figure(1, figsize=(10, 10))
                plt.plot([0, 1], [0, 1], 'k--')
                plt.plot(fpr_rf, tpr_rf,
                        label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.title('ROC curve')
                plt.legend(loc='best')
            except:
                print(
                    f"Error in generating ROC curve for {labels[i]}. "
                    f"Dataset lacks enough examples."
                )
        plt.show()
        return auc_roc_vals