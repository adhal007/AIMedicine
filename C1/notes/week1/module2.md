## Handling class imbalance and small training datasets (Model Training Theory)

Building and training a Model for medical diagnosis

Example: Chest X-Ray

**I. Background**:
- Most common, about 2 Billion per year 
- Important for detection of pneumonia, lung cancer, etc
- Radiologist looks at the lungs and heart to search for disease suggestive clues 
- Application of deep learning algorithm will be to identify a **mass**.
    - or our own reference, a mass is defined as a lesion, or in other words damage of tissue, seen on a chest X-ray as greater than 3 centimeters in diameter.
  
**II. Methods**:

- **Setup: Training, Prediction and loss**:
  - Inputs: Training data with images, labels [0, 1]
  - Outputs: Probability of each image being a mass (0, 1)
  - Error: (Desired label) - probability
    - Example, if image is a mass (label = 1), and a probability of 0.51, the error (loss) = 0.49
- **Challenges of training medical image classifiers**:
  - class imbalance
  - multiclass 
  - dataset size 
  
  **I. Class imbalance challenge**:
  - **Binary cross entropy loss function**: 
    - L(X, y) = -log10(P(Y=1|X)) if y = 1
    - L(X, y) = -log10(P(Y=0|X)) if y = 0
  - **Impact of class imbalance on loss**:
    - Most contribution to loss will end up from majority class 
  ![Figure 1](/C1/images/class_imbalance_loss_bias.png)
    - Algorithm will hence optimize loss to classify majority class correctly compared to minority class
  - **Solutions for handling class imbalance**:
    - Weighted loss
    ![Figure 2](/C1/images/class_imbalance_weighted_loss.png) 
      - This will make the sum of loss for each category equal. 
      - In the above example, the weights:
        - **Mass** group's weight (w~p~) =  6/8 
        - **Normal** group's weight (w~n~) = 2/8
      - Then $\sum_{i=1}^{2} L_{p_{i}} = \sum_{i=1}^{6} L_{n_{i}}$
        - where $L_{p_{i}}$ denotes the individual Loss of positive class (Normal) for $i^{th}$  sample 
        - $L_{n_{i}}$ denotes the individual loss of negative class (Mass) for $i^{th}$ sample
      - Weights formula:
        - $w_{p} = N_{n}/N$
        - $w_{n} = N_{p}/N$
        - $N = N_{n} + N_{p}$
    - Resampling:
    ![Figure 3](/C1/images/class_imbalance_resampling.png)
      
      - under sampling of majority class
      - over sampling of minority class
  
  **II. Multi-class challenge**:
   - Setup:
    ![Figure 4](/C1/images/Setup_Multitask_learning.png) 
      - Build a classifier that can predict probabilities for multiple labels as shown in the figure above 
    - Loss function modification:
    ![Figure 5](/C1/images/Loss_function_modification.png) 
      - sum of losses of each class considered (Why?)
      - Weighted loss of multi-class will have 6 weights 
        - Notation: 
          - $L(X, y_{mass}) = -w_{p, mass}log(P(Y=1|x))$ if y = 1  
          - $L(X, y_{mass}) = -w_{n, mass}log(P(Y=0|x))$ if y = 0
        - Different $w_{k, j}$ where $k \in (p, n)$ and $j \in $ (mass, pneumonia, edema) depending on count(k) of positive (p) and negative(n) samples for each label j.  

  **III. Dataset size challenge**:
   - Medical imaging diagnosis classification problems are addressed by a class of models called Convolutional neural networks. 
   - Well suited for 2D images, medical signal processing, 3D images (CT-Scans)
   - Some commonly used ConvNet models (Inception-V3, ResNet-34, DenseNet, ResNeXt, EfficientNet)
   - In medical problems, try out different tasks and see which ones work best
   - Medical imaging problems typically have a lower size of images (10k - 100K) compared to millions in other datasets (animals, cars, etc)
   - **Solutions**:
     - **Transfer learning for small dataset size**:
       - Transfer learning = pre-training + fine-tuning 
      ![Figure 6](/C1/images/CNN_tuning_for_small_sample_sizes.png)  
       - Pre-training: To learn general features and provide a good starting point to learner 
       - Fine-tuning: To learn dataset specific features for our classifier 
       - Strategies for fine-tuning:
        ![Figure 7](/C1/images/FIne_tuning_CNN_strategies.png)
         - Use general features from pre-trained model from early layers, learn specific features for dataset from later layers (faster compute time)
         - **General features** from animal classifier would be **identification of boundaries** 
         - **Higher level features** from animal classifier would be **"head of the penguin"** in the example
     
     - **Data Augmentation**:
       - Tricking the network into thinking we have more training samples at our disposal than we normally do.
       - Transformations for data augmentation:
         - Rotation
         - Zooming in 
         - Applying contrast 
       - Considerations while applying data augmentation transformations:
         - Variations for data augmentation should enable replicate real-world scenarios so that classifier works well on the test set. 
         - Preservation of labels (For eg: lateral inversion of chest X-ray does not) - Why? (Figure below)
        ![Figure 8](/C1/images/Data_augmentation_cosiderations_example1.png)
         - Hence different imaging problems require different data augmentations.
           - Examples:
            ![Figure 9](/C1/images/Data_augmentation_examples_2.png)
  
**III. Assignment Details**:
    - In the assignment, you will generalize this to calculating the loss for any number of classes.
    - Also in the assignment, you will learn how to avoid taking the log of zero by adding a small number (more details will be explained in the assignment).
    - Note that in the lecture videos and in this lecture notebook, you are taking the sum of losses for all examples. In the assignment, you will take the average (the mean) for all examples.
    - Finally, in the assignment, you will work with "tensors" in TensorFlow, so you will use the TensorFlow equivalents of the numpy operations (keras.mean instead of numpy.mean).