## Checking your model performance (Model Testing Theory)

**I. Background**
- When applying ML/DL models to a dataset we have a training and testing dataset that are more often split as below:
    ![Figure 1](/C1/images/Model_testing_structure.png)
- Components:
  - Training: 
    - Comprises of training and validation
    - Goal: Learn features and tune hyper-parameters of the model
    - Cross-validation - Multiple sets of training and validation to average out best performance
  - Testing:
    - Completely held out dataset to apply the final model on.
    - Could be a blinded dataset never seen before.

**II. Key Challenges**
- Patient overlap
  - Same patient coming in twice for image diagnosis at 2 different timepoints 
  ![Figure 2](/C1/images/Patient_Overlap.png) 
  - If the 2 images end up in training and testing sets, then the model might memorize the label associated with the embeddings resulting in an over optimistic score. 
  - Solutions:
    - Keep single patient only in one set 
    - All images belonging to the same patient should be kept in the same set. 
- Set Sampling
  - When few examples are present, then during sampling to evaluate performance, one class might be completely absent from the set 
  - Solution:
    - Sample a population with atleast X% of the minority class (For example 50%)
    - Sample the same distribution of classes as test set for the validation set 
    - Finally put all remaining patients into the training set
    - Training set will have small number of disease labelled samples but we can easily apply strategies to fix imbalancing in the training set. 
- Ground truth/Reference standard
  - Problems:
    - user disagreement between radiologists (inter-observer disagreement) common in medical settings
  - Solution: 
    - Apply consensus voting to set ground truth for inter-observer disagreement 
    - Use a more definitive test such as CT scan