###### II. Eye disease and Cancer diagnosis

II. a. Example of diabetic retinopathy 
- Retinal fundus images (back of the eye) for detection of diabetic retinopathy 
- Detecting DR is time consuming and manual process, requires clinical expertise 
- 120K images, 30% had DR 
  - imbalanced dataset
- Results: 
  - Performance of deep learning algorithm as good as opthamologists
  - majority vote of multiple opthamologist was used to set grouth truth/labels

II. b. Histopathology 

- Medical specialty involving examination of tissues under a microscope 
- Pathologists look at scanned microscopic images of tissue (whole slide images) to determine extent of cancer spreading. 
  - Helps plan treatment, disease progression timeline and chance of recovery 
- 2017, using 270 slides, deep learning algorithms developed 
- Problem:
  - Large image files, cannot be fed directly into an algorithm without breaking down

- Solution:
  -  Break down large image into small images of patches (high magnification) with original labels and feed into a neural network to train on hundreds of thousands of patches.
-  **Potential of a deep learning algorithm to improve pathologist accuracy and efficiency in a digital pathology workflow** [(Paper Reference)](https://pubmed.ncbi.nlm.nih.gov/30312179/)
   - How?
     -  Algorithm-assisted pathologists demonstrated **higher accuracy** than either the algorithm or the pathologist alone
     -  algorithm assistance **significantly increased the sensitivity of detection** for micrometastases (**91% vs. 83%, P=0.02**)
     - **Significantly shorter average review time**:  with assistance than without assistance for both micrometastases (61 vs. 116 s, P=0.002) and negative images (111 vs. 137 s, P=0.018)
     - pathologists considered the image review of micrometastases to be significantly easier when interpreted with assistance (P=0.0005)
   - This paper's results outlines the metrics for showcasing ability of deep learning algorithms for medical diagnostic assistance. 