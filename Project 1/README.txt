Name: Yogesh Patodia
Email: yp2607@columbia.edu

-------------------------------------------------------------------------------------------------------------------------------

Digits_Classification.py:
  Used to run MLE and KNN classifiers to identify handwritten digits

Compass Classification.py:
  Used to run MLE, KNN and Naive Bayes classifiers to predict 2 year recidivism
  on COMPAS dataset

Classifiers
1. NB.py: This file contains the naive bayes classifier implementation using count probabilities.
2. KNN.py: This file contains the KNN classifier implementation. Default k = 5 and uses L2 metric.
3. MLE.py: This file contains the maximum likelihood estimate classifier implementation.


These files should be present outside the compas_dataset directory to run.

The directory structure to be followed:

Project1
   |
   |---compas_dataset
   |	|--- propublicaTrain.csv
   |	|--- propublicaTest.csv
   |
   |---models
   |  |--- NB_COMPAS.py
   |  |--- KNN_COMPAS.py
   |  |--- MLE_COMPAS.py
   |
   |---Digits_Classification.py
   |---Compas_Classification.py
   |
   |---digits.mat

-------------------------------------------------------------------------------------------------------------------------------

STEPS TO RUN THE CLASSIFIERS
1. Open Console/ Terminal
2. Go to directory where the files are present.
3. Run the following command to see the classification report for that classifier
    python <filename.py>
