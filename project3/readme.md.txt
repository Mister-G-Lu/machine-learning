Drugs are typically small organic molecules that achieve their desired activity by binding to a target site on a receptor. The first step in the discovery of a new drug is usually to identify and isolate the receptor to which it should bind, followed by testing many small molecules for their ability to bind to the target site. This leaves researchers with the task of determining what separates the active (binding) compounds from the inactive (non-binding) ones. Such a determination can then be used in the design of new compounds that not only bind, but also have all the other properties required for a drug (solubility, oral absorption, lack of side effects, appropriate duration of action, toxicity, etc.). 

The goal of this competition is to allow you to develop predictive models that can determine given a particular compound whether it is active (1) or not (0).  As such, the goal would be develop the best binary classification model.

A molecule can be represented by several thousands of binary features which represent their topological shapes and other characteristics important for binding.

Since the dataset is imbalanced the scoring function will be the F1-score (macro averaging) instead of Accuracy.

Caveats:

+ Remember not all features will be good for predicting activity. Think of feature selection, engineering, reduction (anything that works)

+ The dataset has an imbalanced distribution i.e., within the training set there are only 78 actives (+1) and 722 inactives (0). No information is provided for the test set regarding the distribution.

+ Use your data mining knowledge learned till now wisely to optimize your results.

------------------------------------------------------
Data Description:

The training dataset consists of 800 records and the test dataset consists of 350 records. We provide you with the training class labels and the test labels are held out. The attributes are binary type and as such are presented in a sparse matrix format within train.dat and test.dat


Train data: Training set (a sparse binary matrix), patterns in lines, features in columns: the indices of the non-zero features are provided with class label 1 or 0 in the first column.

Columns seem to go up to 100,000 in number. 

Test data: Testing set (a sparse binary matrix), patterns in lines, features in columns: the indices of non-zero features are provided).

Format example: A sample submission with 350 entries randomly chosen to be 0 or 1. 

------------------------------------------------------

A random input will get 0 on miner and 0.35 on personal test cases. 

Attempts to improve: Categorical data has recommended Chi squared method, however, 
[ 0 0 
  1 1 ] 
type of data is said to be invalid -- it would get NaN for the Chi Squared, which happens all too often fo this kind of data, so this is not recommended.

- One suggestion is to SelectKBest with Chi test, but this requires fit_transform back into numpy, and the result is too large.
- Tried to remove low-variance columns, but 0~20% variance removal got no success.
- importing keras had issues since it can't find tensorflow_internal, for whatever reason. 
- Python torch is only supported on Py 3.7-3.9, 64 bit.
- Gradient boosting is said to improve accuracy, but doesn't seem to help on miner all that much.
------------------------

How to run: use python ./p1-forest.py (while inside /src). The methods are all inside the python code. Training is done within each of the functions. (decision_tree, naive_bayes).

 

