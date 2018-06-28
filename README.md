# Recommendation
Developing Recommendation Engine using Python

1. Fetch movie datasets from LightFM.datasets.
2. Fetch training and testing data from the dataset.
3. Train the model (warp - Weighted Approximate Rank Pairwise) using training data.
4. Get known positives (movies they already love) with the help of Training data.
5. Predict Movies they may like with the help of model generated in setp 3.
6. Compare the result.

Libraries:

1. numpy:
  - Fundamental package for scientific computing.
  - Powerful n-dimensional array object
   
2. lightfm:
  - Python implementation of a number of popular recommendation algorithm.
  - It represents each user and item as sum of latent representatios of their features. Thus allowing recommendations to generalise to new     features and to new users.
  
