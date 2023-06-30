# Credit Card Fraud Detection

This notebook is dedicated to the study and prediction of credit card fraud from a large transaction dataset.
Follow the link to access the [github page](https://chris-caballero.github.io/Fraud-Transaction-Detection/) to see the notebook and results!

## Requirements
Make sure you have the following packages if you want to run the notebook locally.
```
- torch
- seaborn
- pandas
- numpy
- matplotlib
- scikit-learn
- ipython
```

## Goal
***
The goal of this project is to get direct experience performing data exploration and analysis on imbalanced data. Improving data quality and consistency is paramount in developing effective models in production environments, so I hoped to try some techniques for handling data (such as sampling techniques and interquartile range outlier removal). I was able to generate some visualizations showcasing the distribution of key features and the joint distributions of those highly correlated with the class.
<br>

I used a variety of statistical models: Logistic Regression, Support Vector Classifier and Random Forest. To see which performed best and had the most significant transfer of performance to the validation set. I then created a simple multilayer perceptron neural network with PyTorch to peform classification and compare results. Most results can be found in the github pages linked at the top of the README.

## Results
***
Here I will display the visualizations which capture key aspects of the project.

### **Exploratory Data Analysis**

- Class Imbalance, before and after Random Undersampling:
<img src="README_files/README_19_0.png" alt="Description" width="300" height="300">
<img src="README_files/README_32_0.png" alt="Description" width="300" height="300">

- Correlation Matrix (balanced dataset):
<br>
<img src="README_files/README_35_0.png" alt="Description" width="500" height="400">

- Correlation Matrix (balanced dataset) - Highly correlated features:
    - In the future I will use this information to perform feature selection.
<br>
<img src="README_files/README_37_0.png" alt="Description" width="300" height="250">

- Known features (Time and Amount) Distributions:
    - The circadian cycle is apparent in the distribution of transactions over time. People buy less stuff at night!
    - Important to note that this data was collected over the period of two days and nights, which explains why this cycle is noticeable.
<br>
<img src="README_files/README_49_0.png" alt="Description" width="600" height="200">

### **Feature Engineering**

Outlier removal using interquartile range and threshold:
- The outliers were only removed for the positive class, Fraud. 
- The outliers are clearly visible in the Non-Fraud boxplot, outside the quartile lines.
<br>
<img src="README_files/README_55_0.png" alt="Description" width="550" height="300">

### **Dimensionality Reduction (PCA and t-SNE)**
Reducing dimensionality allows us to see if the data naturally clusters. From the images below, while there is some ambiguity, the data clusters well.
<br>
<img src="README_files/README_60_0.png" alt="Description" width="300" height="250">
<img src="README_files/README_60_1.png" alt="Description" width="300" height="250">

### **Model Performance**
**NOTE:** Full results, including precision, recall and auc-roc scores, can be found in the notebook execution.
<br>
First I chose to visualize the performance of a couple models trained with imbalanced data.
- Train on the whole dataset (excluding the balanced dataset).
- Evaluate on the balanced dataset (easier to interpret).
- The model is mostly predicting 0, doesn't look like it fit the data.
<img src="README_files/README_69_0.png" alt="Description" width="550" height="250">

Then I train and evalute the model on the balanced dataset (using train_test_split).
- The first uses the data without outliers removed.
<img src="README_files/README_71_0.png" alt="Description" width="600" height="200">
- The second uses the data with outliers removed.
<img src="README_files/README_77_0.png" alt="Description" width="600" height="200">

Lastly, we evaluate the ROC Curve performance on the entire holdout set (imbalanced validation set):
<img src="README_files/README_84_0.png" alt="Description" width="500" height="400">


## Future Work
***
- Create Docker version with necessary requirements to generate consistent results.
- Try synthetic transaction data generation: https://github.com/namebrandon/Sparkov_Data_Generation
    - This will give a sense of realism, applying this to data which looks and behaves like what you would find in a real world setting.
    - The original data is anonymized which makes understanding the data much harder.
- Try various sampling techniques (SMOTE, Tomek Links, ...)
- Create a new notebook exploring all the same stuff using **Tensorflow Extended (TFX)** and **Tensorflow Transform (TFT)** for scalable feature selection, feature engineering and data/execution monitoring.

