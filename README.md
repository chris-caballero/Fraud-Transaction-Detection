# Credit Card Fraud Detection
This notebook is dedicated to the study and prediction of credit card fraud from a large transaction dataset.
<br>
Follow the link to access the [github page](https://chris-caballero.github.io/Fraud-Transaction-Detection/).

## Imports and Globals

Getting relevant packages for data exploration, processing, and classification. I also read in the transaction csv file and define some colors I will use for my plots throughout this notebook.


```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from IPython.display import HTML, display

dataset_file = 'data/creditcard.csv'

_red = sns.color_palette('hls')[0]
_blue = sns.color_palette('hls')[4]
cmap = [_blue, _red]
nonfraud_patch = mpatches.Patch(color=_blue, label='Non-Fraud')
fraud_patch = mpatches.Patch(color=_red, label='Fraud')

df = pd.read_csv(dataset_file)
```

## A look at the data

A quick glance at our dataframe shows us that there are 3 known attributes: {Time, Amount, Class}, and 28 unknown attributes: {V1,...,V28}.
- Class Name and Class Dist were added by me to allow for better visualizations later on.

No NULL values is good. In a real scenario, this is unlikely to be the case, but pandas provides plenty of tools for dealing with it.


```python
print('Total number of NULL values:', df.isna().sum().sum())
```

    Total number of NULL values: 0


The unknown attributes were obtained by reducing the dimensions of a larger set of known private attributes. 
- This is done for security purposes. 
- In a production environment, feature engineering would need to be done to select the most important private attributes. 
    - Then a dimensionality reduction technique like PCA could be used if we want a similar dataset.
    - This would not match the data we have here, but we could employ similar EDA techniques to study and classify on the data.


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



The **Time** and **Amount** columns can get fairly large (> 20000). I choose to scale the data to regularize it. 
- StandardScaler and RobustScaler are both good choices, I use StandardScaler.


```python
df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>284807.000000</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>...</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>284807.000000</td>
      <td>284807.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>94813.859575</td>
      <td>1.168375e-15</td>
      <td>3.416908e-16</td>
      <td>-1.379537e-15</td>
      <td>2.074095e-15</td>
      <td>9.604066e-16</td>
      <td>1.487313e-15</td>
      <td>-5.556467e-16</td>
      <td>1.213481e-16</td>
      <td>-2.406331e-15</td>
      <td>...</td>
      <td>1.654067e-16</td>
      <td>-3.568593e-16</td>
      <td>2.578648e-16</td>
      <td>4.473266e-15</td>
      <td>5.340915e-16</td>
      <td>1.683437e-15</td>
      <td>-3.660091e-16</td>
      <td>-1.227390e-16</td>
      <td>88.349619</td>
      <td>0.001727</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47488.145955</td>
      <td>1.958696e+00</td>
      <td>1.651309e+00</td>
      <td>1.516255e+00</td>
      <td>1.415869e+00</td>
      <td>1.380247e+00</td>
      <td>1.332271e+00</td>
      <td>1.237094e+00</td>
      <td>1.194353e+00</td>
      <td>1.098632e+00</td>
      <td>...</td>
      <td>7.345240e-01</td>
      <td>7.257016e-01</td>
      <td>6.244603e-01</td>
      <td>6.056471e-01</td>
      <td>5.212781e-01</td>
      <td>4.822270e-01</td>
      <td>4.036325e-01</td>
      <td>3.300833e-01</td>
      <td>250.120109</td>
      <td>0.041527</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-5.640751e+01</td>
      <td>-7.271573e+01</td>
      <td>-4.832559e+01</td>
      <td>-5.683171e+00</td>
      <td>-1.137433e+02</td>
      <td>-2.616051e+01</td>
      <td>-4.355724e+01</td>
      <td>-7.321672e+01</td>
      <td>-1.343407e+01</td>
      <td>...</td>
      <td>-3.483038e+01</td>
      <td>-1.093314e+01</td>
      <td>-4.480774e+01</td>
      <td>-2.836627e+00</td>
      <td>-1.029540e+01</td>
      <td>-2.604551e+00</td>
      <td>-2.256568e+01</td>
      <td>-1.543008e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>54201.500000</td>
      <td>-9.203734e-01</td>
      <td>-5.985499e-01</td>
      <td>-8.903648e-01</td>
      <td>-8.486401e-01</td>
      <td>-6.915971e-01</td>
      <td>-7.682956e-01</td>
      <td>-5.540759e-01</td>
      <td>-2.086297e-01</td>
      <td>-6.430976e-01</td>
      <td>...</td>
      <td>-2.283949e-01</td>
      <td>-5.423504e-01</td>
      <td>-1.618463e-01</td>
      <td>-3.545861e-01</td>
      <td>-3.171451e-01</td>
      <td>-3.269839e-01</td>
      <td>-7.083953e-02</td>
      <td>-5.295979e-02</td>
      <td>5.600000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>84692.000000</td>
      <td>1.810880e-02</td>
      <td>6.548556e-02</td>
      <td>1.798463e-01</td>
      <td>-1.984653e-02</td>
      <td>-5.433583e-02</td>
      <td>-2.741871e-01</td>
      <td>4.010308e-02</td>
      <td>2.235804e-02</td>
      <td>-5.142873e-02</td>
      <td>...</td>
      <td>-2.945017e-02</td>
      <td>6.781943e-03</td>
      <td>-1.119293e-02</td>
      <td>4.097606e-02</td>
      <td>1.659350e-02</td>
      <td>-5.213911e-02</td>
      <td>1.342146e-03</td>
      <td>1.124383e-02</td>
      <td>22.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>139320.500000</td>
      <td>1.315642e+00</td>
      <td>8.037239e-01</td>
      <td>1.027196e+00</td>
      <td>7.433413e-01</td>
      <td>6.119264e-01</td>
      <td>3.985649e-01</td>
      <td>5.704361e-01</td>
      <td>3.273459e-01</td>
      <td>5.971390e-01</td>
      <td>...</td>
      <td>1.863772e-01</td>
      <td>5.285536e-01</td>
      <td>1.476421e-01</td>
      <td>4.395266e-01</td>
      <td>3.507156e-01</td>
      <td>2.409522e-01</td>
      <td>9.104512e-02</td>
      <td>7.827995e-02</td>
      <td>77.165000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>172792.000000</td>
      <td>2.454930e+00</td>
      <td>2.205773e+01</td>
      <td>9.382558e+00</td>
      <td>1.687534e+01</td>
      <td>3.480167e+01</td>
      <td>7.330163e+01</td>
      <td>1.205895e+02</td>
      <td>2.000721e+01</td>
      <td>1.559499e+01</td>
      <td>...</td>
      <td>2.720284e+01</td>
      <td>1.050309e+01</td>
      <td>2.252841e+01</td>
      <td>4.584549e+00</td>
      <td>7.519589e+00</td>
      <td>3.517346e+00</td>
      <td>3.161220e+01</td>
      <td>3.384781e+01</td>
      <td>25691.160000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>




```python
scaler = StandardScaler()
df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
```

These columns are now normalized, which will definitely improve model performance down the line.


```python
df.describe()[['Time', 'Amount']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-3.065637e-16</td>
      <td>2.913952e-17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000002e+00</td>
      <td>1.000002e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.996583e+00</td>
      <td>-3.532294e-01</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-8.552120e-01</td>
      <td>-3.308401e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-2.131453e-01</td>
      <td>-2.652715e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.372174e-01</td>
      <td>-4.471707e-02</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.642058e+00</td>
      <td>1.023622e+02</td>
    </tr>
  </tbody>
</table>
</div>



#### Data Summary - Class Imbalance
***
This data is highly imbalanced! We have over 280k transaction records of which only 0.17% are fraudulent. 
- We will use techniques to balance the data so that our model can properly fit our dataset. 
    - **Undersampling**
        - Random Undersampling:
        - Tomek Links Unersampling:
    - **Oversampling**
        - Sythetic Minority Oversampling Technique (SMOTE):
        - Borderline SMOTE:


```python
def class_distribution(df):
    fraud = round(len(df.loc[df['Class'] == 1]) / len(df) * 100, 2)
    nonfraud = round(len(df.loc[df['Class'] == 0]) / len(df) * 100, 2)
    print('% dataset with class 0: {}%\n% dataset with class 1: {}%\n'.format(nonfraud, fraud))
```


```python
print('Size of dataset: {} samples'.format(len(df)))
class_distribution(df)
```

    Size of dataset: 284807 samples
    % dataset with class 0: 99.83%
    % dataset with class 1: 0.17%
    


## Exploratory Data Analysis

We start by getting a view of the class imbalance with a simple Count plot. Since over 99% of our data is non-fraudulent transactions, it will be almost impossible for our model to do anything other than trivialize the predictions to the majority class.


```python
fig, ax = plt.subplots(figsize=(5, 5))
ax = sns.countplot(x='Class', data=df, palette=cmap)
```


![png](README_files/README_19_0.png)


#### Amount and Time distributions
***
To get a sense of the only two attributes we actually know, I chose to first just see the overall distribution. 
- Interestingly enough, since the data was collected over a two day period, the day/night cycle is apparent. This is a form of **data drift** that happens because people buy less stuff when they are asleep :)
- This will also be reflected, in a production system, through seasonal buying patterns. Therefore, whatever model we fit may perform worse if the data comes from an entirely different time in the year.


```python
def amount_time_distributions(df):
    _, ax = plt.subplots(1, 2, figsize=(20, 6))
    sns.kdeplot(x='Time', ax=ax[0], data=df, color=cmap[0])
    ax2 = ax[0].twinx()
    sns.histplot(x='Time', ax=ax2, data=df, color=cmap[1])
    ax[0].set_title('Distribution of Transactions over Time', fontsize=14)
    ax[0].set_xlim([min(df['Time']), max(df['Time'])])
    
    sns.kdeplot(x='Amount', ax=ax[1], data=df, color=cmap[0])
    ax2 = ax[1].twinx()
    sns.histplot(x='Amount', ax=ax2, data=df, color=cmap[1])
    ax[1].set_title('Distribution of Transactions over Amount', fontsize=14)
    ax[1].set_xlim([min(df['Amount']), max(df['Amount'])])

    plt.show()

amount_time_distributions(df)
```


![png](README_files/README_21_0.png)


#### Correlation Matrix
***
Given the huge number of 0 samples, it is hard for the correlation matrix to pick up anything. I will perform these same tests on the balanced dataset and compare the differences.


```python
fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.heatmap(df.corr())
```


![png](README_files/README_23_0.png)


## Train and Testset Creation
***
Before balancing the data, I will be splitting it into a training set and a testing set. The test set should come from the original distribution, so we use train_test_split and stratify based on the class distribution.
- The output shows that our distributions are the same. We have less than 100 positive samples in the test set, but this should still be enough to get a gauge of how well the model performs later. 


**holdout_df** will contain the test samples and will not be touched outside model evaluation later.


```python
from sklearn.model_selection import train_test_split

X = df.drop(labels=['Class'], axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

train_df = pd.concat([X_train, y_train], axis=1)
holdout_df = pd.concat([X_test, y_test], axis=1)

print('TRAINING SET')
print('- Fraud class size: {}\n- Non-Fraud class size: {}'.format(len(train_df[train_df.Class == 1]), len(train_df[train_df.Class == 0])))
class_distribution(train_df)

print('TESTING SET')
print('- Fraud class size: {}\n- Non-Fraud class size: {}'.format(len(holdout_df[holdout_df.Class == 1]), len(holdout_df[holdout_df.Class == 0])))
class_distribution(holdout_df)
```

    TRAINING SET
    - Fraud class size: 394
    - Non-Fraud class size: 227451
    % dataset with class 0: 99.83%
    % dataset with class 1: 0.17%
    
    TESTING SET
    - Fraud class size: 98
    - Non-Fraud class size: 56864
    % dataset with class 0: 99.83%
    % dataset with class 1: 0.17%
    


## Random Undersampling

The main technique employed here, we simply select an equal number of non-fraud samples as we do fraud samples. 

Performing random undersampling! Just randomly choose equal portion non-fraud samples and fraud samples. Since there are more non-fraud samples, we only have to portion down this field, and we keep all the fraud-samples.


```python
new_df = train_df.groupby('Class', group_keys=False).apply(lambda x: x.sample(len(train_df[train_df.Class == 1])))
print(len(new_df))
```

    788


### EDA - Random Undersampling

The Count plot shows that our dataset is now evenly distributed with less than 800 samples remaining.


```python
fig, ax = plt.subplots(figsize=(6, 5))
ax = sns.countplot(x='Class', data=new_df, palette=cmap)
plt.title('Balanced Dataset')
plt.show()
```


![png](README_files/README_32_0.png)


#### Correlation Matrix
***
- Now that the data is balanced, the correlation matrix is showing some useful information!
- When comparing the correlations of {V1,...,V28} to the Class (Fraudulent or Non-Fraudulent), we notice that features V1-V18 appear to have the strongest correlations.
    - I expect these will be helpful predictors for the model later on.
- Features like V21 and V22 are strongly correlated, neither seems to be correlated to the Class in any particular way, so this information alone may not be directly useful for predicting the class. 
- I will focus visualizations on the features which have a decent correlation with the Class
    - correlation $\geq 0.6$:
    - The specific intervals can be modified to consider more or less features.
***
The code I wrote allows for more modular data exploration, allowing for more or less features to be considered from trimming and analysis depending on the values chosen. Feel free to change these values to see how other features correlate to the class and remove outliers from them as well!

Correlation matrix on balanced data. Features with strong correlation with Class are selected for further exploration.


```python
corr = abs(new_df.corr())

fig, ax = plt.subplots(figsize=(25, 20))
ax = sns.heatmap(corr, annot=True, fmt=".2f")
```


![png](README_files/README_35_0.png)


#### We can see some features with very high correlations
- Some of them may be removed if we want to reduce the dimensionality further.
- I will be testing model performance with and without some of these.
- From here, it looks like:
    - $V11 \sim V12 \sim V14$
    - $V12 \sim V16$
    - $V16 \sim V17 \sim V18$ 
    - where A ~ B reads A is strongly correlated to B.
    - Keeping $V11, V12, V16$ should preserve most of the information from the 6 of them. 


```python
new_corr = abs(new_df[['V11', 'V12', 'V14', 'V16', 'V17', 'V18']].corr())
sns.heatmap(new_corr, annot=True, fmt='.2f')
plt.show()
```


![png](README_files/README_37_0.png)


#### Here I define some helper functions to add attributes to our dataframe
- Class Name: Used for identifying classes of examples.
- Class Dist: Used for plotting, a distribution with variance > 0 is needed for the plots we use!


```python
def create_class_names(df, col='Class Name'):
    df.loc[df['Class'] == 1, col] = 'Fraud'
    df.loc[df['Class'] == 0, col] = 'Non-Fraud'
    
    return df

def create_class_distribution(df, col='Class Dist'):
    zero_class_dist = np.random.normal(loc=0, scale=0.1, size=len(df.loc[df['Class'] == 0]))
    one_class_dist = np.random.normal(loc=1, scale=0.1, size=len(df.loc[df['Class'] == 1]))

    df.loc[df['Class'] == 0, col] = zero_class_dist
    df.loc[df['Class'] == 1, col] = one_class_dist
    
    return df

new_df = create_class_names(new_df)
new_df = create_class_distribution(new_df)
```

#### Now we define a function to get and display the featurs with a correlation with the class of $\geq$ 0.6
- This will be used for statistical purposes. We want to display plots for a relatively small subset of the highest correlated features.
- The correlation here is strictly with Class.


```python
def get_high_corr_feats(df, min_val, excluded_feats=['Class', 'Class Name', 'Class Dist'], summary=True):
    class_corr = abs(df.corr()['Class'])
    # don't include class for this (1 correlation with itself)
    corr_feats = class_corr.loc[(class_corr >= min_val) & (~class_corr.index.isin(excluded_feats))]
    
    if summary:
        print('Features Strongly Correlated with Class')
        for idx in corr_feats.index:
            print(idx, corr_feats[idx])

    return corr_feats
```

We have 5 features to plot.


```python
corr_feats = get_high_corr_feats(new_df, min_val=0.69)
```

    Features Strongly Correlated with Class
    V4 0.7145431553816947
    V14 0.7487031417936583


#### Joint Distribution
***
This plot is why we added the **Class Dist** attribute earlier. To plot the joint distribution, the kdeplot needs a variable with a non-trivial standard deviation. By creating a variable which approximates the distribution of class labels, 0 and 1, with a very small variance, we can see the relative distributions in an intuitive way. 
- This is similar to a boxplot but for the purposes of this project, I chose to visualize it this way as I find it more beautiful. 
- Box-plot would likely be easier to work with and better for having a direct comparison of the distributions (box plot can work with integer class labels).


```python
def plot_joint_distribution(df, corr_feats, x_label='Class Dist', hue='Class Name'):
    fig, axs = plt.subplots(1, len(corr_feats), figsize=(20, 5))
    for i, feat in enumerate(corr_feats.index):
        sns.kdeplot(data=df, x=x_label, y=feat, ax=axs[i], hue=hue, palette=cmap)
        axs[i].set_title('{} vs. Class Distribution - Correlation: {}'.format(feat, round(corr_feats[feat], 3)))
        plt.suptitle('Joint Distribution of Features and Class\nPositive Correlation with Class')

    plt.tight_layout(w_pad=10.0)
    plt.show()
```

V4 and V11 seem to be positively correlated with class while the rest are negatively correlated.
- The type of correlation is not relevant since we don't know what any of the features are. Taking the absolute value allows us to find the most correlated features indiscriminantly.


```python
plot_joint_distribution(new_df, corr_feats)
```


![png](README_files/README_47_0.png)


#### Histograms of the number of transactions over time and amount
We find the same circadian cycle for transactions over time.
- We don't expect this to change much since we randomly sampled the dataset. 
- This just has less points to use to plot the distribution.


```python
amount_time_distributions(new_df)
```


![png](README_files/README_49_0.png)


#### Removing Outliers
***
It is important for the training data to capture the general distribution of the data as best as possible. One way to do this is by removing examples that are extreme (outliers) from the training set. We can do this by capturing the interquartile range of the data along particular attributes, and using a threshold to apply a bounded filter to the data. 
- This will remove all the datapoints that fall outside some inner-range of the data's distribution. 
- The threshold we use is 1.5, giving bounds:
    - lower_bound = lower_quartile - threshold * interquartile_range
    - upper_bound = upper_quartile + threshold * interquartile_range


```python
def get_irq_from_dataframe(df):
    lower_quartile = df.quantile(0.25)
    upper_quartile = df.quantile(0.75)
    return upper_quartile - lower_quartile, lower_quartile, upper_quartile

def trim_outliers(df, irq, lower_quartile, upper_quartile, slice, threshold=1.5):
    lower_bound = lower_quartile - threshold * irq
    upper_bound = upper_quartile + threshold * irq

    outlier_indices = slice[(slice > upper_bound) | (slice < lower_bound)].index
    
    df = df.drop(outlier_indices)

    return df, lower_bound, upper_bound

def show_outliers(slice, lower_bound, upper_bound):
    outlier_indices = slice[(slice > upper_bound) | (slice < lower_bound)].index
    noutliers = len(outlier_indices)
    
    outliers = [x for x in slice[outlier_indices]]
    
    print('{} # outliers: {}'.format(slice.name, noutliers))
    print('{} outliers: {}'.format(slice.name, outliers))
    
def trim_feature_outliers(df, feats, threshold=1.5, summary=True):
    trimmed_df = df.copy()

    for feat in feats:
        feat_fraud = trimmed_df[feat].loc[trimmed_df['Class'] == 1]
        irq, lower_quartile, upper_quartile = get_irq_from_dataframe(feat_fraud)
        
        if summary:
            print(f'FEATURE {feat}:')
            print('Dataset size (pre-trim): ', len(trimmed_df))
            print('\nInterquartile Range: {}\nQuartiles: {}\n'.format(irq, [lower_quartile, upper_quartile]))
            
        trimmed_df, lower_bound, upper_bound = trim_outliers(trimmed_df, irq, lower_quartile, upper_quartile, slice=feat_fraud, threshold=threshold)
        show_outliers(feat_fraud, lower_bound, upper_bound)
        
        if summary:
            print('Bounds: {}\n'.format([lower_bound, upper_bound]))
            print('Dataset size (post-trim): ', len(trimmed_df))
            print('---'*45)
    
    return trimmed_df

def plot_outliers_removed(df, feats, corr_label='Positively'):
    fig, axs = plt.subplots(1, len(feats), figsize=(5*len(feats),5))
    
    for i, feat in enumerate(feats.index):
        sns.boxplot(x="Class", y=feat, data=df, ax=axs[i], palette=cmap)
        axs[i].set_title(f"{feat} Feature Distribution\nOutliers Removed", fontsize=14)
        axs[i].legend(handles=[nonfraud_patch, fraud_patch])
        
    plt.suptitle(f'Features {corr_label} Correlated with Class', fontsize=16)
    plt.tight_layout(w_pad=5.0)
    plt.show()
```

The resulting dataframe (with less outliers), is stored into trimmed_df
- The output shows the features we remove outliers from. 
- For each feature in **feats** we capture the distribution of that feature across the dataset, **new_df**.


```python
feats = corr_feats.index
# feats = [feat for feat in new_df.columns if feat not in ['Class', 'Class Dist', 'Class Name', 'Time', 'Amount']]
trimmed_df = trim_feature_outliers(new_df, feats=feats, threshold=1.4, summary=True)
```

    FEATURE V4:
    Dataset size (pre-trim):  788
    
    Interquartile Range: 3.9201573614314205
    Quartiles: [2.421115118476715, 6.3412724799081355]
    
    V4 # outliers: 5
    V4 outliers: [11.9061699078901, 11.8648680803607, 11.8447765860728, 11.927511869244, 11.8450129100508]
    Bounds: [-3.0671051875272735, 11.829492785912123]
    
    Dataset size (post-trim):  783
    ---------------------------------------------------------------------------------------------------------------------------------------
    FEATURE V14:
    Dataset size (pre-trim):  783
    
    Interquartile Range: 5.27091548842971
    Quartiles: [-9.56016927087188, -4.28925378244217]
    
    V14 # outliers: 9
    V14 outliers: [-17.6206343516773, 3.44242199594215, -18.4937733551053, -18.8220867423816, -17.230202160711, -18.0499976898594, -17.4759212828566, -19.2143254902614, -17.7216383537133]
    Bounds: [-16.93945095467347, 3.0900279013594227]
    
    Dataset size (post-trim):  774
    ---------------------------------------------------------------------------------------------------------------------------------------


With the outliers removed for the given features, we can now visualize the new distributions with boxplots.
- We apply outlier removal to the 'Fraud' class. 
- This can be contrasted with the 'Non-Fraud' class which did not have any outliers removed.
    - There are far less outliers in the Fraud plot than the Non-Fraud.


```python
plot_outliers_removed(trimmed_df, corr_feats, corr_label='Strongly')
```


![png](README_files/README_55_0.png)


#### Dimensionality Reduction

By reducing dimensionality, we can look for signs of clustering inherent to the data.
- The data we have processed so far will be reduced to 2-dimensions and plotted. 
- The points will be colored with respect to their class.


```python
def plot_reduced_dimensions(df, method='tsne'):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle('Dimensionality Reduction Clustering Visualization')

    X = df.drop(['Class', 'Class Name', 'Class Dist'], axis=1)
    y = df['Class']

    if method == 'tsne':
        X_tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=40).fit_transform(X)
        ax.scatter(X_tsne[:, 0], X_tsne[:,1], c=(y == 0), cmap='coolwarm', label='Non-Fraud')
        ax.scatter(X_tsne[:, 0], X_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud')
        ax.set_title('t-SNE')
        ax.legend(handles=[nonfraud_patch, fraud_patch])
        plt.show()
        
        return X_tsne
    
    elif method == 'pca':
        X_pca = PCA(n_components=2).fit_transform(X)
        ax.scatter(X_pca[:, 0], X_pca[:,1], c=(y == 0), cmap='coolwarm', label='Non-Fraud')
        ax.scatter(X_pca[:, 0], X_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud')
        ax.set_title('PCA')
        ax.legend(handles=[nonfraud_patch, fraud_patch])
        
        return X_pca
    
    else:
        print('Error: Invalid dimensionality reduction method. Please choose a method from ["tsne", "pca"].')
    
    return None
```

#### t-SNE shows clear clustering with respect to class.
- The classes seem reasonably separated about x = 5. 
- There is some ambiguity, but this is bound to happen when we reduce the dimensionality so much.
#### PCA shows relatively good clustering. The boundary is not as clear as in t-SNE.
- Non-Fraud samples are tighly grouped along the principle components.
- Fraud samples have a wider range of values, but tend to hover left of the non-fraud cluster. 
- The ambiguity is greater here, but at least it appears the data is mostly separable.


```python
X_tsne = plot_reduced_dimensions(trimmed_df, method='tsne')
X_pca = plot_reduced_dimensions(trimmed_df, method='pca')
```


![png](README_files/README_60_0.png)



![png](README_files/README_60_1.png)


## Model Selection and Training
***
It is important to test simple and effective models on this data to get an idea of how good the data is, and establish some baseline performance. 
- RandomForestClassifer. We may choose to use this for future feature engineering as we can get feature importance inherent to the class. 
- LogisticRegression. This classifier will find the best decision boundary separating the data according the a probability that it belongs to a given class.
- SVM. This will simply find the decision boundary that maximizes the margin from data points to the hyperplane separating them.

#### Imports and Helper Functions


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve
```


```python
def evaluate_rfc_crossval(model, X, y, cv=5):
    """
    Evaluates a RandomForestClassifier using cross-validation and returns the average accuracy score.
    
    Parameters:
    - X: The input features as a 2D array or DataFrame.
    - y: The target variable as a 1D array or Series.
    - cv: The number of cross-validation folds (default: 5).
    
    Returns:
    - The average accuracy score across the cross-validation folds.
    """
    
    scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'roc_auc': 'roc_auc'}
    
    # Perform cross-validation and compute the scores
    scores = cross_validate(model, X, y, scoring=scoring, cv=cv)

    avg_accuracy = scores['test_accuracy'].mean()
    avg_precision = scores['test_precision'].mean()
    avg_recall = scores['test_recall'].mean()
    avg_roc_auc = scores['test_roc_auc'].mean()
    
    # Return the average accuracy score
    return {'accuracy': avg_accuracy, 'precision': avg_precision, 'recall': avg_recall, 'roc_auc': avg_roc_auc}

def evaluate_model(model, X, y, label='Random Forest', show_cm=True):
    y_pred = model.predict(X)
    
    acc, precision, recall, roc = accuracy_score(y, y_pred), precision_score(y, y_pred), recall_score(y, y_pred), roc_auc_score(y, y_pred)

    scores = [acc, precision, recall, roc]
    scores = [round(score, 4) for score in scores]
    scores_df = pd.DataFrame([scores], columns=['Accuracy', 'Precision', 'Recall', 'AUC-ROC'])

    if show_cm:
        _confusion_matrix = plot_confusion_matrix(y, y_pred, label=label)
    else:
        _confusion_matrix = confusion_matrix(y, y_pred)
    
    # Return the average accuracy score
    return scores_df, _confusion_matrix

def plot_confusion_matrix(y_true, y_pred, label='Random Forest'):
    _confusion_matrix = confusion_matrix(y_true, y_pred)

    # Maps the confusion matrix so each row is a distribution for that row. Easier to visualize.
    # confusion_matrix_ = confusion_matrix_.astype('float') / confusion_matrix_.sum(axis=1).reshape(2, 1)

    sns.heatmap(_confusion_matrix, annot=True)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix - {label}')
    plt.show()
    
    return _confusion_matrix

def get_splits_from_dataframe(df, drop_cols=['Class', 'Class Name', 'Class Dist'], label='Class', test_size=0.2, random_state=42):
    X = df.drop(drop_cols, 1)
    y = df[label]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(model, data, val_data=None, drop_cols=['Class', 'Class Name', 'Class Dist'], label='Logistic Regression', show_cm=True):
    if val_data is None:
        X_train, X_test, y_train, y_test = get_splits_from_dataframe(data, drop_cols=drop_cols)
    else:
        X_test = val_data.drop(drop_cols, axis=1)
        y_test = val_data['Class']
        
        # select proper columns from the dataset (data) 
        # and find samples that exclude the testset (val_data)
        mask = ~data.isin(val_data).all(axis=1)
        X_train = data[mask].drop(drop_cols, axis=1)
        y_train = data[mask]['Class']
        
    model.fit(X_train, y_train)
    scores, _confusion_matrix = evaluate_model(model, X_test, y_test, label=label, show_cm=show_cm)
    return scores, _confusion_matrix

def train_and_evaluate_models(models, data, val_data=None, drop_cols=['Class', 'Class Name', 'Class Dist']):
    fig, axs = plt.subplots(1, len(models), figsize=(20, 5))
    scores = []
    for i, name in enumerate(models):
        model = models[name]
        
        score_df, _confusion_matrix = train_and_evaluate_model(model, data, val_data, label=name, show_cm=False, drop_cols=drop_cols)
        score_df.insert(0, 'Model', name)
        scores.append(score_df)

        sns.heatmap(_confusion_matrix, ax=axs[i], annot=True)
        axs[i].set_xlabel('Predicted label')
        axs[i].set_ylabel('True label')
        axs[i].set_title(f'Confusion Matrix - {name}')
    
    plt.suptitle('Confusion Matrix by Model')
    plt.show()
        
    scores_df = pd.concat(scores)
    
    return scores_df
```

#### Model Initialization
- We define our models here and add them to a dictionary so we can easily compare results.
- These parameters are all optional, feel free to change them to see how they affect the performance.
- The confusion matrix is calculated on the test set derived from the training set. 
- Results on the entire holdout set we split earlier in the project are difficult to interpret this way since the class imbalance is so large and there are so many samples. 
    - The performance of the model on the holdout set is best visualized through an ROC plot.


```python
def create_models():
    rfc = RandomForestClassifier(
        n_estimators=500,
        min_samples_split=0.02,
        min_samples_leaf=1, 
        class_weight='balanced', 
        max_features='sqrt'
    )

    lr = LogisticRegression(
        class_weight='balanced', 
        solver='liblinear',
        multi_class='ovr'
    )

    svc = SVC(
        C = 5.0,
        class_weight='balanced', 
        probability=True
    )

    return {'Random Forest': rfc, 'Logistic Regression': lr, 'Support Vector Classifier': svc}
```

#### Model Training and Evaluation

Evaluating a couple of models on the entire dataset shows that we mostly trivialize the solution. 
- Evaluating this on a balanced split shows how bad the models actually are at predicting the class. 
- It is approaching random assignment (some runs less than 0.7 AUC-ROC).


```python
models = {'Logistic Regression': LogisticRegression(), 'Support Vector Classifier': SVC()}
val_df = new_df.drop(['Class Name', 'Class Dist'], axis=1)
scores = train_and_evaluate_models(models, data=df, val_data=val_df, drop_cols=['Class'])
HTML(scores.to_html(index=False))
```


![png](README_files/README_69_0.png)





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>AUC-ROC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Logistic Regression</td>
      <td>0.6992</td>
      <td>1.0</td>
      <td>0.3985</td>
      <td>0.6992</td>
    </tr>
    <tr>
      <td>Support Vector Classifier</td>
      <td>0.6802</td>
      <td>1.0</td>
      <td>0.3604</td>
      <td>0.6802</td>
    </tr>
  </tbody>
</table>



Based on our test set (model was not fit on these samples), we can see that the model is making very good predictions all around. The model tends to classify some fraud examples as non-fraud, but the degree to which this occurs is very low. 
- We are achieving around .97 precision and 0.85 recall for each model! This is also somewhat subject to random initializations.
- High precision is good for this problem since we don't want the model flagging everything as fraudulent (potentially cancelling valid transactions if we were to integrate this model into a bank system).
    - If the precision is high and the recall is low, we will miss a lot of fraudulent transactions. 
    - The recall scores are decent so the models are performing well.


```python
models_balanced_data = create_models()
trainset = new_df.drop(['Class Name', 'Class Dist'], 1)
balanced_scores1 = train_and_evaluate_models(models_balanced_data, data=trainset, drop_cols=['Class'])
balanced_scores2 = train_and_evaluate_models(models_balanced_data, data=trainset, val_data=holdout_df, drop_cols=['Class'])
```


![png](README_files/README_71_0.png)



![png](README_files/README_71_1.png)


Scores for validation set (balanced dataset **new_df**)


```python
HTML(balanced_scores1.to_html(index=False))
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>AUC-ROC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Random Forest</td>
      <td>0.9304</td>
      <td>0.9737</td>
      <td>0.8916</td>
      <td>0.9324</td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>0.9367</td>
      <td>0.9620</td>
      <td>0.9157</td>
      <td>0.9378</td>
    </tr>
    <tr>
      <td>Support Vector Classifier</td>
      <td>0.9241</td>
      <td>0.9610</td>
      <td>0.8916</td>
      <td>0.9258</td>
    </tr>
  </tbody>
</table>



Scores for holdout set


```python
HTML(balanced_scores2.to_html(index=False))
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>AUC-ROC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Random Forest</td>
      <td>0.9773</td>
      <td>0.0647</td>
      <td>0.9082</td>
      <td>0.9428</td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>0.9707</td>
      <td>0.0514</td>
      <td>0.9184</td>
      <td>0.9446</td>
    </tr>
    <tr>
      <td>Support Vector Classifier</td>
      <td>0.9810</td>
      <td>0.0765</td>
      <td>0.9082</td>
      <td>0.9446</td>
    </tr>
  </tbody>
</table>



Recall seems to improve with the data that was trimmed of outliers. Often times outliers negatively affect model performance since they make it harder to learn the distribution.
- AUC-ROC also goes up, showing that the model improved because of the data.
- On the holdout set, we see that the model flags a lot of cases as fraudulent, which is not ideal. But there is inherent ambiguity in the data to begin with - so we may be approaching baseline performance. 


```python
models_trimmed_data = create_models()
trainset = trimmed_df.drop(['Class Name', 'Class Dist'], 1)
trimmed_scores1 = train_and_evaluate_models(models_trimmed_data, data=trainset, drop_cols=['Class'])
trimmed_scores2 = train_and_evaluate_models(models_trimmed_data, data=trainset, val_data=holdout_df, drop_cols=['Class'])
```


![png](README_files/README_77_0.png)



![png](README_files/README_77_1.png)


Scores for validation set (comparing **new_df** *top* with **trimmed_df** *bottom*)
- trimmed_df has a more balanced performance across the board. Its the winner!
- I enjoy seeing how simple tweaks to the data can improve performance in the same model.


```python
display(HTML(balanced_scores1.to_html(index=False)), HTML(trimmed_scores1.to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>AUC-ROC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Random Forest</td>
      <td>0.9304</td>
      <td>0.9737</td>
      <td>0.8916</td>
      <td>0.9324</td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>0.9367</td>
      <td>0.9620</td>
      <td>0.9157</td>
      <td>0.9378</td>
    </tr>
    <tr>
      <td>Support Vector Classifier</td>
      <td>0.9241</td>
      <td>0.9610</td>
      <td>0.8916</td>
      <td>0.9258</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>AUC-ROC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Random Forest</td>
      <td>0.9355</td>
      <td>0.9241</td>
      <td>0.9481</td>
      <td>0.9356</td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>0.9484</td>
      <td>0.9259</td>
      <td>0.9740</td>
      <td>0.9486</td>
    </tr>
    <tr>
      <td>Support Vector Classifier</td>
      <td>0.9484</td>
      <td>0.9481</td>
      <td>0.9481</td>
      <td>0.9484</td>
    </tr>
  </tbody>
</table>


Scores both previous models being evaluated on the entire holdout set


```python
display(HTML(balanced_scores2.to_html(index=False)), HTML(trimmed_scores2.to_html(index=False)))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>AUC-ROC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Random Forest</td>
      <td>0.9773</td>
      <td>0.0647</td>
      <td>0.9082</td>
      <td>0.9428</td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>0.9707</td>
      <td>0.0514</td>
      <td>0.9184</td>
      <td>0.9446</td>
    </tr>
    <tr>
      <td>Support Vector Classifier</td>
      <td>0.9810</td>
      <td>0.0765</td>
      <td>0.9082</td>
      <td>0.9446</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Model</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>AUC-ROC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Random Forest</td>
      <td>0.9773</td>
      <td>0.0641</td>
      <td>0.8980</td>
      <td>0.9377</td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>0.9696</td>
      <td>0.0497</td>
      <td>0.9184</td>
      <td>0.9440</td>
    </tr>
    <tr>
      <td>Support Vector Classifier</td>
      <td>0.9804</td>
      <td>0.0742</td>
      <td>0.9082</td>
      <td>0.9443</td>
    </tr>
  </tbody>
</table>


#### ROC Curves
***
To see how the model performs on the original holdout set (+50k examples with high class imbalance)


```python
def plot_roc_curves(curves, label='Classifiers'):
    def plot_roc_curve(fpr, tpr, label=None, color='b'):
        plt.plot(fpr, tpr, label=label, color=color)
    
    plt.figure(figsize=(7, 5))
    plt.title(f'ROC Curve for {label}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([-.05, 1, 0, 1.05])
    
    palette = sns.color_palette('hls', 8)
        
    for i, name in enumerate(curves):
        fpr, tpr, _ = curves[name]
        plot_roc_curve(fpr, tpr, label=name, color=palette[(i+1)*2])
    
    plt.legend()
    plt.show()
        

def get_roc_curves(models, X, y):
    curves = {}
    
    for name in models:
        y_pred_proba = models[name].predict_proba(X)
        fpr, tpr, threshold = roc_curve(y, y_pred_proba[:, 1])
        
        curves[name] = (fpr, tpr, threshold)
    
    return curves
```


```python
roc_curves = get_roc_curves(models_balanced_data, X_test, y_test)
plot_roc_curves(roc_curves, label='Classifiers - Balanced Dataset')
```


![png](README_files/README_84_0.png)



```python
roc_curves = get_roc_curves(models_trimmed_data, X_test, y_test)
plot_roc_curves(roc_curves, label='Classifiers - Outliers Removed')
```


![png](README_files/README_85_0.png)

