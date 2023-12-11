# Customer Segmentation Analysis Report 
## Submitted by: Mariano Viola, PhD 

## Executive Summary
>The goal of this customer segmentation analysis report is to help the Company to get to know their customer base better so they can identify specific groups for product development or product upsell or to communicate a targeted marketing message. We used Python programming to build several models that identify and segment various cluster groups based on their demographic variables. While each cluster group has unique characteristics that differentiate them from one another, members of the same group are similar to one another. Each customer segmentation model is described in detail and their analytical results are visualized in various charts and graphs. 

### What is Customer Segmentation and why is it useful?

- Customer segmentation is the process of dividing a company’s customers into groups based on common characteristics so companies can market to each group effectively and appropriately. 
- In business-to-consumer marketing, companies often segment customers according to demographics that include: 
    - Age
    - Gender
    - Marital status
    - Location (urban, suburban, rural) 
    - Life stage (single, married, divorced, empty-nester, retired, etc.) 
    
### Segmentation allows marketers to better tailor their marketing efforts to various audience subsets. Those efforts can relate to both communications and product development. Specifically, segmentation helps a company: 

- Create and communicate targeted marketing messages that will resonate with specific groups of customers 
- Select the best communication channel for the segment, which might be email, social media posts, radio advertising, or another approach, depending on the segment 
- Identify ways to improve products or new product or service opportunities 
- Establish better customer relationships 
- Test pricing options 
- Focus on the most profitable customers 
- Improve customer service 
- Upsell and cross-sell other products and services 

### How do we build Customer Segmentation Models? 

- We used Python programming to leverage big data analytics on the company's customer data set. 
- We employed the following Python programming algorithms to enable us to effectively segment customer data: 
    1. Hierarchical clustering: to cluster the customer data and identify the number of different groups;
    2. K-means segmentation: to segment the customer groups and analyze key characteristics of the group and its members;
    3. Principal Component Analysis (PCA): to reduce dimensionality in the data set and identify key common characteristics that explain the main variations in the data set. 

### What are the Key Findings and Insights of our Customer Segmentation Study? 
- We have identified 4 customer cluster groups that each exhibit their own characteristics: 
    1. Standard segment;
    2. Career-focus segment;
    3. Fewer Opportunities segment, and
    4. Well-off segment.

### Key Characteristics of each customer cluster segment: 

- The largest segment with 692 individuals is the **standard cluster segment**. It represents roughly around 35% of the total data set. Members belonging to this group are mostly married females with an average age of around 29 years. They're mostly high school graduates employed in skilled or official positions (e.g., office assistants) and have an average income of around 106,000. They mostly reside in small to mid-sized cities. **Individuals in the standard segment tend to put a high value on lifestyle. They are not particularly career conscious nor are they interested in their work or life experience.** 

- The second largest group in the data set is the **career focus cluster** group with 583 people or 29% of the whole data set. 
Most members are single males with an average age of 35.6 years with a high school education level; their average income is around 141,218 and they tend to be self-employed or in skilled to highly skilled jobs. They mostly reside in mid-sized to big cities. **Career focused individuals tend to put a high emphasis on their career goals and are not so interested in lifestyle or their life/work experiences.**  

- The 3rd largest group is the **fewer opportunities cluster** with 460 people making up about 23.0% of the total dataset. 
Group members are about 70% male, mostly single with an average age of around 35.3 years with a high school education level or below. The group average income is around 93,692 and individuals are mostly employed in low-skilled jobs or are unemployed. The average member tends to reside in small cities. **Individuals in this group put a high emphasis on their work and life experience and put a low emphasis on career goals and education and lifestyle values.** 

- The smallest segment is the **well-off cluster** group with about 265 members making up around 13% of the total data set. This group is divided equally between male and female, and members tend to be married, divorced or separated. They tend to be highly educated (University level) with a high average income of around 158,338. Most are employed in skilled to highly skilled jobs and they tend to reside in mid-sized to big cities. **Individuals in this group are career focused and put a high emphasis on education and lifestyle. However, they put low values on work and life experience.** 

## Directories


```python
%cd "C:\Users\maria\OneDrive\Desktop\Customer_Segmentation_Analytics"
```

    C:\Users\maria\OneDrive\Desktop\Customer_Segmentation_Analytics
    

## Libraries


```python
import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

import pickle

import warnings
warnings.filterwarnings("ignore")
```

## Import Data 


```python
# Load the data, contained in the segmentation data csv file
df_segmentation = pd.read_csv('segmentation_data.csv', index_col = 0)
```

## Explore the Data

### Data Legend - describes each variable or feature in our data set. 
- **ID** - shows the customer ID
- **Sex** - Customer gender
    - 0 male
    - 1 female
- **Marital Status**
    - 0 Single
    - 1 non-Single (divorced/separated/married/widowed)
- **Age** - customer age in years (current Yr. - DOB at time of data creation)
    - 18 - minimum value (lowest age observed in the dataset)
    - 76 - maximum value (highest age observed in the dataset)
- **Education** - customer's education level
    - 0 - other / unknown
    - 1 - High School
    - 2 - University
    - 3 - Graduate school
- **Income** - self reported annual income in US dollars
    - 35,832 - minimum value (lowest income observed in the dataset)
    - 309,364 - maximum value (highest income observed in the dataset)
- **Occupation** - customer's occupation
    - 0 - unemployed / unskilled
    - 1 - skilled employee / official
    - 2 - management / self-employed / highly qualified / officer
- **Settlement size** - the size of the city the customer lives in
    - 0 - small city
    - 1 - mid-sized city
    - 2 - big city

### Step 1: We perform a descriptive analysis of the data set by examining the data visually to gain some insight. 


```python
df_segmentation.head()
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
      <th>Sex</th>
      <th>Marital status</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
      <th>Occupation</th>
      <th>Settlement size</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100000001</th>
      <td>0</td>
      <td>0</td>
      <td>67</td>
      <td>2</td>
      <td>124670</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>100000002</th>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>1</td>
      <td>150773</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>100000003</th>
      <td>0</td>
      <td>0</td>
      <td>49</td>
      <td>1</td>
      <td>89210</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>100000004</th>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>1</td>
      <td>171565</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>100000005</th>
      <td>0</td>
      <td>0</td>
      <td>53</td>
      <td>1</td>
      <td>149031</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Step 2: We determine the statistical distribution of each variable based on their count, average (mean), standard deviation, minimum value, 25th, 50th and 75th percentile and their maximum value. Each feature in the dataset contains 2,000 data points as shown in the "count" row. 


```python
df_segmentation.describe()
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
      <th>Sex</th>
      <th>Marital status</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
      <th>Occupation</th>
      <th>Settlement size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.00000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.457000</td>
      <td>0.496500</td>
      <td>35.909000</td>
      <td>1.03800</td>
      <td>120954.419000</td>
      <td>0.810500</td>
      <td>0.739000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.498272</td>
      <td>0.500113</td>
      <td>11.719402</td>
      <td>0.59978</td>
      <td>38108.824679</td>
      <td>0.638587</td>
      <td>0.812533</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>18.000000</td>
      <td>0.00000</td>
      <td>35832.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.000000</td>
      <td>1.00000</td>
      <td>97663.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>33.000000</td>
      <td>1.00000</td>
      <td>115548.500000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>42.000000</td>
      <td>1.00000</td>
      <td>138072.250000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>76.000000</td>
      <td>3.00000</td>
      <td>309364.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Step 3: Determine whether features in the dataset are correlated to each other by computing the Correlation Estimate: 
- We compute the Pearson correlation coefficient for each feature in our data set. The correlation coefficient describes whether certain features are correlated (positively or negatively and the degree of their correlation - high or low) with other features in the dataset. 
- When two features are positively correlated, they tend to move in the same direction. So, if the numerical value of one feature increases or decreases, then the numerical value of the other feature has a tendency to change in the same way.
- By contrast, when two features are negatively (or 'inversely') correlated, they tend to move in the opposite direction. So, if the numerical value of one feature increases, then the numerical value of the other feature has a tendency to move in the opposite direction (i.e., to decrease). 


```python
df_segmentation.corr()
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
      <th>Sex</th>
      <th>Marital status</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
      <th>Occupation</th>
      <th>Settlement size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Sex</th>
      <td>1.000000</td>
      <td>0.566511</td>
      <td>-0.182885</td>
      <td>0.244838</td>
      <td>-0.195146</td>
      <td>-0.202491</td>
      <td>-0.300803</td>
    </tr>
    <tr>
      <th>Marital status</th>
      <td>0.566511</td>
      <td>1.000000</td>
      <td>-0.213178</td>
      <td>0.374017</td>
      <td>-0.073528</td>
      <td>-0.029490</td>
      <td>-0.097041</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>-0.182885</td>
      <td>-0.213178</td>
      <td>1.000000</td>
      <td>0.654605</td>
      <td>0.340610</td>
      <td>0.108388</td>
      <td>0.119751</td>
    </tr>
    <tr>
      <th>Education</th>
      <td>0.244838</td>
      <td>0.374017</td>
      <td>0.654605</td>
      <td>1.000000</td>
      <td>0.233459</td>
      <td>0.064524</td>
      <td>0.034732</td>
    </tr>
    <tr>
      <th>Income</th>
      <td>-0.195146</td>
      <td>-0.073528</td>
      <td>0.340610</td>
      <td>0.233459</td>
      <td>1.000000</td>
      <td>0.680357</td>
      <td>0.490881</td>
    </tr>
    <tr>
      <th>Occupation</th>
      <td>-0.202491</td>
      <td>-0.029490</td>
      <td>0.108388</td>
      <td>0.064524</td>
      <td>0.680357</td>
      <td>1.000000</td>
      <td>0.571795</td>
    </tr>
    <tr>
      <th>Settlement size</th>
      <td>-0.300803</td>
      <td>-0.097041</td>
      <td>0.119751</td>
      <td>0.034732</td>
      <td>0.490881</td>
      <td>0.571795</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Step 4: Plot the Correlation Heatmap: 
- We plot the correlations using a Heat Map. Heat Maps are a great way to visualize correlations using color coding. We use the "Red-Blue" (RdBu) color scheme with a range of 1.0 to -1.0. A dark blue 1.0 displays a perfectly positive 100% correlation between two features. A dark red -1.0 shows a perfectly negative 100% correlation between two features in the dataset.


```python
plt.figure(figsize = (7, 5))
s = sns.heatmap(df_segmentation.corr(),
               annot = True, 
               cmap = 'RdBu',
               vmin = -1, 
               vmax = 1)
s.set_yticklabels(s.get_yticklabels(), rotation = 0, fontsize = 11)
s.set_xticklabels(s.get_xticklabels(), rotation = 90, fontsize = 11)
plt.title('Correlation Heatmap')
plt.show()
```


    
![png](output_18_0.png)
    


### Results of our correlation heatmap shows: 
- a strong positive correlation between Age and Education (0.65). This could indicate that individuals who are older tend to have higher levels of education;
- a strong positive correlation between Income and Occupation (0.68). This may indicate that higher skilled individuals tend to generate higher levels of income;
- a strong positive correlation between Occupation and Settlement size (0.57). This may indicate that individuals in higher skilled occupations tend to reside in larger cities. 

### Step 5: Visualize the Raw Data 


```python
plt.figure(figsize = (8, 6))
plt.scatter(df_segmentation.iloc[:, 2], df_segmentation.iloc[:, 4])
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Figure 1: Visualization of raw data')
```




    Text(0.5, 1.0, 'Figure 1: Visualization of raw data')




    
![png](output_21_1.png)
    


### Figure 1: no pattern or trend can be observed in the data in Figure 1, just a lot of noise. 
- Figure 1 plots our data of 2,000 data points scattered across Age and Income. There is no observative pattern, just noise. 
- This is because the data has not yet been standardized. The Age data in years is not directly comparable to Income data which is in US dollar currency format. 
- The difference in the age range is 50 years while the difference in income range is 250,000. The model algorithm has only modeled the currency data and has completely ignored the age data since it is insignificant compared to the currency data. 
- To remove the bias in our data, all the data of the various features need to be standardized or scaled so that their values fall within the same numerical range and the differences between their values will be comparable to each other. 

## Data Standardization 
- We use Python's standard scaler which we import from the Sklearn module in order to pre-process and standardize our data. 


```python
scaler = StandardScaler()
segmentation_std = scaler.fit_transform(df_segmentation)
```

## Model No. 1: Hierarchical Clustering 
- We begin our customer segmentation analysis by applying our first segmentation algorithm to our dataset - the hierarchical clustering model. 
- The hierarchical clustering model groups similar objects in the dataset into groups called clusters. The endpoint is a set of clusters, where each cluster is distinct from one another, but the objects within each cluster are broadly similar to each other. 


```python
hier_clust = linkage(segmentation_std, method = 'ward')
```


```python
plt.figure(figsize = (7, 5))
plt.title('Figure 2: Hierarchical Clustering Dendrogram')
plt.xlabel('Observations')
plt.ylabel('Distance')
dendrogram(hier_clust,
           truncate_mode = 'level',
           p = 4,
           show_leaf_counts = False,
           no_labels = True)
plt.show()
```


    
![png](output_27_0.png)
    


### Figure 2: 
- Our hierarchical clustering dendrogram is comprised of 4 levels. The bottom level, level 4, shows our dataset as having around four different clusters or groups. Each group has features that are different from each other. 

## Model No. 2: K-means Clustering 
- The goal of K-means clustering is similar to hierarchical clustering - to group similar data points together and discover underlying patterns. To achieve this objective, K-means looks for a fixed number (k) of clusters in a dataset. 
- The K-means algorithm identifies *k* number of centroids within the dataset and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible. 
- When applying K-means clustering, we assume our dataset has 1 to 10 clusters and we perform 10 iterations.
- We run the algorithm at many different starting points - *k means plus plus*. 
- We also establish a random state for reproducibility. 


```python
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(segmentation_std)
    wcss.append(kmeans.inertia_)
```

### Figure 3: The WCSS curve of K-means Clustering 
- We plot the *Within Cluster Sum of Squares (WCSS)* curve for the ten different number of clusters in our dataset. 
- We look for an *elbow* or a *kink* in the curve which identifies how many groups or clusters there are in the dataset. 


```python
plt.figure(figsize = (7, 5))
plt.plot(range(1, 11), wcss, marker = 'o', linestyle = '--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Figure 3: K-means Clustering')
plt.show()
```


    
![png](output_32_0.png)
    


### Figure 3: 
- We have identified that the elbow in the WCSS curve occurs when the number of clusters in our dataset is equal to 4. 
- We now run the K-means model where the number of clusters is equal to 4. 


```python
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
```

### We now segment the dataset into four clusters and then fit our K-means model into the dataset. 


```python
kmeans.fit(segmentation_std)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KMeans(n_clusters=4, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">KMeans</label><div class="sk-toggleable__content"><pre>KMeans(n_clusters=4, random_state=42)</pre></div></div></div></div></div>



## Analyzing the new Dataset with 4 Different Customer Segments 

### We create a new data frame with the original features and add a new column with the assigned clusters for each point. 
- We also calculate mean values for each of the 4 clusters. 
- We now need to label each group and analyze each customer segment thoroughly. 


```python
df_segm_kmeans = df_segmentation.copy()
df_segm_kmeans['Segment K-means'] = kmeans.labels_
```


```python
df_segm_analysis = df_segm_kmeans.groupby(['Segment K-means']).mean()
df_segm_analysis
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
      <th>Sex</th>
      <th>Marital status</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
      <th>Occupation</th>
      <th>Settlement size</th>
    </tr>
    <tr>
      <th>Segment K-means</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.501901</td>
      <td>0.692015</td>
      <td>55.703422</td>
      <td>2.129278</td>
      <td>158338.422053</td>
      <td>1.129278</td>
      <td>1.110266</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.352814</td>
      <td>0.019481</td>
      <td>35.577922</td>
      <td>0.746753</td>
      <td>97859.852814</td>
      <td>0.329004</td>
      <td>0.043290</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.853901</td>
      <td>0.997163</td>
      <td>28.963121</td>
      <td>1.068085</td>
      <td>105759.119149</td>
      <td>0.634043</td>
      <td>0.422695</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.029825</td>
      <td>0.173684</td>
      <td>35.635088</td>
      <td>0.733333</td>
      <td>141218.249123</td>
      <td>1.271930</td>
      <td>1.522807</td>
    </tr>
  </tbody>
</table>
</div>



## Distinguishing features of the 4 K-means cluster segments: (see Data Legend for feature descriptions) 
**Segment 0**: 
- Group is roughly equal between male and female (0.50), mostly married/divorced/separated (0.69), tend to be highly educated (University level), high average income - 158,338; tend to be employed in skilled to highly skilled jobs and they tend to reside in mid-sized to big cities. 

**Segment 1**:
- Group is about 1/3 male (0.35), mostly single, average age of around 35.6 years, average education is high school or below, members have an average income of 97,859; most are employed in low-skilled jobs or unemployed and tend to reside in small cities.

**Segment 2**:
- Most members are female (0.85) and tend to be married (1.0), the average age is around 29 years, the average education is a high school education level with an average income of around 105,759; most are employed in skilled or official positions and they tend to reside in small to mid-sized cities.

**Segment 3**:
- Most members are male (0.03) and tend to be single; the average age is 35.6 years and they tend to have a high school education level; the average income is 141,218; the average job tends to be skilled/self-employed/highly skilled and they tend to reside in mid-sized to big cities. 


```python
df_segm_analysis['N Obs'] = df_segm_kmeans[['Segment K-means','Sex']].groupby(['Segment K-means']).count()
df_segm_analysis['Prop Obs'] = df_segm_analysis['N Obs'] / df_segm_analysis['N Obs'].sum()
```

### We compute the size and proportions of the four clusters: 


```python
df_segm_analysis
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
      <th>Sex</th>
      <th>Marital status</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
      <th>Occupation</th>
      <th>Settlement size</th>
      <th>N Obs</th>
      <th>Prop Obs</th>
    </tr>
    <tr>
      <th>Segment K-means</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.501901</td>
      <td>0.692015</td>
      <td>55.703422</td>
      <td>2.129278</td>
      <td>158338.422053</td>
      <td>1.129278</td>
      <td>1.110266</td>
      <td>263</td>
      <td>0.1315</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.352814</td>
      <td>0.019481</td>
      <td>35.577922</td>
      <td>0.746753</td>
      <td>97859.852814</td>
      <td>0.329004</td>
      <td>0.043290</td>
      <td>462</td>
      <td>0.2310</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.853901</td>
      <td>0.997163</td>
      <td>28.963121</td>
      <td>1.068085</td>
      <td>105759.119149</td>
      <td>0.634043</td>
      <td>0.422695</td>
      <td>705</td>
      <td>0.3525</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.029825</td>
      <td>0.173684</td>
      <td>35.635088</td>
      <td>0.733333</td>
      <td>141218.249123</td>
      <td>1.271930</td>
      <td>1.522807</td>
      <td>570</td>
      <td>0.2850</td>
    </tr>
  </tbody>
</table>
</div>



### Segment Results: 
- **Segment 0**: has 263 observations comprising about 13.2% of the total dataset.
- **Segment 1**: has 462 observations making up about 23.1% of the total dataset.
- **Segment 2**: has 705 observations for around 35.2% of the total dataset.
- **Segment 3**: has about 570 observations for about 28.5% of the total dataset. 

### We now assign a descriptive label to each segment cluster based on their distinguishing characteristics: 
- **Segment 0** - **Well-off** 
- **Segment 1** - **Fewer Opportunities** 
- **Segment 2** - **Standard** 
- **Segment 3** - **Career-Focused** 


```python
df_segm_analysis.rename({0:'well-off',
                         1:'fewer-opportunities',
                         2:'standard',
                         3:'career focused'})
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
      <th>Sex</th>
      <th>Marital status</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
      <th>Occupation</th>
      <th>Settlement size</th>
      <th>N Obs</th>
      <th>Prop Obs</th>
    </tr>
    <tr>
      <th>Segment K-means</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>well-off</th>
      <td>0.501901</td>
      <td>0.692015</td>
      <td>55.703422</td>
      <td>2.129278</td>
      <td>158338.422053</td>
      <td>1.129278</td>
      <td>1.110266</td>
      <td>263</td>
      <td>0.1315</td>
    </tr>
    <tr>
      <th>fewer-opportunities</th>
      <td>0.352814</td>
      <td>0.019481</td>
      <td>35.577922</td>
      <td>0.746753</td>
      <td>97859.852814</td>
      <td>0.329004</td>
      <td>0.043290</td>
      <td>462</td>
      <td>0.2310</td>
    </tr>
    <tr>
      <th>standard</th>
      <td>0.853901</td>
      <td>0.997163</td>
      <td>28.963121</td>
      <td>1.068085</td>
      <td>105759.119149</td>
      <td>0.634043</td>
      <td>0.422695</td>
      <td>705</td>
      <td>0.3525</td>
    </tr>
    <tr>
      <th>career focused</th>
      <td>0.029825</td>
      <td>0.173684</td>
      <td>35.635088</td>
      <td>0.733333</td>
      <td>141218.249123</td>
      <td>1.271930</td>
      <td>1.522807</td>
      <td>570</td>
      <td>0.2850</td>
    </tr>
  </tbody>
</table>
</div>



### We add the segment labels to our table 


```python
df_segm_kmeans['Labels'] = df_segm_kmeans['Segment K-means'].map({0:'well-off', 
                                                                  1:'fewer opportunities',
                                                                  2:'standard', 
                                                                  3:'career focused'})
```

### We plot the results from the K-means algorithm. 
- Each point in the data set has been plotted with the color of the clusters that it was assigned to. 


```python
x_axis = df_segm_kmeans['Age']
y_axis = df_segm_kmeans['Income']
plt.figure(figsize = (7, 6))
sns.scatterplot(x = x_axis, y = y_axis, hue = df_segm_kmeans['Labels'], palette = ['g', 'r', 'c', 'm'])
plt.title('Figure 4: Segmentation K-means')
plt.show()
```


    
![png](output_51_0.png)
    


## Figure 4: K-means did a good job but we can get even better results by combining K-means with another classification algorithm - Principal Component Analysis (PCA) 
- Based on the K-means cluster graph, we can say that the K-means algorithm did a satisfactory job in separating the 2,000 points in the data set into 4 clusters. 
- We observe the green segment “well-off” is clearly separated as it is highest in both age and income. 
- However, the other three clusters are grouped together, so it's difficult to get more insight on the other 3 clusters by just looking at the graph. 
- To separate the other 3 clusters, we will combine K-means with principal component analysis to try to get a better result. 

## Model No. 3: Principal Component Analysis - PCA 
### What is Principal Component Analysis (PCA)? 

>Principal component analysis, or PCA, is a statistical procedure that allows you to summarize the information content in large data tables by means of a smaller set of “summary indices” (or principal components) that can be more easily visualized and analyzed. 
>
>This methodology is also known as “dimensionality reduction” where large numbers of features can be reduced into a handful of variables that can explain the bulk of the variance (or the movements) of the data points in a large dataset. 

### Dimensionality Reduction 

>One of the reasons why it’s difficult to observe a pattern in Figure 4 is because the data points are clustered so close to each other – making it hard to see the clear lines of differentiation between each cluster. 
>
>This is because there are seven features (sex, age, marital status, education, etc.) that are plotted in a 2-dimensional (2D: x axis, y axis) graph. Ideally, to show each cluster segmentation clearly, all the data points need to be plotted in a 7-dimensional graph (7D), which is not possible. 
>
>By employing dimensionality reduction (through linear algebra), we can reduce the number of features to around 3 to 4 essential features which can explain around 80% to 90% of the variability in the data set. This makes it easier to see how each cluster segment is different from each other. 

## Applying PCA to the dataset 


```python
pca = PCA()
```


```python
pca.fit(segmentation_std)
```




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>PCA()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">PCA</label><div class="sk-toggleable__content"><pre>PCA()</pre></div></div></div></div></div>



### The variance ratio shows how much variance is explained by each of the seven individual components. 


```python
pca.explained_variance_ratio_
```




    array([0.35696328, 0.26250923, 0.18821114, 0.0755775 , 0.05716512,
           0.03954794, 0.02002579])



## What is the Explanatory Power of each Principal Component? 
- Component 1: explains about 35.7% of the variation in the data set.
- Component 2: explains around 26.3% of the movement in the data set.
- Component 3: explains about 18.8% of the variance in the data set.
- Component 4: explains around 8% of the variation in the data set.
- **All told, the 4 components explain around 90% of the movement in the data set.** 

## Plot the cumulative variance explained by total number of components. 
### On the graph we choose the subset of components we want to keep. We select 3 components because it explains around 80% of the variance in the data set. 


```python
plt.figure(figsize = (7, 5))
plt.plot(range(1,8), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
plt.title('Figure 5: Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
```




    Text(0, 0.5, 'Cumulative Explained Variance')




    
![png](output_64_1.png)
    


### The PCA model with the 3 components is fitted to the data set. 


```python
pca = PCA(n_components = 3)
```


```python
pca.fit(segmentation_std)
```




<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>PCA(n_components=3)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">PCA</label><div class="sk-toggleable__content"><pre>PCA(n_components=3)</pre></div></div></div></div></div>



## PCA Results 


```python
pca.components_
```




    array([[-0.31469524, -0.19170439,  0.32609979,  0.15684089,  0.52452463,
             0.49205868,  0.46478852],
           [ 0.45800608,  0.51263492,  0.31220793,  0.63980683,  0.12468314,
             0.01465779, -0.06963165],
           [-0.29301261, -0.44197739,  0.60954372,  0.27560461, -0.16566231,
            -0.39550539, -0.29568503]])



### Components 1 - 3 generate a set of correlations (aka "vector loadings") that show the degree of influence that each feature has on it. The closer the correlation approaches 1.0, the stronger its influence. 


```python
df_pca_comp = pd.DataFrame(data = pca.components_,
                           columns = df_segmentation.columns.values,
                           index = ['Component 1', 'Component 2', 'Component 3'])
df_pca_comp
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
      <th>Sex</th>
      <th>Marital status</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
      <th>Occupation</th>
      <th>Settlement size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Component 1</th>
      <td>-0.314695</td>
      <td>-0.191704</td>
      <td>0.326100</td>
      <td>0.156841</td>
      <td>0.524525</td>
      <td>0.492059</td>
      <td>0.464789</td>
    </tr>
    <tr>
      <th>Component 2</th>
      <td>0.458006</td>
      <td>0.512635</td>
      <td>0.312208</td>
      <td>0.639807</td>
      <td>0.124683</td>
      <td>0.014658</td>
      <td>-0.069632</td>
    </tr>
    <tr>
      <th>Component 3</th>
      <td>-0.293013</td>
      <td>-0.441977</td>
      <td>0.609544</td>
      <td>0.275605</td>
      <td>-0.165662</td>
      <td>-0.395505</td>
      <td>-0.295685</td>
    </tr>
  </tbody>
</table>
</div>



## PCA Heat Map for the 3 components 


```python
sns.heatmap(df_pca_comp,
            vmin = -1, 
            vmax = 1,
            cmap = 'RdBu',
            annot = True)
plt.yticks([0, 1, 2], 
           ['Component 1', 'Component 2', 'Component 3'],
           rotation = 45,
           fontsize = 8)
```




    ([<matplotlib.axis.YTick at 0x27bc8377510>,
      <matplotlib.axis.YTick at 0x27bc5db3710>,
      <matplotlib.axis.YTick at 0x27bc83e2490>],
     [Text(0, 0, 'Component 1'),
      Text(0, 1, 'Component 2'),
      Text(0, 2, 'Component 3')])




    
![png](output_73_1.png)
    


### Component 1: Career-Focus
- There is a positive correlation between Component 1 and the features age, income, occupation, and settlement size. The features describe the career focus of the individual. We assign the label of "career-focus" to component 1.

### Component 2: Education and Lifestyle 
- The loadings of Component 2 seems to emphasize the individual's sex, marital status, and education which shows the person's education and lifestyle as the most prominent determinants. We therefore label component 2 as "Education and Lifestyle". 

### Component 3: Work and Life Experience 
- Component 3 shows age, marital status, and occupation as the most important determinants which relate to the person's work experience or life experience. We therefore assign the label of "Work and Life Experience" to Component 3. 

## PCA Scores: 
- We apply Components 1, 2 and 3 using the "PCA Transform" method to our previous segmented data set with 7 features reducing it to just 3 features ("dimensionality reduction"). 
- This generates a new data set of PCA scores for each of the 2,000 data points shown below. 


```python
pca.transform(segmentation_std)
```




    array([[ 2.51474593,  0.83412239,  2.1748059 ],
           [ 0.34493528,  0.59814564, -2.21160279],
           [-0.65106267, -0.68009318,  2.2804186 ],
           ...,
           [-1.45229829, -2.23593665,  0.89657125],
           [-2.24145254,  0.62710847, -0.53045631],
           [-1.86688505, -2.45467234,  0.66262172]])



### Perform K-means clustering on the newly transformed 3D data set "scores_pca". 
- Save the new data set to the variable “scores PCA”. 
- Proceed to the next step of the analysis by performing K-means clustering again on our new data set of PCA scores.  


```python
scores_pca = pca.transform(segmentation_std)
```

## Model No. 4: K-means clustering with PCA 
- We segment our newly transformed data set of PCA scores using the K-means algorithm and we apply clustering using principal components as our features. 
- We fit K-means using the transformed data from the PCA. 
- We plot the Within Cluster Sum of Squares (wcss) for the K-means PCA model. 


```python
wcss = []
for i in range(1,11):
    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)
```

### Based on the "elbow" in the curve in Figure 6, we choose 4 cluster segments for our data set. 


```python
plt.figure(figsize = (7, 5))
plt.plot(range(1, 11), wcss, marker = 'o', linestyle = '--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Figure 6: K-means with PCA Clustering')
plt.show()
```


    
![png](output_84_0.png)
    


### We run K-means with 4 cluster segments. 


```python
kmeans_pca = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
```

### We fit our data with the K-means PCA model. 


```python
kmeans_pca.fit(scores_pca)
```




<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KMeans(n_clusters=4, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">KMeans</label><div class="sk-toggleable__content"><pre>KMeans(n_clusters=4, random_state=42)</pre></div></div></div></div></div>



### K-means clustering with PCA Results 
- **The K-means algorithm has assigned each data point to one of the 4 cluster segments** based on the assigned PCA scores from Components 1, 2 and 3 that the individual has. 


```python
df_segm_pca_kmeans = pd.concat([df_segmentation.reset_index(drop = True), pd.DataFrame(scores_pca)], axis = 1)
df_segm_pca_kmeans.columns.values[-3: ] = ['Component 1', 'Component 2', 'Component 3']
df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_
```


```python
df_segm_pca_kmeans
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
      <th>Sex</th>
      <th>Marital status</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
      <th>Occupation</th>
      <th>Settlement size</th>
      <th>Component 1</th>
      <th>Component 2</th>
      <th>Component 3</th>
      <th>Segment K-means PCA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>67</td>
      <td>2</td>
      <td>124670</td>
      <td>1</td>
      <td>2</td>
      <td>2.514746</td>
      <td>0.834122</td>
      <td>2.174806</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>1</td>
      <td>150773</td>
      <td>1</td>
      <td>2</td>
      <td>0.344935</td>
      <td>0.598146</td>
      <td>-2.211603</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>49</td>
      <td>1</td>
      <td>89210</td>
      <td>0</td>
      <td>0</td>
      <td>-0.651063</td>
      <td>-0.680093</td>
      <td>2.280419</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>1</td>
      <td>171565</td>
      <td>1</td>
      <td>1</td>
      <td>1.714316</td>
      <td>-0.579927</td>
      <td>0.730731</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>53</td>
      <td>1</td>
      <td>149031</td>
      <td>1</td>
      <td>1</td>
      <td>1.626745</td>
      <td>-0.440496</td>
      <td>1.244909</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>1</td>
      <td>0</td>
      <td>47</td>
      <td>1</td>
      <td>123525</td>
      <td>0</td>
      <td>0</td>
      <td>-0.866034</td>
      <td>0.298330</td>
      <td>1.438958</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>1</td>
      <td>1</td>
      <td>27</td>
      <td>1</td>
      <td>117744</td>
      <td>1</td>
      <td>0</td>
      <td>-1.114957</td>
      <td>0.794727</td>
      <td>-1.079871</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>0</td>
      <td>0</td>
      <td>31</td>
      <td>0</td>
      <td>86400</td>
      <td>0</td>
      <td>0</td>
      <td>-1.452298</td>
      <td>-2.235937</td>
      <td>0.896571</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>1</td>
      <td>1</td>
      <td>24</td>
      <td>1</td>
      <td>97968</td>
      <td>0</td>
      <td>0</td>
      <td>-2.241453</td>
      <td>0.627108</td>
      <td>-0.530456</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>0</td>
      <td>68416</td>
      <td>0</td>
      <td>0</td>
      <td>-1.866885</td>
      <td>-2.454672</td>
      <td>0.662622</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 11 columns</p>
</div>



### Calculate the means of each cluster segment group 


```python
df_segm_pca_kmeans_freq = df_segm_pca_kmeans.groupby(['Segment K-means PCA']).mean()
df_segm_pca_kmeans_freq
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
      <th>Sex</th>
      <th>Marital status</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
      <th>Occupation</th>
      <th>Settlement size</th>
      <th>Component 1</th>
      <th>Component 2</th>
      <th>Component 3</th>
    </tr>
    <tr>
      <th>Segment K-means PCA</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.900289</td>
      <td>0.965318</td>
      <td>28.878613</td>
      <td>1.060694</td>
      <td>107551.500000</td>
      <td>0.677746</td>
      <td>0.440751</td>
      <td>-1.107019</td>
      <td>0.703776</td>
      <td>-0.781410</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.027444</td>
      <td>0.168096</td>
      <td>35.737564</td>
      <td>0.734134</td>
      <td>141525.826758</td>
      <td>1.267581</td>
      <td>1.480274</td>
      <td>1.372663</td>
      <td>-1.046172</td>
      <td>-0.248046</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.306522</td>
      <td>0.095652</td>
      <td>35.313043</td>
      <td>0.760870</td>
      <td>93692.567391</td>
      <td>0.252174</td>
      <td>0.039130</td>
      <td>-1.046406</td>
      <td>-0.902963</td>
      <td>1.003644</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.505660</td>
      <td>0.690566</td>
      <td>55.679245</td>
      <td>2.128302</td>
      <td>158019.101887</td>
      <td>1.120755</td>
      <td>1.101887</td>
      <td>1.687328</td>
      <td>2.031200</td>
      <td>0.844039</td>
    </tr>
  </tbody>
</table>
</div>



### Calculate the size of each cluster and its proportion to the entire data set and assign the same descriptive labels from the previous cluster segments. 
- **Component 1: Career-Focus**
- **Component 2: Education and Lifestyle**
- **Component 3: Work and Life Experience** 


```python
df_segm_pca_kmeans_freq['N Obs'] = df_segm_pca_kmeans[['Segment K-means PCA','Sex']].groupby(['Segment K-means PCA']).count()
df_segm_pca_kmeans_freq['Prop Obs'] = df_segm_pca_kmeans_freq['N Obs'] / df_segm_pca_kmeans_freq['N Obs'].sum()
df_segm_pca_kmeans_freq = df_segm_pca_kmeans_freq.rename({0:'standard', 
                                                          1:'career focused',
                                                          2:'fewer opportunities', 
                                                          3:'well-off'})
df_segm_pca_kmeans_freq
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
      <th>Sex</th>
      <th>Marital status</th>
      <th>Age</th>
      <th>Education</th>
      <th>Income</th>
      <th>Occupation</th>
      <th>Settlement size</th>
      <th>Component 1</th>
      <th>Component 2</th>
      <th>Component 3</th>
      <th>N Obs</th>
      <th>Prop Obs</th>
    </tr>
    <tr>
      <th>Segment K-means PCA</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>standard</th>
      <td>0.900289</td>
      <td>0.965318</td>
      <td>28.878613</td>
      <td>1.060694</td>
      <td>107551.500000</td>
      <td>0.677746</td>
      <td>0.440751</td>
      <td>-1.107019</td>
      <td>0.703776</td>
      <td>-0.781410</td>
      <td>692</td>
      <td>0.3460</td>
    </tr>
    <tr>
      <th>career focused</th>
      <td>0.027444</td>
      <td>0.168096</td>
      <td>35.737564</td>
      <td>0.734134</td>
      <td>141525.826758</td>
      <td>1.267581</td>
      <td>1.480274</td>
      <td>1.372663</td>
      <td>-1.046172</td>
      <td>-0.248046</td>
      <td>583</td>
      <td>0.2915</td>
    </tr>
    <tr>
      <th>fewer opportunities</th>
      <td>0.306522</td>
      <td>0.095652</td>
      <td>35.313043</td>
      <td>0.760870</td>
      <td>93692.567391</td>
      <td>0.252174</td>
      <td>0.039130</td>
      <td>-1.046406</td>
      <td>-0.902963</td>
      <td>1.003644</td>
      <td>460</td>
      <td>0.2300</td>
    </tr>
    <tr>
      <th>well-off</th>
      <td>0.505660</td>
      <td>0.690566</td>
      <td>55.679245</td>
      <td>2.128302</td>
      <td>158019.101887</td>
      <td>1.120755</td>
      <td>1.101887</td>
      <td>1.687328</td>
      <td>2.031200</td>
      <td>0.844039</td>
      <td>265</td>
      <td>0.1325</td>
    </tr>
  </tbody>
</table>
</div>



## Model Results and Analysis: 

### Defining Group Characteristics: 
- **Standard** segment is low on both component 1 (career focus) and component 3 (work and life experience). However, it has high values on component 2 (education and lifestyle). 
- **Career focused** segment shows high values for component 1 (career) but low for component 2 (education and lifestyle) and component 3 (work and life experience). 
- **Fewer Opportunities** is low on both component 1 (career focus) and component 2 (education and lifestyle) but high on component 3 (work and life experience). 
- **Well-off** has the highest values on both component 1 (career focus) and component 2 (education and lifestyle) but low on component 3 (work and life experience). 

### Key Findings and Insights of our Customer Segmentation Study: 
- The largest segment with 692 individuals is the **standard segment**. It represents roughly around 35% of the total data set. Members belonging to this group are mostly married females with an average age of around 29 years. They're mostly high school graduates employed in skilled or official positions (e.g., office assistants) and have an average income of around 106,000. They mostly reside in small to mid-sized cities. **Individuals in this group tend to put a high value on lifestyle. They are not particularly career conscious nor are they interested in their work or life experience.** 

- The second largest group in the data set is the **career focus cluster** group with 583 people or 29% of the whole data set. 
Most members are single males with an average age of 35.6 years with a high school education level; their average income is around 141,218 and they tend to be self-employed or in skilled to highly skilled jobs. They mostly reside in mid-sized to big cities. **Career focused individuals tend to put a high emphasis on their career goals and are not so interested in lifestyle or their life/work experiences.** 

- The 3rd largest group is the **fewer opportunities cluster** with 460 people making up about 23.0% of the total dataset. 
Group members are about 70% male, mostly single with an average age of around 35.3 years with a high school education level or below. The group average income is around 93,692 and individuals are mostly employed in low-skilled jobs or are unemployed. The average member tends to reside in small cities. **Individuals in this group put a high emphasis on their work and life experience and put a low emphasis on career goals and education and lifestyle values.** 

- The smallest segment is the **well-off cluster** group with about 265 members making up around 13% of the total data set. This group is divided equally between male and female, and members tend to be married, divorced or separated. They tend to be highly educated (University level) with a high average income of around 158,338. Most are employed in skilled to highly skilled jobs and they tend to reside in mid-sized to big cities. **Individuals in this group are career focused and put a high emphasis on education and lifestyle. However, they put low values on work and life experience.** 


```python
df_segm_pca_kmeans['Legend'] = df_segm_pca_kmeans['Segment K-means PCA'].map({0:'standard', 
                                                          1:'career focused',
                                                          2:'fewer opportunities', 
                                                          3:'well-off'})
```

## Cluster Segment Visualization on a 2D plane. 

### The point of PCA was to determine the most important components so we can be certain that the first two components explain more variance than the third one. 

### Plot data by PCA components. Component 1 (Career-Focus: 36% explanatory power) is on the y-axis, Component 2 (Education / Lifestyle: 26% explanatory power) is on the x-axis. Total variation explained: 62% 


```python
x_axis = df_segm_pca_kmeans['Component 2']
y_axis = df_segm_pca_kmeans['Component 1']
plt.figure(figsize = (8, 6))
sns.scatterplot(x = x_axis, y = y_axis, hue = df_segm_pca_kmeans['Legend'], palette = ['g', 'r', 'c', 'm'])
plt.title('Figure 7: Clusters by PCA Components')
plt.show()
```


    
![png](output_100_0.png)
    


## Figure 7 analysis: 

### - The well-off group is both high on career and education and lifestyle values. The group is clustered mostly on the upper right quadrant of the graph. 

### - In the opposite quadrant on the lower left of the graph is the fewer opportunities group which puts low values on both components. 

### - The standard group is mostly in the middle of the graph, which shows average values and indicates that this group is modestly interested in career goals and educational / lifestyle interests. Interesting to note is the overlap between the standard and fewer opportunities group. 

### - Lastly, the career focus group shows a strong interest in career goals and not so much interest in educational and lifestyle experiences. 

### Plot data by PCA components. The y-axis is Component 1 (Career-Focus: 36% explanatory power), the x-axis is Component 3 (Work and Life experience: 19% explanatory power). Total variation explained: 55% 


```python
x_axis_1 = df_segm_pca_kmeans['Component 3']
y_axis_1 = df_segm_pca_kmeans['Component 1']
plt.figure(figsize = (8, 6))
sns.scatterplot(x = x_axis_1, y = y_axis_1, hue = df_segm_pca_kmeans['Legend'], palette = ['g', 'r', 'c', 'm'])
plt.title('Figure 8: Clusters by PCA Components')
plt.show()
```


    
![png](output_103_0.png)
    


### Plot data by PCA components. The y-axis is Component 2 (Education and Lifestyle: 26% explanatory power), x-axis is Component 3 (Work and Life experience: 19% explanatory power). Total variation explained: 45%. 


```python
x_axis_1 = df_segm_pca_kmeans['Component 3']
y_axis_1 = df_segm_pca_kmeans['Component 2']
plt.figure(figsize = (8, 6))
sns.scatterplot(x = x_axis_1, y = y_axis_1, hue = df_segm_pca_kmeans['Legend'], palette = ['g', 'r', 'c', 'm'])
plt.title('Figure 9: Clusters by PCA Components')
plt.show()
```


    
![png](output_105_0.png)
    

