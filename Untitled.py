#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install transformers


# In[1]:


pip install -U sentence-transformers


# In[3]:


import pandas as pd
import numpy as np

file_name = r"testing (2).xlsx"
dtest = pd.read_excel(file_name)

name = r"training (2).xlsx"
dtrain = pd.read_excel(name)


# In[4]:


from sentence_transformers import SentenceTransformer
model=SentenceTransformer('sentence-transformers/sentence-t5-base')
dtest['EmbeddingsLM']=dtest['Equation'].apply(lambda x:model.encode(x))
t5_test=pd.DataFrame(dtest['EmbeddingsLM'].tolist(),index=dtest.index).add_prefix('embed_')


# In[6]:


from sentence_transformers import SentenceTransformer
model=SentenceTransformer('sentence-transformers/sentence-t5-base')
dtrain['EmbeddingsLM']=dtrain['input'].apply(lambda x:model.encode(str(x)))
t5_train=pd.DataFrame(dtrain['EmbeddingsLM'].tolist(),index=dtrain.index).add_prefix('embed_')


# In[7]:


from sklearn.metrics import pairwise_distances
import numpy as np

class_0_embeddings = t5_train[dtrain['output'] == 0]  
class_1_embeddings = t5_train[dtrain['output'] == 1]  

class_0_centroid = np.mean(class_0_embeddings, axis=0)
class_1_centroid = np.mean(class_1_embeddings, axis=0)

intra_class_spread_0 = np.mean(pairwise_distances(class_0_embeddings, [class_0_centroid]))
intra_class_spread_1 = np.mean(pairwise_distances(class_1_embeddings, [class_1_centroid]))

inter_class_distance = np.linalg.norm(class_0_centroid - class_1_centroid)

print(f"Intra-Class Spread for Class 0: {intra_class_spread_0}")
print(f"Intra-Class Spread for Class 1: {intra_class_spread_1}")
print(f"Inter-Class Distance: {inter_class_distance}")


# In[8]:


class_0_centroid = np.mean(class_0_embeddings, axis=0)
class_1_centroid = np.mean(class_1_embeddings, axis=0)

print("Centroid for Class 0:")
print(class_0_centroid)

print("Centroid for Class 1:")
print(class_1_centroid)


# In[9]:


std_dev_class_0 = np.std(class_0_embeddings, axis=0)
std_dev_class_1 = np.std(class_1_embeddings, axis=0)

print("Standard Deviation for Class 0:")
print(std_dev_class_0)

print("Standard Deviation for Class 1:")
print(std_dev_class_1)


# In[10]:


class_0_centroid = np.mean(class_0_embeddings, axis=0)
class_1_centroid = np.mean(class_1_embeddings, axis=0)

distance_between_classes = np.linalg.norm(class_0_centroid - class_1_centroid)

print("Distance between Class 0 and Class 1 mean vectors:")
print(distance_between_classes)


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt

file_name = r"testing (2).xlsx"
df = pd.read_excel(file_name)

selected_feature = 'output'

num_bins = 20 
plt.hist(df[selected_feature], bins=num_bins, edgecolor='k')
plt.xlabel(selected_feature)
plt.ylabel('Frequency')
plt.title(f'Histogram of {selected_feature}')
plt.show()

# Calculate the mean and variance
feature_mean = df[selected_feature].mean()
feature_variance = df[selected_feature].var()

print(f"Mean of {selected_feature}: {feature_mean}")
print(f"Variance of {selected_feature}: {feature_variance}")


# In[12]:


import pandas as pd
import matplotlib.pyplot as plt

file_name = r"training (2).xlsx"
df = pd.read_excel(name)

selected_feature = 'output'

num_bins = 20 
plt.hist(df[selected_feature], bins=num_bins, edgecolor='k')
plt.xlabel(selected_feature)
plt.ylabel('Frequency')
plt.title(f'Histogram of {selected_feature}')
plt.show()

feature_mean = df[selected_feature].mean()
feature_variance = df[selected_feature].var()

print(f"Mean of {selected_feature}: {feature_mean}")
print(f"Variance of {selected_feature}: {feature_variance}")


# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_name = r"testing (2).xlsx"
df1 = pd.read_excel(file_name)

file_name = r"training (2).xlsx"
df2 = pd.read_excel(name)

feature1 = df1['output'].values
feature2 = df2['output'].values

r_values = range(1, 11)

distances1 = []
distances2 = []

for r in r_values:
    minkowski_distances1 = []
    minkowski_distances2 = []
    
    for i in range(len(feature1)):
        for j in range(len(feature2)):
            distance = np.abs(feature1[i] - feature2[j])**r
            minkowski_distances1.append(distance**(1/r))
    
    mean_distance1 = np.mean(minkowski_distances1)
    mean_distance2 = np.mean(minkowski_distances2)
    
    distances1.append(mean_distance1)
    distances2.append(mean_distance2)

plt.plot(r_values, distances1, label='Dataset 1', marker='o', linestyle='-')
plt.plot(r_values, distances2, label='Dataset 2', marker='x', linestyle='-')
plt.xlabel('r')
plt.ylabel('Mean Minkowski Distance')
plt.title('Mean Minkowski Distance vs. r')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




