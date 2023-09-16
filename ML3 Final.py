#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install transformers


# In[2]:


pip install -U sentence-transformers


# In[16]:


import pandas as pd
import numpy as np

file_name = r"testing (2) (1).xlsx"
dtest = pd.read_excel(file_name)

name = r"training (2) (1).xlsx"
dtrain = pd.read_excel(name)


# In[17]:


from sentence_transformers import SentenceTransformer
model=SentenceTransformer('sentence-transformers/sentence-t5-base')
dtest['EmbeddingsLM']=dtest['Equation'].apply(lambda x:model.encode(x))
t5_test=pd.DataFrame(dtest['EmbeddingsLM'].tolist(),index=dtest.index).add_prefix('embed_')


# In[18]:


from sentence_transformers import SentenceTransformer
model=SentenceTransformer('sentence-transformers/sentence-t5-base')
dtrain['EmbeddingsLM']=dtrain['input'].apply(lambda x:model.encode(str(x)))
t5_train=pd.DataFrame(dtrain['EmbeddingsLM'].tolist(),index=dtrain.index).add_prefix('embed_')


# In[19]:


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


# In[20]:


class_0_centroid = np.mean(class_0_embeddings, axis=0)
class_1_centroid = np.mean(class_1_embeddings, axis=0)

print("Centroid for Class 0:")
print(class_0_centroid)

print("Centroid for Class 1:")
print(class_1_centroid)


# In[21]:


std_dev_class_0 = np.std(class_0_embeddings, axis=0)
std_dev_class_1 = np.std(class_1_embeddings, axis=0)

print("Standard Deviation for Class 0:")
print(std_dev_class_0)

print("Standard Deviation for Class 1:")
print(std_dev_class_1)


# In[22]:


class_0_centroid = np.mean(class_0_embeddings, axis=0)
class_1_centroid = np.mean(class_1_embeddings, axis=0)

distance_between_classes = np.linalg.norm(class_0_centroid - class_1_centroid)

print("Distance between Class 0 and Class 1 mean vectors:")
print(distance_between_classes)


# In[23]:


import pandas as pd
import matplotlib.pyplot as plt

file_name = r"testing (2) (1).xlsx"
df = pd.read_excel(file_name)

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


# In[24]:


import pandas as pd
import matplotlib.pyplot as plt

file_name = r"training (2) (1).xlsx"
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


# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_excel("testing (2) (1).xlsx")
df2 = pd.read_excel("training (2) (1).xlsx")

output1 = df1['output'].values
output2 = df2['output'].values

r_values = range(1, 11)

distances1 = [np.mean([(np.abs(output1 - y) ** r) ** (1 / r) for y in output2]) for r in r_values]
distances2 = [np.mean([(np.abs(x - output2) ** r) ** (1 / r) for x in output1]) for r in r_values]

plt.plot(r_values, distances1, label='Dataset 1', marker='o', linestyle='-')
plt.plot(r_values, distances2, label='Dataset 2', marker='x', linestyle='-')
plt.xlabel('r')
plt.ylabel('Mean Minkowski Distance')
plt.title('Mean Minkowski Distance vs. r')
plt.legend()
plt.grid(True)
plt.show()


# In[27]:


import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = t5_train.values
y = dtrain['Classification'].values

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X, y)
print(neigh.fit(X, y))
print(X)
print(y)


# In[28]:


from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X, y)

accuracy = neigh.score(X, y)
print(f"Accuracy: {accuracy:.2f}")


# In[29]:


predicted_labels = neigh.predict(X)

for i in range(len(X)):
    print(f"Predicted: {predicted_labels[i]}, Actual: {y[i]}")


# In[34]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


accuracies_kNN = []
accuracies_NN = []

X, X_test, y, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
k_values = range(1, 12)

for k in k_values:
    
    kNN_classifier = KNeighborsClassifier(n_neighbors=k)
    kNN_classifier.fit(X, y)
    
    
    y_pred_kNN = kNN_classifier.predict(X_test)
    
   
    accuracy_kNN = accuracy_score(y_test, y_pred_kNN)
    accuracies_kNN.append(accuracy_kNN)

    
        
    NN_classifier = KNeighborsClassifier(n_neighbors=1)
    NN_classifier.fit(X, y)
        
       
    y_pred_NN = NN_classifier.predict(X_test)
        
        
    accuracy_NN = accuracy_score(y_test, y_pred_NN)
    accuracies_NN.append(accuracy_NN)


plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies_kNN, marker='o', label='kNN (k=3)')
plt.plot(k_values, accuracies_NN, marker='o', label='NN (k=1)')

plt.title('Accuracy vs. k Value')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()


# In[35]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

train_predictions = neigh.predict(X_train)
test_predictions = neigh.predict(X_test)

train_confusion_matrix = confusion_matrix(y_train, train_predictions)
train_classification_report = classification_report(y_train, train_predictions)

test_confusion_matrix = confusion_matrix(y_test, test_predictions)
test_classification_report = classification_report(y_test, test_predictions)

print("Confusion Matrix (Training Data):\n", train_confusion_matrix)
print("\nClassification Report (Training Data):\n", train_classification_report)

print("\nConfusion Matrix (Test Data):\n", test_confusion_matrix)
print("\nClassification Report (Test Data):\n", test_classification_report)


# In[ ]:




