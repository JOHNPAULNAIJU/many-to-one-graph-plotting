from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from data import user_interests #array of interests

vectorizer = CountVectorizer()

vectorized_array = vectorizer.fit_transform(user_interests)

dense_array = vectorized_array.toarray()

array = []

for i in dense_array[0]:
    for j in dense_array[i]:
        array.append(j)

npArray = np.array(array)
mean = np.mean(npArray)

plt.scatter(mean, 0)
plt.show()
#use pirincipal component analysis for more understanding
# pca = PCA(n_components=2)
# xy_coordinates = pca.fit_transform(dense_array)

# plt.scatter(xy_coordinates[:,0], xy_coordinates[:,1])
# plt.show()
