import numpy as np
from sklearn.svm import SVC
X = np.array([[-1, -1,5], [-2, -1,5], [1, 1,4], [2, 1,3]])
y = np.array([1, 1, 2, 4])
clf = SVC(C=1.0, cache_size=2000)
clf.fit(X, y) 
print(clf.predict([[-0.8, -1,4]]))