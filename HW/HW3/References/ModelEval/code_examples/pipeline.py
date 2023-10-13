import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
#standard import and split code

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, \
    test_size=0.20, 
    stratify=y,
    random_state=1)

#This is the setup; nothing gets executed yet…
pipe_lr = make_pipeline(StandardScaler(), \
    PCA(n_components=2),
    LogisticRegression(random_state=1))
#This will evoke the pipeline
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))