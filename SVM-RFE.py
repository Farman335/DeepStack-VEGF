# Avoiding warning
import warnings
def warn(*args, **kwargs): pass
warnings.warn = warn
# _______________________________

# Essential Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# _____________________________

# np.random.seed(seed=111)

# scikit-learn :
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import *
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
cv = StratifiedKFold(n_splits=5, shuffle=True)


iRec = 'hyb_trainL2.csv'
D = pd.read_csv(iRec)#header=None)  # Using pandas

x = D.iloc[:, :-1].values
y = D.iloc[:, -1].values
# feature extraction
selector=RFE(estimator=ExtraTreesClassifier(),n_features_to_select=300)
X=selector.fit_transform(x,y)

for (train_index, test_index) in cv.split(X, y):
    X_train = X[train_index]
    X_test = X[test_index]

    y_train = y[train_index]
    y_test = y[test_index]

clf=ExtraTreesClassifier()

clf.fit(X_train,y_train)

print(clf.feature_importances_)
print("Test score:%s"%(clf.score(X_test,y_test)))
