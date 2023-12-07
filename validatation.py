from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

X, Y = make_classification(n_samples=500, n_classes=5, n_features=50, n_informative=10, n_redundant=5, n_clusters_per_class=3, random_state=1000)

ss = StandardScaler()

X = ss.fit_transform(X)


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, StratifiedKFold, cross_val_score

lr_model  = LogisticRegression(solver='lbfgs', random_state=1000)
splits = StratifiedKFold(n_splits=10, shuffle=True, random_state=1000)


train_size = np.linspace(0.1,1.0, 20)

lr_train_sizes , lr_train_scores, lr_test_scores = learning_curve(lr_model, X, Y, cv=splits, train_sizes=train_size, n_jobs=-1, scoring='accuracy', shuffle=True, random_state=1000)


mean_scores = []

cvs = [ x for x in range(5,100, 10)]

for cv in cvs:
    score = cross_val_score(LogisticRegression(solver='lbfgs', random_state=1000), X, Y, scoring='accuracy', n_jobs=-1, cv=cv)

    mean_scores.append(np.mean(score))

print(mean_scores)