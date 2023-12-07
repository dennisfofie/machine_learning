from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = pd.Series(iris.target)

df['target_names'] = df['target'].apply(lambda y: iris.target_names[y])
# print(df.sample(10))
# print()
# print(df.head())

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.3)

X_train = df_train.drop(columns=['target', 'target_names'], axis=1)
X_test = df_test[iris.feature_names]

Y_train = df_train['target']
Y_test = df_test['target']

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

clf.fit(X_train, Y_train)
y_test_pred = clf.predict(X_test)

cost_decision = clf.cost_complexity_pruning_path(X=X_test, y=y_test_pred)
print(f"cost_decision: -- {cost_decision}")

count = 0

for i, j in zip(y_test_pred, Y_test):
    if i != j:
        count += 1

print(count)

from sklearn.metrics import accuracy_score

score = accuracy_score(Y_test, y_test_pred)
print(score)

hello = pd.DataFrame(
  {
    'feature_names': iris.feature_names,
    'feature_importances': clf.feature_importances_
  }
).sort_values(
  'feature_importances', ascending=False
).set_index('feature_names')

print(hello)
