import pandas as pd
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('./DataSet/train.csv')
test = pd.read_csv('./DataSet/test.csv')

train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

new_data_train = pd.get_dummies(train)
new_data_test = pd.get_dummies(test)

new_data_train['Age'].fillna(new_data_train['Age'].mean(), inplace=True)
new_data_test['Age'].fillna(new_data_test['Age'].mean(), inplace=True)

new_data_test['Fare'].fillna(new_data_test['Fare'].mean(), inplace=True)

X = new_data_train.drop('Survived', axis=1)
y = new_data_train['Survived']

tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X, y)
tree.score(X, y)

submission = pd.DataFrame()
submission['PassengerId'] = new_data_test['PassengerId']
submission['Survived'] = tree.predict(new_data_test)

submission.to_csv('./ResultSet/submission.csv', index=False)

