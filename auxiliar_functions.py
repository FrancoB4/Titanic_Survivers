import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def read_csv(path):
    df = pd.read_csv(path)
    return df


def one_preprocessing(df):
  try:
    x, y = df.drop(['Survived', 'Name', 'Ticket'], axis=1), df['Survived']
  except:
    x = df.drop(['Name', 'Ticket'], axis=1)
  x['Age'] = x['Age'].fillna(int(np.mean(df['Age'])))
  x['Cabin'] = x['Cabin'].fillna('U').apply(lambda x: x[0])

  for column in ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female',
       'Sex_male', 'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E',
       'Cabin_F', 'Cabin_G', 'Cabin_T', 'Cabin_U', 'Embarked_C', 'Embarked_Q',
       'Embarked_S']:

    if column not in pd.get_dummies(x).columns:
      x[column] = np.zeros(len(x))
  
  X = pd.get_dummies(x).fillna(method='ffill')
  X = X[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female',
       'Sex_male', 'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E',
       'Cabin_F', 'Cabin_G', 'Cabin_T', 'Cabin_U', 'Embarked_C', 'Embarked_Q',
       'Embarked_S']]
  try:
    return X, y
  except:
    return X


def ordinal_preprocessing(df):
  try:
    x, y = df.drop(['Survived', 'Name', 'Ticket'], axis=1), df['Survived']
  except:
    x = df.drop(['Name', 'Ticket'], axis=1)
  x['Age'] = x['Age'].fillna(int(np.mean(x['Age'])))
  x['Cabin'] = OrdinalEncoder().fit_transform(np.array(x['Cabin'].fillna('Unknow').apply(lambda x: x[0])).reshape(-1, 1))
  x['Sex'] = OrdinalEncoder().fit_transform(np.array(x['Sex']).reshape(-1, 1))
  x['Embarked'] = OrdinalEncoder().fit_transform(np.array(x['Embarked']).reshape(-1, 1))
  x['Embarked'] = x['Embarked'].fillna(0.0)
  try:
    return x.fillna(method='ffill'), y
  except:
    return x.fillna(method='ffill')