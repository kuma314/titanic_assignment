import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report, accuracy_score
# # trainデータ
train = pd.read_csv('train.csv')
# testデータ
test = pd.read_csv('test.csv')
# # サンプル提出ファイル
sample_file = pd.read_csv('gender_submission.csv')
# # 'Cabin'は欠損値が多いので、削除
train = train.drop('Cabin', axis = 1)
test = test.drop('Cabin', axis = 1)
# 'Age'の欠損値を平均値補完
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())
# # 'Sex'をマッピング
sex_mapping = {'male':0, 'female':1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)
# # X, yにデータを代入
X = train.loc[:, ['Sex', 'Age', 'SibSp', 'Parch']].values
y = train.loc[:, ['Survived']].values.reshape(-1)

# # データの標準化
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# # ホールド・アウト法による分割
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=0)
# # カーネルSVMによる学習
# # gamma: 決定曲線の複雑さ
# # C: 誤分類への厳しさ
svm = SVC(kernel='rbf', gamma=0.1, C=10)
svm.fit(X_train, y_train)
joblib.dump(svm, "nn.pkl", compress=True)
pred = svm.predict(X_test)
print("result: ", svm.score(X_test, y_test))
print(classification_report(y_test, pred))
