from flask import Flask, render_template, request
from wtforms import Form, FloatField, SubmitField, validators, ValidationError
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

test = pd.read_csv('test.csv')
test = test.drop('Cabin', axis = 1)
test['Age'] = test['Age'].fillna(test['Age'].mean())
sex_mapping = {'male':0, 'female':1}
test['Sex'] = test['Sex'].map(sex_mapping)
test_data = test.loc[:, ['Sex', 'Age', 'SibSp', 'Parch']].values
# 学習モデルを読み込み予測する
def predict(parameters):
    # モデル読み込み
    model = joblib.load('./nn.pkl')
    params = parameters.reshape(1,-1)
    pred = model.predict(params)
    return pred


def life(label):
    print(label)
    if label == 0:
        return "dead"
    elif label == 1: 
        return "alive"
    else: 
        return "Error"

app = Flask(__name__)

# Flaskとwtformsを使い、index.html側で表示させるフォームを構築する
class titanicForm(Form):
    Sex = FloatField("性別　男性は０　女性は１としてください",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=1)])

    Age  = FloatField("年齢",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=120)])

    SibSp = FloatField("タイタニックに同乗している兄弟/配偶者の数",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=100)])

    Parch  = FloatField("タイタニックに同乗している親/子供の数",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=100)])

    # html側で表示するsubmitボタンの表示
    submit = SubmitField("判定")

@app.route('/', methods = ['GET', 'POST'])
def predicts():
    form = titanicForm(request.form)
    if request.method == 'POST':
        if form.validate() == False:
            return render_template('index.html', form=form)
        else:            
            Sex = float(request.form["Sex"])   
            Age  = float(request.form["Age"])          
            SibSp = float(request.form["SibSp"])            
            Parch  = float(request.form["Parch"])

            x = np.array([Sex, Age, SibSp, Parch]).reshape(1, -1)
            x_a=np.concatenate([x, test_data])
            scaler = StandardScaler()
            x_std = scaler.fit_transform(x_a)
            x_b=x_std[0].reshape(1, -1)
            pred = predict(x_b)
            life_saved = life(pred)

            return render_template('result.html', life_a=life_saved)
    elif request.method == 'GET':

        return render_template('index.html', form=form)

if __name__ == "__main__":
    app.run()