import pandas as pd
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')
## データの取得
df = pd.read_csv('may_toyohashi.csv')
df.head()
## 説明変数と目的変数
x = df[['rain','sunshine','wing']]
y = df['temp']
## 線型回帰オブジェクト、モデル作成
reg = linear_model.LinearRegression()
reg.fit(x, y)
## 係数の導出
print(f"傾き:{reg.coef_}")
print(f"切片:{reg.intercept_}")
## 予測の実行
temp = reg.predict([[217.5,209.4,3.4]])
print(f"予測:{temp}")