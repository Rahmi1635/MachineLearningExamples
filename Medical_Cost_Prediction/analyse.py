import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
import joblib 

#1 - Veriyi Yükleme
df=pd.read_csv("medical-charges.csv")


#2 - boolean verileri bite çevirme işlemi 
df['smoker_numeric']=df['smoker'].map({'yes':1,'no':0})
df['sex_numeric']=df['sex'].map({'male':1,'female':0})

#Her bölgeyi ayrı birer değişken yapma

df_region=pd.get_dummies(df['region'],prefix='reg',drop_first=True,dtype=int)
df=pd.concat([df,df_region],axis=1)

df['bmi_smoker']=df['bmi']*df['smoker_numeric']

df['age_smoker']=df['age']*df['smoker_numeric']

df['age_sq']=df['age']**2

#3 - Korelasyon analizi yapıldı

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm")
plt.show()

#4 - train/test ayrımı ve model eğitimi

y=df['charges']
X=df[["age","age_sq","bmi","children","smoker_numeric","sex_numeric","reg_northwest", "reg_southeast", "reg_southwest","bmi_smoker","age_smoker"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

regr=linear_model.LinearRegression()
regr.fit(X_train,y_train)
y_pred=regr.predict(X_test)

#5 - Hata ve başarı skoru 

score=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
print(f"Başarı Skoru : {score:.2f}")
print(f"Ortalama Mutlak Hata : {mae:.2f}")

#6 - Model Kaydetme 

joblib.dump(regr,'medical_insurance_model.joblib')

print("Model Başarıyla kaydedildi! Dosya Adı : medical_insurance_model.joblib")
