import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import joblib

# 1. Veriyi Yükleme
df = pd.read_csv("Telco-Customer-Churn.csv")

# 2. Veri Temizleme: TotalCharges sütununu sayıya çevirme ve boşları silme
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# 3. Basit Dönüşümler: 'Yes'/'No' ve Cinsiyet sütunlarını 1-0 yapma
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})

# 4. Gereksiz sütunu atma
df.drop("customerID", axis=1, inplace=True)

# 5. One-Hot Encoding: Geriye kalan tüm metin (object) sütunlarını sayıya çevirir
# Bu işlem sütun sayısını artıracak ama modeli çalışır hale getirecektir.
df = pd.get_dummies(df) #One-Hot Encoding yöntemini kullanır


# 6. X ve y Belirleme
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 7. Train/Test Ayırma (%80 Eğitim - %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. GridSearchCV ile En İyi Derinliği Bulma
dt_model=DecisionTreeClassifier(random_state=42,class_weight="balanced")
param_grid = {"max_depth": [3, 5, 7, 10, 15, None]}
grid_search = GridSearchCV(dt_model, param_grid=param_grid, cv=5)

# Modeli eğitme
grid_search.fit(X_train, y_train)


# Sonuçları yazdırma
print(f"En iyi derinlik değeri: {grid_search.best_params_}")
print(f"En iyi eğitim skoru: {grid_search.best_score_:.4f}")

# En iyi modeli seçme 
best_model=grid_search.best_estimator_

# Test verisiyle tahmin yapalım
y_pred=best_model.predict(X_test)

#Sonuçlar
print("--- Karmaşıklık Matrisi ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Sınıflandırma Raporu ---")
print(classification_report(y_test, y_pred))
accuracy=accuracy_score(y_test,y_pred)
print("\n--- Doğruluk Oranı ---")
print(accuracy)

#Modeli Kaydetme

joblib.dump(best_model,"customer_churn_model.joblib")