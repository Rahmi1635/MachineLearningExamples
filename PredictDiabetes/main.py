from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from DataCleaning.cleaning import clean_database,scale_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1 -veriyi Yükleme,Temizleme,Ölçeklendirme

df=pd.read_csv("diabetes.csv")
df=clean_database(df)

X,y,scaler=scale_data(df)

# 2 - Train/Test Ayrımı

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3 - Parametre Optimizasyonu(GridSearch)
# Önce en iyi parametreleri bulalım ki final modelini ona göre kuralım

param_grid={
    'n_neighbors':range(1,31),
    'weights':['uniform','distance'],
    'metric':['euclidean','manhattan']
}

grid=GridSearchCV(KNeighborsClassifier(),param_grid,cv=5,scoring="recall")
grid.fit(X_train,y_train)

print(f"Hastaları en iyi yakalayan (Recall odaklı) K: {grid.best_params_['n_neighbors']}")
print(f"En iyi ağırlık tipi: {grid.best_params_['weights']}")

# 4 - Modeli Eğitme
best_knn=grid.best_estimator_ # bulduğu en iyi parametrelerle (K=9, uniform vs.) zaten eğitilmiş bir model döndürür.
best_knn.fit(X_train,y_train)

# 5 - (5-Fold) Çapraz Doğrulama yapalım
# cv=5 => veriyi 5'e böl ve 5 farklı deney yap demek

cv_scores=cross_val_score(best_knn,X,y,cv=5)

print(f"Ortalama Doğruluk (CV Accuracy): {np.mean(cv_scores):.4f}")
print(f"Standart Sapma: {np.std(cv_scores):.4f}")

# 5 - Tahmin Yapma ve Raporlama

y_pred=best_knn.predict(X_test)

print("\n--- Final Model Performans Raporu ---")
print(f"Test Seti Doğruluk Oranı : {accuracy_score(y_test,y_pred):.4f}")
print("\nDetaylı Sınıflandırma Raporu : ")
print(classification_report(y_test, y_pred))

# 6 - Veriyi Görselleştirme 

cm=confusion_matrix(y_test,y_pred)


plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt='d',cmap="Blues",
            xticklabels=["Sağlıklı (0)","Diyabet (1)"],
            yticklabels=["Sağlıklı (0)","Diyabet (1)"])

plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')
plt.title('Diyabet Tahmini - Confusion Matrix')
plt.show()

# 7 - Örnekleme

yeni_veri=np.array([[2,150,85,33,190,34.5,0.5,45]])
#"150 Glikoz" değerinin, 0 ile 1 arasında nereye denk geldiğini eğitimdeki minimum ve maksimum değerlere bakarak anlamalı.
yeni_veri_scaled=scaler.transform(yeni_veri)

# 3. Tahmin yap
tahmin=best_knn.predict(yeni_veri_scaled)
olasilik=best_knn.predict_proba(yeni_veri_scaled)

# 4. Sonucu Yazdır

print(f"\n--- Yeni Veri Tahmin Sonucu ---")
if tahmin[0] == 1:
    print("Sonuç: DİYABET RİSKİ VAR (Sınıf 1)")
else:
    print("Sonuç: SAĞLIKLI GÖRÜNÜYOR (Sınıf 0)")

print(f"9 Komşu Arasındaki Oran: {olasilik[0]}")

# 8 - Modeli Kaydet 

joblib.dump(best_knn,"diabets_model.joblib")

joblib.dump(scaler,"diabet_scaler.joblib")

print("\n--- Model ve Scaler başarıyla kaydedildi! ---")


