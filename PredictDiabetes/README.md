🩺 Diabetes Prediction with KNN & Machine Learning
Bu proje, Pima Indians Diabetes veri seti kullanılarak, bir kişinin sağlık parametrelerine göre diyabet riskini tahmin eden uçtan uca bir makine öğrenmesi uygulamasıdır. Projede K-Nearest Neighbors (KNN) algoritması kullanılmış ve tıbbi hassasiyetler göz önünde bulundurularak optimize edilmiştir.

🚀 Proje Özellikleri
Modüler Yapı: Veri temizleme (DataCleaning) ve modelleme işlemleri birbirinden ayrı modüller halinde tasarlanmıştır.

Veri Temizleme (Data Cleaning): Mantıksız 0 değerleri (Glikoz, Kan Basıncı vb.) tespit edilip medyan değerler ile doldurulmuştur.

Ölçeklendirme (Feature Scaling): Mesafe tabanlı KNN algoritmasının doğru çalışması için MinMaxScaler uygulanmıştır.

Hiperparametre Optimizasyonu: GridSearchCV kullanılarak en yüksek Recall değerini veren K komşu sayısı ve mesafe metrikleri belirlenmiştir.

Model Kalıcılığı: Eğitilmiş model ve veri terazisi (scaler), tekrar kullanılabilmesi için .joblib formatında kaydedilmiştir.

🛠 Kullanılan Teknolojiler
Dil: Python

Kütüphaneler: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

Model Kayıt: Joblib

📊 Model Performansı ve Analiz
Modelimiz sadece genel doğruluğa (Accuracy) değil, hastaların doğru tespit edilmesine (Recall) odaklanmıştır.

Neden Recall Odaklı?
Tıbbi teşhislerde "yanlış negatif" (hasta birine sağlıklı demek) riski kritik olduğu için GridSearchCV aşamasında Recall skoru optimize edilmiştir. Yapılan optimizasyonlar sonucunda:

K Değeri: 9

Ağırlık Tipi: Uniform

Genel Doğruluk (Accuracy): %76

Hata Analizi: Kritik "hastayı kaçırma" oranı minimize edilmiştir.