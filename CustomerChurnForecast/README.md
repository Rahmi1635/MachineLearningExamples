📊 Müşteri Kaybı Tahmini (Customer Churn Forecast)
Bu proje, bir telekomünikasyon şirketinin verilerini kullanarak, hangi müşterilerin hizmeti bırakma (churn) eğiliminde olduğunu tahmin etmek amacıyla geliştirilmiştir. Projenin temel odak noktası, şirketin finansal kayıplarını önlemek için ayrılma ihtimali yüksek olan müşterileri doğru tespit etmektir. 📱💸

🎯 İş Etkisi ve Sonuçlar
Model geliştirme sürecinde yapılan optimizasyonlar sonucunda:

Müşteri Yakalama Kapasitesi: Ayrılacak müşterileri doğru tahmin etme oranı (Recall) %57'den %74'e yükseltilmiştir. 🚀

Şirket Riski Yönetimi: "Ayrılacak" olarak etiketlenen müşterilere yönelik proaktif aksiyonlar alınması (kampanya, indirim vb.) sağlanarak müşteri kaybı riski minimize edilmiştir. 🛡️

🛠️ Kullanılan Teknolojiler
Python 3.x 🐍

Pandas & NumPy: Veri manipülasyonu ve analizi.

Scikit-Learn: Makine öğrenmesi modeli ve değerlendirme metrikleri.

Joblib: Eğitilen modelin kaydedilmesi.

⚙️ Uygulanan Adımlar
Veri Ön İşleme: TotalCharges gibi sayısal olmayan veriler dönüştürüldü ve eksik veriler temizlendi. 🧹

Özellik Mühendisliği (Encoding): Kategorik değişkenler One-Hot Encoding ve Label Encoding ile sayısal forma getirildi. 🔢

Sınıf Dengelenmesi: Veri setindeki dengesizliği gidermek için class_weight='balanced' parametresi kullanılarak azınlık sınıfın (ayrılan müşteriler) önemi artırıldı. ⚖️

Hiperparametre Optimizasyonu: GridSearchCV ile en ideal max_depth (derinlik) değeri 5 olarak belirlendi. 🌳

Model Kayıt: En iyi performansı veren model customer_churn_model.joblib olarak kaydedildi. 💾

📈 Model Performansı (Confusion Matrix)
Model, özellikle ayrılacak müşterileri (True Positive) bulmaya odaklanmış ve bu grupta yüksek başarı sağlamıştır.