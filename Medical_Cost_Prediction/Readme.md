🏥 Sağlık Sigortası Maliyet Analizi ve Tahmini
Bu proje, bir bireyin kişisel ve fiziksel özelliklerini analiz ederek yıllık sağlık sigortası maliyetini (medical charges) yüksek doğrulukla tahmin eden bir makine öğrenmesi çalışmasıdır. Proje boyunca uygulanan veri önişleme ve özellik mühendisliği teknikleriyle model başarısı %78'den %87'ye yükseltilmiştir.

🚀 Proje Amacı
Sigorta şirketleri için risk analizi hayati önem taşır. Bu çalışmada, basit bir lineer modelin ötesine geçilerek; yaş, BMI ve yaşam tarzı (sigara) arasındaki karmaşık ilişkiler matematiksel olarak modellenmiş ve tahmin hatası minimize edilmiştir.

Metrik   ,Başlangıç Modeli  ,Gelişmiş Model (Final)
R² Skoru (Başarı oranı),%78  ,%87
MAE (Ortalama Hata),~4100 $  ,2757 $

🛠️ Uygulanan Mühendislik Adımları
Modeli %78 bandından yukarı taşımak için aşağıdaki stratejik adımlar uygulanmıştır:

Veri Dönüştürme (Encoding): * sex ve smoker gibi ikili veriler bit bazlı (0 ve 1) hale getirildi.

region verisi, modelin coğrafi farklılıkları anlayabilmesi için sayısal verilere (One-Hot Encoding) dönüştürüldü.

Etkileşim Özellikleri (Interaction Terms): * Sadece yaş veya sadece sigara kullanımının değil, bu iki faktörün birleştiğinde maliyeti nasıl katladığını göstermek için age_smoker ve bmi_smoker değişkenleri üretildi.

Polinomsal Geliştirme (Non-linear Mapping): * Yaş ilerledikçe sağlık harcamalarının doğrusal değil, artan bir ivmeyle (parabolik) yükseldiği tespit edildi. Bu eğriyi yakalamak için yaşın karesi (age_sq) modele dahil edildi


📈 Bulgular
Sigara Faktörü: Maliyet üzerindeki en güçlü belirleyici olduğu (0.79 korelasyon) kanıtlanmıştır.

VKI ve Sigara Kombo Etkisi: Sigara içen bireylerde vücut kitle indeksindeki (VKI) artışın, içmeyenlere göre maliyeti çok daha sert yükselttiği gözlemlenmiştir.

Hata Payı: Yapılan optimizasyonlar sayesinde modelin tahminlerindeki ortalama sapma (MAE) yaklaşık 1350 $ düşürülmüştür.


💻 Kullanılan Teknolojiler
Python 3.x

Pandas & NumPy (Veri manipülasyonu)

Scikit-Learn (Model eğitimi ve metrikler)

Seaborn & Matplotlib (Görselleştirme ve korelasyon analizi)

Joblib (Modelin kaydedilmesi ve canlıya alınması)