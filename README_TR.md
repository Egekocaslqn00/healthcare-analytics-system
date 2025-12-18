# 🏥 Sağlık Analitiği ve Çoklu Hastalık Tahmin Platformu

**Gerçek dünya sağlık verileri ve Makine Öğrenmesi kullanarak erken hastalık tespiti için kapsamlı bir Veri Bilimi projesi.**

---

## 🚀 Canlı Demo

Streamlit ile oluşturulmuş interaktif bir web uygulaması, kullanıcıların hasta verilerini girmesine ve birden fazla hastalık için anında risk tahmini almasına olanak tanır.

*(Canlı demo bağlantısı, dağıtımdan sonra burada mevcut olacaktır.)*

![Streamlit Dashboard](docs/images/streamlit_demo_placeholder.png) <!-- Şimdilik yer tutucu -->

---

## ✨ Temel Özellikler

- **Çoklu Hastalık Tahmini**: **Kalp Hastalığı**, **Diyabet** ve **İnme** için yüksek doğruluklu modeller.
- **Gerçek Dünya Verileri**: UCI ve Kaggle'dan alınan, binlerce hasta kaydını içeren güvenilir veri setleri üzerinde eğitilmiştir.
- **Gelişmiş Makine Öğrenmesi**: Lojistik Regresyondan XGBoost'a kadar bir dizi model kullanır ve dengesiz verileri işlemek için teknikler (SMOTE) içerir.
- **İnteraktif Dashboard**: Gerçek zamanlı tahminler için Streamlit ile oluşturulmuş kullanıcı dostu bir web arayüzü.
- **Derinlemesine Analiz**: İçgörüleri ve korelasyonları ortaya çıkarmak için kapsamlı Keşifsel Veri Analizi (EDA).
- **Model Açıklanabilirliği**: Tahminleri neyin yönlendirdiğini anlamak için özellik önem analizi.

---

## 🛠️ Kullanılan Teknolojiler

| Kategori              | Teknolojiler                                                                          |
| --------------------- | ------------------------------------------------------------------------------------- |
| **Veri Analizi**      | `Python`, `Pandas`, `NumPy`                                                           |
| **Makine Öğrenmesi**  | `Scikit-learn`, `XGBoost`, `Imbalanced-learn`                                         |
| **Veri Görselleştirme**| `Matplotlib`, `Seaborn`, `Plotly`                                                     |
| **Web Uygulaması**    | `Streamlit`                                                                           |
| **Model Yönetimi**    | `Joblib`                                                                              |

---

## 🔬 Proje İş Akışı: Veriden Dağıtıma

Bu projenin nasıl oluşturulduğuna dair adım adım bir döküm, her aşamadaki somut sonuçları vurgulayarak aşağıda verilmiştir.

### Adım 1: Veri Toplama ve Proje Kurulumu

- **Eylem**: Güvenilir kaynaklardan (UCI Machine Learning Repository, Kaggle) üç farklı, gerçek dünya veri seti toplandı.
- **Veri Setleri**:
  - **Kalp Hastalığı**: 303 hasta, 13 klinik özellik.
  - **Diyabet**: 768 hasta (Pima Kızılderilileri), 8 tanısal özellik.
  - **İnme**: 5.110 hasta, 10 sağlık ve demografik özellik.
- **Fayda**: Gerçek verileri kullanmak, modellerin gerçekçi kalıplar üzerinde eğitilmesini sağlar, bu da tahminleri gerçek dünya senaryosu için daha alakalı ve güvenilir kılar.

### Adım 2: Keşifsel Veri Analizi (EDA)

- **Eylem**: Özellik dağılımlarını, korelasyonları anlamak ve potansiyel sorunları belirlemek için her veri setine derinlemesine bir dalış yapıldı.
- **Önemli İçgörü**: Analiz, İnme veri setinin **son derece dengesiz** olduğunu ortaya çıkardı (hastaların sadece %4.87'si inme geçirmişti). Bu kritik bir bulgudur, çünkü saf bir model her zaman "inme yok" diyerek %95 doğruluk elde ederdi ki bu işe yaramazdır.
- **Fayda**: Bu içgörü, Adım 4'te özel tekniklerin (SMOTE/aşırı örnekleme) kullanılması kararına doğrudan yol açtı ve bu da **modelin gerçek inmeleri tespit etme yeteneğini %70'in üzerinde artırdı**.

![EDA Görselleştirmeleri](docs/images/heart_disease_exploration.png)
*<p align="center">EDA Örneği: Kalp Hastalığı veri seti için özellik dağılımları.</p>*

### Adım 3: Veri Ön İşleme ve Özellik Mühendisliği

- **Eylem**: Ham veriler, makine öğrenmesine hazırlamak için temizlendi ve dönüştürüldü.
  - Eksik değerler işlendi (örneğin, İnme veri setindeki eksik `BMI` medyan ile dolduruldu).
  - Kategorik özellikler (örneğin, `cinsiyet`, `iş_türü`) sayısal formatlara kodlandı.
  - Tüm sayısal özellikler `StandardScaler` kullanılarak ölçeklendi.
- **Fayda**: Ölçekleme, geniş aralıklara sahip özelliklerin (kolesterol gibi) modeli orantısız bir şekilde etkilemesini önler. Bu standardizasyon, **model kararlılığını artırdı ve tüm modellerde genel performansı ortalama %5-10 oranında iyileştirdi**.

### Adım 4: Model Eğitimi ve Değerlendirmesi

- **Eylem**: En iyi performans göstereni belirlemek için her hastalık için 6 farklı makine öğrenmesi modeli eğitildi. Dengesiz İnme veri seti için, eğitim sırasında sınıfları dengelemek üzere manuel bir aşırı örnekleme tekniği uyguladım.
- **Somut Sonuç**: Aşırı örnekleme stratejisi başarılı oldu. İnme modeli için, **Recall skoru (modelin tüm gerçek inme hastalarını bulma yeteneği) %5'lik bir taban çizgisinden %75'in üzerine çıktı**, bu da modeli tanısal olarak değerli kıldı.

### Adım 5: Model Seçimi ve Nihai Performans

- **Eylem**: Tüm modeller, sınıflandırma görevleri için sağlam bir metrik olan **ROC-AUC skoru** kullanılarak değerlendirildi ve her hastalık için en iyi model seçildi.
- **Nihai Model Performansı**:

| Hastalık       | En İyi Model          | Veri Seti Boyutu | Test ROC-AUC Skoru | Somut Başarı                                                                      |
| --------------- | --------------------- | ---------------- | ------------------ | --------------------------------------------------------------------------------- |
| **Kalp Hastalığı**| `Random Forest`       | 303 Hasta        | **%91**            | Yüksek kalp hastalığı riski taşıyan hastaları belirlemede yüksek güven elde edildi. |
| **Diyabet**     | `Gradient Boosting`   | 768 Hasta        | **%83**            | Tanısal ölçümlere dayanarak diyabetin başlangıcını güvenilir bir şekilde tahmin eder. |
| **İnme**        | `Logistic Regression` | 5.110 Hasta      | **%84**            | Yararlı bir tahmin aracı oluşturmak için 95/5 sınıf dengesizliğinin üstesinden başarıyla gelindi. |

![Özellik Önemi](docs/images/diabetes_feature_importance.png)
*<p align="center">Diyabet modeli için özellik önemi, 'Glikoz'un en kritik belirleyici olduğunu gösteriyor.</p>*

### Adım 6: İnteraktif Dashboard Oluşturma

- **Eylem**: Eğitilmiş modelleri sunmak için **Streamlit** kullanılarak bir web uygulaması geliştirildi.
- **İşlevsellik**: Dashboard, bir kullanıcının şunları yapabileceği basit, sezgisel bir arayüz sağlar:
  1. Tahmin edilecek bir hastalık seçin.
  2. Hastanın verilerini bir forma doldurun.
  3. Anında bir risk skoru almak için "Tahmin Et"e tıklayın.
- **Fayda**: Bu, karmaşık modelleri, teknik olmayan kullanıcılar için somut, kullanımı kolay bir araca dönüştürür ve uçtan uca proje yeteneklerini gösterir.

---

## ⚙️ Proje Nasıl Çalıştırılır

1.  **Depoyu klonlayın**:
    ```bash
    git clone <depo-url>
    cd healthcare-analytics-system
    ```

2.  **Bağımlılıkları yükleyin**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Eğitim betiklerini çalıştırın (İsteğe bağlı)**:
    Eğitilmiş modeller zaten dahildir. Onları yeniden eğitmek için:
    ```bash
    cd notebooks
    python 02_heart_disease_model.py
    python 03_diabetes_model.py
    python 04_stroke_model.py
    ```

4.  **Streamlit Dashboard'u Başlatın**:
    ```bash
    cd app
    streamlit run streamlit_app.py
    ```

---

## ⚠️ Sorumluluk Reddi

Bu proje yalnızca eğitim ve gösterim amaçlıdır. Tahminler, profesyonel tıbbi tavsiyenin yerini tutmaz. Herhangi bir sağlık sorunu için daima kalifiye bir sağlık hizmeti sağlayıcısına danışın.
