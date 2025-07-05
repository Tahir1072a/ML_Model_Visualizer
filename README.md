# ML Classification Algorithms Compare

![Lisans](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37.1-ff4b4b)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.0-orange)

Farklı makine öğrenmesi sınıflandırma algoritmalarının performansını ve karar sınırlarını görsel olarak karşılaştırmak için geliştirilmiş interaktif bir Streamlit uygulamasıdır.

---

### 🚀 Web Sitesi
```
https://algoscope.streamlit.app/
```
Bu linke tıklayarak web sitesi üzerinden interaktif uygulamaya erişebilirsiniz. 

---

### 📖 Proje Hakkında

Bu proje, temel sınıflandırma algoritmalarının farklı veri yapılarında nasıl davrandığını anlamak için bir eğitim ve analiz aracı olarak tasarlanmıştır. Kullanıcılar, sentetik veri setleri üzerinde gürültü ve örnek sayısı gibi parametreleri değiştirerek ve modellerin hiperparametreleriyle oynayarak algoritmaların güçlü ve zayıf yönlerini sezgisel olarak keşfedebilirler.

Ana hedef, teorik bilgiyi pratik ve görsel bir deneyime dönüştürmektir.

---

### ✨ Özellikler

* **İnteraktif Arayüz:** Streamlit kullanılarak geliştirilmiş, kullanıcı dostu ve hızlı bir arayüz.
* **Çeşitli Veri Setleri:**
    * Lineer olarak ayrılabilen kümeler
    * İç içe geçmiş aylar (Moons)
    * Eş merkezli daireler (Circles)
* **Dinamik Veri Kontrolü:** Örnek sayısını ve veri setindeki gürültü seviyesini anlık olarak ayarlayabilme.
* **Ayarlanabilir Hiperparametreler:** KNN, SVM, Decision Tree ve Random Forest gibi popüler modellerin temel hiperparametrelerini yan panelden kontrol edebilme.
* **Karşılaştırmalı Görselleştirme:**
    * Tüm algoritmaların karar sınırlarını (decision boundaries) ve test verisi üzerindeki performanslarını bir ızgara yapısında yan yana görme.
    * Her bir model için doğruluk (accuracy) skorunu grafik üzerinde anında görüntüleme.
* **Detaylı Performans Metrikleri:** Her model için `Classification Report` (Precision, Recall, F1-Score) ve `Confusion Matrix`'i inceleyebilme.

---

### 🛠️ Kullanılan Teknolojiler

Bu proje aşağıdaki teknolojiler kullanılarak geliştirilmiştir:

* **Python:** Ana programlama dili.
* **Streamlit:** İnteraktif web uygulamasını oluşturmak için.
* **Scikit-learn:** Makine öğrenmesi modelleri, veri setleri ve metrikler için.
* **Pandas:** `Classification Report`'u yapılandırmak için.
* **Matplotlib:** Karar sınırlarını ve veri dağılımını görselleştirmek için.
* **Numpy:** Veri manipülasyonu için.

---

### ⚙️ Kurulum ve Başlatma

Bu projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

**1. Ön Gereksinimler**
* Python 3.9 veya daha üstü bir sürümün yüklü olması gerekmektedir.

**2. Projeyi Klonlayın**
```bash
git clone https://github.com/Tahir1072a/ML_Model_Visualizer
```

**3. Gerekli Kütüphaneleri Yükleyin**
```bash
pip install -r requirements.txt
```

**4. Uygulamayı Başlatın**
```bash
streamlit run app.py
```
Uygulama https://algoscope.streamlit.app/ sitesinde deploy edilmiştir. Localde açmak için yukarıda verilen komutları sırayla çalıştırabilirsiniz.
---

### 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Daha fazla bilgi için `LICENSE` dosyasına göz atın.

---

### İletişim Bilgileri

Tahiri Fidan - https://www.linkedin.com/in/thr-fdn-4a88a620a/ - tahirifdn@gmail.com
