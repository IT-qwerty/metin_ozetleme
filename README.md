# Çok Dilli Doğal Dil İşleme Tabanlı Otomatik Metin Özetleme Sistemi

Bu proje, farklı dillerdeki (özellikle Türkçe) uzun metinleri otomatik olarak özetleyebilen ve kullanıcıya kısa, anlamlı içerikler sunan hibrit bir Doğal Dil İşleme (NLP) yazılımıdır.

## Proje Hakkında
Günümüzde yapay zeka tabanlı özetleme sistemleri genellikle tek bir yönteme dayanmaktadır. Bu proje, **Transformer tabanlı modeller** kullanılarak hem **Çıkarımsal (Extractive)** hem de **Tümevarımsal (Abstractive)** özetleme tekniklerini aynı çatı altında birleştiren hibrit bir yaklaşım sunar. 

Sistem; akademik makaleler, haberler ve raporlar gibi farklı türdeki metinleri analiz ederken, Abstractive modellerin en büyük dezavantajı olan "halüsinasyon (bilgi uydurma)" problemini Extractive modül ile telafi edecek şekilde tasarlanmıştır.

## Kullanılan Modeller ve Teknolojiler

* **Tümevarımsal (Abstractive) Modül:** Google `mT5-base` mimarisi kullanılmıştır. Model, BBC XL-Sum veri seti üzerinde *Fine-Tuning* işleminden geçirilerek Türkçe metinleri kendi kelimeleriyle yeniden yazacak şekilde eğitilmiştir.
* **Çıkarımsal (Extractive) Modül:** Çok dilli `Sentence-BERT (mBERT)` mimarisi kullanılmıştır. Metin içindeki cümleler vektör uzayına dönüştürülüp, *Cosine Similarity* (Kosinüs Benzerliği) algoritmasıyla ana fikre en yakın olan orijinal cümleler cımbızlanarak seçilir.
* **Arayüz:** Kullanıcı deneyimi için **Gradio** kütüphanesi ile web tabanlı bir arayüz geliştirilmiştir.

## Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda veya Google Colab üzerinde çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

**1. Gerekli kütüphaneleri yükleyin:**
```bash
pip install transformers datasets sentence-transformers gradio nltk accelerate
2. Projeyi klonlayın:

Bash
git clone [https://github.com/IT-qwerty/metin_ozetleme.git](https://github.com/IT-qwerty/metin_ozetleme.git)
cd metin_ozetleme
3. Arayüzü başlatın:

Bash
python summarize.py
(Not: summarize.py dosyası içindeki model yükleme yollarını, modelinizi indirdiğiniz kendi yerel dizininize göre güncellemeniz gerekebilir.)

Teknik Analiz ve Çıkarımlar
Proje geliştirme sürecinde, Abstractive modellerin (mT5) eğitim veri setindeki (haberler) jargona aşırı uyum sağladığı (overfitting) ve akademik metinlerde veri seti eğilimi kaynaklı "alan kayması (domain shift)" yaşayabildiği gözlemlenmiştir. Bu durum, akademik ve resmi metinler için Extractive (BERT) yönteminin neden sisteme entegre edilmesi gerektiğini mühendislik açısından doğrulamaktadır.

Bu proje, bilgisayar mühendisliği lisans eğitimi kapsamında geliştirilmiştir.
