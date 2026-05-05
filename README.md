# metin_ozetleme
ogrenci projeleri metin ozetleme

Cok Dilli Doğal Dil İşleme Tabanlı Otomatik Metin Özetleme Sistemi
1. Proje Özeti
Bu proje, uzun metinlerin içeriğini anlamını bozmadan kısaltmayı amaçlayan hibrit bir metin özetleme sistemidir. Sistem, modern Doğal Dil İşleme (NLP) tekniklerinden hem Tümevarımsal (Abstractive) hem de Çıkarımsal (Extractive) özetleme mimarilerini bir arada sunar.

2. Kullanılan Teknolojiler ve Modeller
Projede iki farklı Transformer tabanlı mimari kullanılmıştır:

Abstractive Modül (mT5 - Multilingual T5): Metni okuyup kendi kelimeleriyle yeni bir özet cümlesi kurar. Google tarafından geliştirilen bu model, projemiz kapsamında özelleştirilmiştir.

Extractive Modül (mBERT / Sentence-BERT): Metin içindeki en önemli cümleleri matematiksel benzerlik skorlarına (Cosine Similarity) göre cımbızlayarak seçer.

Arayüz: Kullanıcı dostu etkileşim için Gradio kütüphanesi kullanılmıştır.

3. Veri Seti ve Eğitim Süreci (Fine-Tuning)
Modelin eğitimi için şu metodoloji izlenmiştir:

Veri Seti: BBC kaynaklı, 44 dilde profesyonel özetler barındıran XL-Sum (Turkish) veri seti kullanılmıştır.

Kapsam: Eğitim hızı ve kaynak yönetimi için ilk aşamada 5.000 adet gerçek haber makalesi ve özeti üzerinde çalışılmıştır.

Donanım Optimizasyonu: Google Colab (T4 GPU) üzerinde gerçekleştirilen eğitimde, bellek hatalarını (Out of Memory) önlemek için Gradient Checkpointing ve Adafactor Optimizer gibi ileri seviye teknikler uygulanmıştır.

4. Uygulama Mimarisi (Hibrit Yaklaşım)
Sistem iki koldan çalışmaktadır:

Tümevarımsal Kol: Kullanıcıdan alınan metin mT5 modeline beslenir. Model, öğrendiği dil bilgisi kuralları ve haber dili şablonları ile yeni bir özet üretir.

Çıkarımsal Kol: Metin cümlelerine ayrılır. Her bir cümlenin anlamsal vektörü çıkarılır ve ana metnin genel vektörüne en yakın olan "en önemli" cümleler seçilir.

5. Teknik Analiz ve Değerlendirme
Proje geliştirme sürecinde önemli mühendislik çıkarımları elde edilmiştir:

Halüsinasyon Problemi: Abstractive modellerin (mT5), eğitim veri setindeki haber diline aşırı uyum sağladığı (overfitting) ve akademik metinlerde veri seti eğilimi (bias) nedeniyle uydurma bilgiler üretebildiği gözlemlenmiştir.

Hibrit Çözümün Avantajı: Abstractive modelin yaratıcılığı ile Extractive modelin (BERT) sadakati birleştirilerek, farklı metin türlerinde (haber, akademik metin, rapor) en güvenilir sonucun elde edilmesi hedeflenmiştir.

6. Kurulum ve Çalıştırma
Proje Google Colab üzerinde T4 GPU hızlandırıcı kullanılarak çalıştırılmaktadır. Gerekli kütüphaneler:

pip install transformers datasets sentence-transformers gradio nltk
