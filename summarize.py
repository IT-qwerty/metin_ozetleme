# =================================================================
# PROJE ADI: Çok Dilli NLP Tabanlı Otomatik Metin Özetleme Sistemi
# GELİŞTİRME AMACI: Okul Projesi
# TEKNOLOJİ: mT5 (Abstractive) & BERT (Extractive)
# =================================================================

# 1. Kütüphane Kurulumları
!pip install -q transformers==4.40.0 datasets==2.19.0 sentence-transformers gradio nltk accelerate

import os
import torch
import zipfile
import nltk
import gradio as gr
from google.colab import drive
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# --- YAPILANDIRMA ---
DRIVE_YOLU = "/content/drive/MyDrive/ozetleme_modeli_yedek.zip" # Burayı Drive'daki dosya adınla teyit et
YEREL_MODEL_KLASORU = "./hazir_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def projeyi_hazirla():
    """Modelleri hazırlar ve Drive'dan yükler"""
    # Drive Bağlantısı
    if not os.path.exists("/content/drive"):
        drive.mount('/content/drive')
    
    # NLTK Verileri
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    # Modeli Zipten Çıkarma (Yoksa)
    if not os.path.exists(YEREL_MODEL_KLASORU):
        print("Model zipten çıkarılıyor...")
        with zipfile.ZipFile(DRIVE_YOLU, 'r') as zip_ref:
            zip_ref.extractall(YEREL_MODEL_KLASORU)
    
    print(f"Cihaz: {DEVICE} üzerinden yükleniyor...")
    
    # Modelleri Belleğe Alma
    tokenizer = AutoTokenizer.from_pretrained(YEREL_MODEL_KLASORU)
    model_mt5 = AutoModelForSeq2SeqLM.from_pretrained(YEREL_MODEL_KLASORU).to(DEVICE)
    model_bert = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    return tokenizer, model_mt5, model_bert

# Modelleri yükleyelim
tokenizer, model_mt5, model_bert = projeyi_hazirla()

def abstractive_ozetle(metin):
    """Kendi eğittiğimiz mT5 ile tümevarımsal özet yapar"""
    input_text = "summarize: " + metin
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    
    outputs = model_mt5.generate(
        inputs["input_ids"], 
        max_length=100, 
        min_length=20, 
        length_penalty=2.5, 
        num_beams=4, 
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extractive_ozetle(metin):
    """BERT ve Cosine Similarity ile çıkarımsal özet yapar"""
    cumleler = nltk.sent_tokenize(metin)
    if len(cumleler) < 3: return metin
    
    metin_vektoru = model_bert.encode(metin, convert_to_tensor=True)
    cumle_vektorleri = model_bert.encode(cumleler, convert_to_tensor=True)
    
    skorlar = util.cos_sim(cumle_vektorleri, metin_vektoru).cpu().numpy().flatten()
    en_iyi_indeksler = sorted(skorlar.argsort()[-2:][::-1]) # En iyi 2 cümle
    
    return " ".join([cumleler[i] for i in en_iyi_indeksler])

def ana_motor(metin):
    """Gradio için iki yöntemi birleştirir"""
    abs_res = abstractive_ozetle(metin)
    ext_res = extractive_ozetle(metin)
    return abs_res, ext_res

# --- GRADIO ARAYÜZÜ ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {os.path.basename(os.getcwd())} - Metin Özetleme Sistemi")
    gr.Markdown("Bu sistem **mT5 (Abstractive)** ve **BERT (Extractive)** modellerini hibrit olarak kullanır.")
    
    with gr.Row():
        giris = gr.Textbox(lines=10, label="Orijinal Metin", placeholder="Metni buraya yapıştırın...")
    
    with gr.Row():
        cikis_abs = gr.Textbox(label="Tümevarımsal Özet (Sıfırdan Cümle Kurar)")
        cikis_ext = gr.Textbox(label="Çıkarımsal Özet (Önemli Cümleleri Seçer)")
        
    btn = gr.Button("Özetle", variant="primary")
    btn.click(fn=ana_motor, inputs=giris, outputs=[cikis_abs, cikis_ext])

demo.launch(share=True, debug=True)
