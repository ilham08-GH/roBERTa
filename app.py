import streamlit as st
import torch
from transformers import RobertaForTokenClassification, RobertaTokenizerFast
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Legal NER Indo - RoBERTa", layout="wide")

st.title("🇮🇩 Legal Document Entity Recognition")
st.markdown("Aplikasi ini menggunakan model **RoBERTa (Cahya)** yang telah di-fine-tune untuk mengekstraksi entitas dari dokumen hukum Indonesia.")

# ---------------------------------------------------------
# 1. DEFINISI LABEL (Sesuai training di PDF )
# ---------------------------------------------------------
# PENTING: Salin dictionary 'ids_to_labels' lengkap dari hasil training Anda ke sini.
# Ini hanya contoh berdasarkan potongan di PDF.
ids_to_labels = {
    0: 'O',
    1: 'B_VERN',
    2: 'I_VERN',
    3: 'B_JUDG',
    4: 'I_JUDG',
    5: 'B_PUNI',
    6: 'I_PUNI',
    7: 'B_TIMV',
    8: 'I_TIMV',
    # ... tambahkan label lainnya sesuai hasil training Anda
}

# ---------------------------------------------------------
# 2. FUNGSI LOAD MODEL & TOKENIZER
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    # Load Tokenizer (Base model)
    tokenizer = RobertaTokenizerFast.from_pretrained("cahya/roberta-base-indonesian-522M")
    
    # Load Model yang sudah di-finetune
    # Jika model ada di folder lokal/GitHub, ganti path di bawah ini
    # Contoh: model = RobertaForTokenClassification.from_pretrained("./model_folder")
    try:
        # Placeholder: Asumsi model disimpan di folder 'saved_model' dalam repo
        model = RobertaForTokenClassification.from_pretrained("saved_model")
    except:
        st.error("Model fine-tuned tidak ditemukan. Pastikan Anda mengunggah folder model hasil training.")
        return None, None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer

model, tokenizer = load_model()

# ---------------------------------------------------------
# 3. FUNGSI PREPROCESSING (Dari PDF [cite: 288-313])
# ---------------------------------------------------------
def align_word_ids(texts, tokenizer):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []
    
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if False else -100) # label_all_tokens=False
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx
    return label_ids

# ---------------------------------------------------------
# 4. UI & INFERENSI
# ---------------------------------------------------------

# Input Text
input_text = st.text_area("Masukkan teks putusan/dokumen legal di sini:", height=150, 
                          placeholder="Contoh: MENYATAKAN TERDAKWA AGNES TERBUKTI BERSALAH...")

if st.button("Analisis Entitas"):
    if input_text and model and tokenizer:
        with st.spinner("Sedang memproses..."):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Tokenisasi input [cite: 321]
            text_tokenized = tokenizer(input_text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
            
            mask = text_tokenized['attention_mask'].to(device)
            input_id = text_tokenized['input_ids'].to(device)
            
            # Align word IDs untuk mendapatkan mapping yang benar [cite: 325]
            # Kita perlu dummy alignment untuk mendapatkan logits yang pas
            label_ids_dummy = torch.Tensor(align_word_ids(input_text, tokenizer)).unsqueeze(0).to(device)

            # Prediksi [cite: 330]
            with torch.no_grad():
                logits = model(input_id, mask, None)
            
            logits_clean = logits[0][label_ids_dummy[0] != -100]
            predictions = logits_clean.argmax(dim=1).tolist()
            
            # Mapping ID ke Label [cite: 335]
            prediction_label = [ids_to_labels.get(i, "UNK") for i in predictions]
            
            # --- Menampilkan Hasil ---
            st.subheader("Hasil Ekstraksi:")
            
            # Token asli (rekonstruksi sederhana untuk visualisasi)
            tokens = tokenizer.convert_ids_to_tokens(text_tokenized["input_ids"][0])
            # Filter token spesial (<s>, <pad>, dll)
            valid_tokens = []
            final_labels = []
            
            # Logika sederhana untuk menyatukan token dan label
            # (Perlu penyesuaian tergantung bagaimana Anda ingin menampilkannya)
            current_word_idx = -1
            word_ids = text_tokenized.word_ids()
            
            # Reconstruct words from tokens
            words = []
            aligned_labels = []
            
            pred_idx = 0
            for i, word_idx in enumerate(word_ids):
                if word_idx is None: continue
                if word_idx != current_word_idx:
                    words.append(tokens[i].replace('Ġ', '')) # Hapus prefix RoBERTa
                    if pred_idx < len(prediction_label):
                        aligned_labels.append(prediction_label[pred_idx])
                        pred_idx += 1
                    else:
                        aligned_labels.append("O")
                    current_word_idx = word_idx
            
            # Tampilkan dalam format warna-warni
            # Menggabungkan token dan label menjadi HTML string
            html_output = ""
            for word, label in zip(words, aligned_labels):
                color = "#ffffff" # Default putih
                if label != "O":
                    if "VERN" in label: color = "#ffcccb" # Merah muda
                    elif "JUDG" in label: color = "#add8e6" # Biru muda
                    elif "PUNI" in label: color = "#90ee90" # Hijau muda
                    # Tambahkan warna lain
                    
                    html_output += f'<span style="background-color: {color}; padding: 2px; border-radius: 3px; margin-right: 3px;" title="{label}">{word} <small style="font-size:0.6em">[{label}]</small></span> '
                else:
                    html_output += f'{word} '
            
            st.markdown(f'<div style="line-height: 2.0; font-family: sans-serif;">{html_output}</div>', unsafe_allow_html=True)
            
            with st.expander("Lihat Raw Data"):
                st.write({"Token": words, "Label": aligned_labels})
                
    elif not model:
        st.error("Model belum dimuat. Cek path model Anda.")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")
