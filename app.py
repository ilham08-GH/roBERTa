import streamlit as st
import torch
from transformers import RobertaForTokenClassification, RobertaTokenizerFast
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Legal NER Indo - RoBERTa", layout="wide")

st.title("🇮🇩 Legal Document Entity Recognition")
st.markdown("Aplikasi ini menggunakan model **RoBERTa (Cahya)** yang telah di-fine-tune untuk mengekstraksi entitas dari dokumen hukum Indonesia.")

# ---------------------------------------------------------
# 1. DEFINISI LABEL (Sesuai Data User)
# ---------------------------------------------------------
# Mapping Label ke ID (dari input Anda)
labels_to_ids = {
    'B_ADVO': 0, 'B_ARTV': 1, 'B_CRIA': 2, 'B_DEFN': 3, 'B_JUDG': 4, 
    'B_JUDP': 5, 'B_PENA': 6, 'B_PROS': 7, 'B_PUNI': 8, 'B_REGI': 9, 
    'B_TIMV': 10, 'B_VERN': 11, 'I_ADVO': 12, 'I_ARTV': 13, 'I_CRIA': 14, 
    'I_DEFN': 15, 'I_JUDG': 16, 'I_JUDP': 17, 'I_PENA': 18, 'I_PROS': 19, 
    'I_PUNI': 20, 'I_REGI': 21, 'I_TIMV': 22, 'I_VERN': 23, 'O': 24
}

# Membalik Mapping menjadi ID ke Label (Agar bisa dibaca manusia)
ids_to_labels = {v: k for k, v in labels_to_ids.items()}

# ---------------------------------------------------------
# 2. FUNGSI LOAD MODEL & TOKENIZER
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    # Load Tokenizer (Base model)
    tokenizer = RobertaTokenizerFast.from_pretrained("cahya/roberta-base-indonesian-522M")
    
    # Load Model yang sudah di-finetune
    # Pastikan folder 'saved_model' ada di direktori yang sama dengan app.py di GitHub
    try:
        model = RobertaForTokenClassification.from_pretrained("saved_model")
    except:
        st.error("Model fine-tuned tidak ditemukan di folder 'saved_model'. Pastikan Anda mengunggahnya.")
        return None, None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer

model, tokenizer = load_model()

# ---------------------------------------------------------
# 3. FUNGSI PREPROCESSING
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
                label_ids.append(1 if False else -100) 
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx
    return label_ids

# ---------------------------------------------------------
# 4. FUNGSI PEWARNAAN ENTITAS
# ---------------------------------------------------------
def get_label_color(label):
    if "O" in label: return "#ffffff" # Putih
    
    # Kategori Warna
    if "VERN" in label: return "#ffadad"  # Merah Muda (Verdict/Putusan)
    if "PUNI" in label: return "#ffd6a5"  # Oranye (Punishment/Hukuman)
    if "DEFN" in label: return "#fdffb6"  # Kuning (Defendant/Terdakwa)
    if "JUDG" in label: return "#caffbf"  # Hijau (Judge/Hakim)
    if "PROS" in label: return "#9bf6ff"  # Biru Cyan (Prosecutor/Jaksa)
    if "ADVO" in label: return "#a0c4ff"  # Biru Langit (Advokat)
    if "ARTV" in label: return "#bdb2ff"  # Ungu (Article/Pasal)
    if "TIMV" in label: return "#ffc6ff"  # Pink (Time/Waktu)
    if "CRIA" in label: return "#fffffc"  # Cream (Criteria/Kriteria)
    if "REGI" in label: return "#d4d4d4"  # Abu-abu (Register Number)
    
    return "#e5e5e5" # Default Abu-abu muda

# ---------------------------------------------------------
# 5. UI & INFERENSI
# ---------------------------------------------------------

# Input Text
input_text = st.text_area("Masukkan teks putusan/dokumen legal:", height=150, 
                          placeholder="Contoh: MENYATAKAN TERDAKWA BUDI SANTOSO TERBUKTI SECARA SAH...")

if st.button("Analisis Entitas"):
    if input_text and model and tokenizer:
        with st.spinner("Sedang memproses entitas..."):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Tokenisasi
            text_tokenized = tokenizer(input_text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
            
            mask = text_tokenized['attention_mask'].to(device)
            input_id = text_tokenized['input_ids'].to(device)
            
            # Dummy alignment untuk mendapatkan shape logits yang benar
            label_ids_dummy = torch.Tensor(align_word_ids(input_text, tokenizer)).unsqueeze(0).to(device)

            # Prediksi Model
            with torch.no_grad():
                logits = model(input_id, mask, None)
            
            logits_clean = logits[0][label_ids_dummy[0] != -100]
            predictions = logits_clean.argmax(dim=1).tolist()
            
            # Convert ID ke Label Teks
            prediction_label = [ids_to_labels.get(i, "O") for i in predictions]
            
            # --- Visualisasi Hasil ---
            st.subheader("Hasil Ekstraksi:")
            st.info("Arahkan kursor ke teks berwarna untuk melihat label entitas.")
            
            tokens = tokenizer.convert_ids_to_tokens(text_tokenized["input_ids"][0])
            
            # Rekonstruksi kata dan label
            current_word_idx = -1
            word_ids = text_tokenized.word_ids()
            
            words = []
            aligned_labels = []
            
            pred_idx = 0
            for i, word_idx in enumerate(word_ids):
                if word_idx is None: continue
                if word_idx != current_word_idx:
                    # Bersihkan token roberta (karakter Ġ)
                    clean_word = tokens[i].replace('Ġ', '').replace('Ċ', '')
                    words.append(clean_word)
                    
                    if pred_idx < len(prediction_label):
                        aligned_labels.append(prediction_label[pred_idx])
                        pred_idx += 1
                    else:
                        aligned_labels.append("O")
                    current_word_idx = word_idx
            
            # Render HTML
            html_output = ""
            for word, label in zip(words, aligned_labels):
                color = get_label_color(label)
                if label != "O":
                    # Tampilan chip berwarna
                    html_output += f'<span style="background-color: {color}; padding: 2px 6px; border-radius: 5px; margin: 0 2px; border: 1px solid #ccc;" title="{label}"><b>{word}</b> <sub style="font-size:0.6em; color:#444;">{label}</sub></span>'
                else:
                    html_output += f'{word} '
            
            st.markdown(f'<div style="line-height: 2.5; font-family: sans-serif; font-size: 1.1em;">{html_output}</div>', unsafe_allow_html=True)
            
            # Tabel Detail
            with st.expander("Lihat Detail Entitas (Tabel)"):
                entity_data = {"Entitas": [], "Label": []}
                for word, label in zip(words, aligned_labels):
                    if label != "O":
                        entity_data["Entitas"].append(word)
                        entity_data["Label"].append(label)
                st.table(entity_data)

    elif not model:
        st.error("Model error.")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")
