import streamlit as st
import torch
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification
from annotated_text import annotated_text # Library tambahan untuk visualisasi entity

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Legal NER Indonesia",
    page_icon="⚖️",
    layout="centered"
)

# --- 1. LOAD MODEL & TOKENIZER ---
@st.cache_resource
def load_model():
    """
    Meload model cahya/bert-base-indonesian-522M yang sudah di-finetune.
    Lokasi folder 'model_output' harus ada di folder yang sama dengan app.py.
    """
    # Load dari folder lokal hasil training [cite: 2, 25]
    model_path = "./model_output" 
    
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_path)
        model = BertForTokenClassification.from_pretrained(model_path)
    except OSError:
        st.error("Folder 'model_output' tidak ditemukan. Pastikan Anda sudah mengupload model ke GitHub.")
        st.stop()
        
    # Gunakan GPU jika tersedia, jika tidak CPU [cite: 426]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, tokenizer, device

model, tokenizer, device = load_model()

# --- 2. DEFINISI LABEL (MAPPING) ---
# Berdasarkan output unique labels di PDF 
# PENTING: Pastikan urutan ID ini SAMA PERSIS dengan 'labels_to_ids' saat training di Colab.
# Anda mungkin perlu menyesuaikan dictionary ini jika hasil prediksi tertukar.
ids_to_labels = {
    0: 'O', 
    1: 'B_ARTV', 2: 'B_CRIA', 3: 'B_DEFN', 4: 'B_JUDG', 
    5: 'B_JUDP', 6: 'B_PENA', 7: 'B_PROS', 8: 'B_PUNI', 
    9: 'B_REG', 10: 'B_TIMV', 11: 'B_VERN', 12: 'B_ADVO',
    # Tambahkan label 'I_' (Intermediate) sesuai urutan training Anda
    13: 'I_ARTV', 14: 'I_CRIA', 15: 'I_DEFN', 16: 'I_JUDG',
    17: 'I_JUDP', 18: 'I_PENA', 19: 'I_PROS', 20: 'I_PUNI',
    21: 'I_REG', 22: 'I_TIMV', 23: 'I_VERN', 24: 'I_ADVO'
}

# --- 3. FUNGSI PREDIKSI ---
def process_text(text):
    """
    Fungsi ini mereplikasi logika 'evaluate_one_text' dan 'align_word_ids' dari PDF[cite: 551, 577].
    """
    # Tokenisasi input
    tokenized_input = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
    
    input_ids = tokenized_input['input_ids'].to(device)
    attention_mask = tokenized_input['attention_mask'].to(device)

    # Prediksi Model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    
    # Ambil prediksi baris pertama (batch size = 1)
    prediction_line = predictions[0].cpu().numpy()
    
    # Mapping Token ke Kata Asli (Word Alignment) [cite: 320, 551]
    word_ids = tokenized_input.word_ids()
    
    result_tokens = []
    result_labels = []
    previous_word_idx = None
    
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    for idx, word_idx in enumerate(word_ids):
        # Skip special tokens ([CLS], [SEP], padding)
        if word_idx is None:
            continue
            
        # Jika ini adalah kata baru (bukan sub-word ##), ambil prediksinya
        if word_idx != previous_word_idx:
            pred_id = prediction_line[idx]
            label = ids_to_labels.get(pred_id, "O")
            
            # Bersihkan token dari karakter ## (jika ada sisa)
            token = input_tokens[idx].replace("##", "")
            
            result_tokens.append(token)
            result_labels.append(label)
        else:
            # Jika ini sub-word (lanjutan kata), kita gabungkan ke token sebelumnya
            # Logic ini untuk visualisasi agar kata tidak terpotong
            token = input_tokens[idx].replace("##", "")
            if result_tokens:
                result_tokens[-1] += token
                
        previous_word_idx = word_idx

    return zip(result_tokens, result_labels)

# --- 4. UI UTAMA (Streamlit) ---
st.title("🇮🇩 Ekstraksi Entitas Hukum (Legal NER)")
st.markdown("""
Aplikasi ini menggunakan model **BERT (Cahya-Base)** yang telah di-finetune untuk mendeteksi entitas hukum 
seperti Nomor Putusan, Nama Hakim, Tuntutan, dll.
""")

st.info("Model dilatih menggunakan dataset Putusan Pengadilan Indonesia.")

# Input Area
input_text = st.text_area(
    "Masukkan Teks Putusan:", 
    height=150,
    placeholder="Contoh: PUTUSAN Nomor 25/Pid.Sus/2022/PN Pwd DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA..."
)

# Tombol Eksekusi
if st.button("Analisis Teks"):
    if not input_text:
        st.warning("Mohon masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Sedang memproses..."):
            results = process_text(input_text)
            
            # Persiapan data untuk visualisasi 'annotated_text'
            visual_output = []
            
            # Warna untuk setiap kategori label
            label_colors = {
                "VERN": "#faa",  # Verdict Number (Merah muda)
                "JUDG": "#afa",  # Judge (Hijau muda)
                "PENA": "#fea",  # Penal (Kuning)
                "TIMV": "#8ef",  # Time (Biru muda)
                "O": None        # Bukan entitas
            }

            for word, label in results:
                # Ambil suffix label (misal B_VERN -> VERN)
                clean_label = label.split("_")[1] if "_" in label else label
                
                if clean_label == "O":
                    visual_output.append(word + " ")
                else:
                    # Tentukan warna, default abu-abu jika tidak ada di dict
                    color = label_colors.get(clean_label, "#ddd")
                    # Tambahkan tuple (kata, label, warna)
                    visual_output.append((word, clean_label, color))
                    visual_output.append(" ")

            st.subheader("Hasil Ekstraksi:")
            st.caption("Entitas yang terdeteksi ditandai dengan warna:")
            
            # Tampilkan hasil visual
            annotated_text(*visual_output)
            
            st.divider()
            
            # Tampilkan JSON mentah jika user butuh data
            with st.expander("Lihat Detail JSON"):
                processed_data = [{"token": w, "label": l} for w, l, c in [x for x in visual_output if isinstance(x, tuple)]]
                st.json(processed_data)

# Footer
st.markdown("---")
st.caption("Dibuat dengan Streamlit • Model based on [cahya/bert-base-indonesian-522M]")
