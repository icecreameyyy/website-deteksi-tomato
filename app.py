import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import os
from streamlit_option_menu import option_menu

# === 1. Konfigurasi halaman ===
st.set_page_config(
    page_title="Deteksi Penyakit Tanaman Tomat",
    page_icon="🍅",
    layout="wide" 
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")

# === MENU ===
selected = option_menu(
    menu_title=None,
    options=["Beranda", "Deteksi Tanaman"],
    icons=["house", "camera"],
    orientation="horizontal"
)

# === 2. Gaya CSS ===
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { 
    background-color: #ffe6e6; color: black; 
}
header[data-testid="stHeader"] {
        background-color: #ff4d4d;
}
.judul-tomat {
    font-size: 40px;
    font-weight: bold;
    text-align: left; 
    line-height: 1.2;
    margin-bottom: 20px;
}
@media (max-width: 600px) {
    .judul-tomat {
        font-size: 28px;
        text-align: center; /* Rata tengah di HP */
    }
}
.welcome-box { 
    background-color: #0B6623; 
    color: white; padding: 25px; 
    border-radius: 15px; 
    margin-bottom: 20px;
}
.hasil-box {   
    border-radius: 10px; 
    padding: 10px; 
    text-align: center; 
    font-size: 22px; 
    font-weight: bold; 
    margin: 20px 0; 
}
.sehat-style { 
    background-color: #e6ffe6; 
    color: #008000; 
    border: 2px solid #008000; 
}
.sakit-style { 
    background-color: #fdf2f2; 
    color: #b22222; 
    border: 2px solid #b22222; 
}
.rekomendasi-box { 
    background-color: #fff9e6; 
    border: 2px dashed #ffcc00; 
    border-radius: 10px; 
    padding: 20px; 
    color: #5c4400; 
}
.class-title { 
    color: #ff4b4b; 
    text-align: center; 
    font-weight: bold; 
}
</style>
""", unsafe_allow_html=True)

# == 3.GAMBAR BERANDA ==
CLASS_IMAGES = {
    "Antranoksa": os.path.join(IMG_DIR, "antranoksa.jpg"),
    "Bercak Daun": os.path.join(IMG_DIR, "bercak_daun.jpg"),
    "Busuk Daun": os.path.join(IMG_DIR, "busuk_daun.jpg"),
    "Daun Sehat": os.path.join(IMG_DIR, "sehat.jpg"),
    "Buah Sehat": os.path.join(IMG_DIR, "buah_sehat.jpg"),
}

CLASS_DESCRIPTIONS = {
    "Antranoksa": "Antraknosa adalah penyakit yang disebabkan oleh jamur dari genus Colletotrichum."
                  "Gejalanya berupa bercak cekung berwarna cokelat kehitaman pada buah. Pada tahap lanjut,"
                  "bercak dapat membesar dan menyebabkan jaringan tanaman mati.",

    "Bercak Daun":"Penyakit bercak daun ditandai dengan munculnya bercak-bercak kecil berwarna cokelat atau kehitaman pada permukaan daun."
                  "Seiring waktu, bercak dapat meluas dan menyebabkan daun menguning serta mengering.",

    "Busuk Daun": "Busuk daun merupakan kondisi dimana jaringan daun mengalami pembusukan yang ditandai dengan perubahan warna menjadi cokelat kehitaman,"
                  "tekstur daun menjadi lunak, serta tampak basah atau layu.",

    "Daun Sehat": "Daun tomat yang sehat memiliki daun hijau segar tanpa bercak,"
                  "tidak mengalami penggulungan ataupun perubahan warna."
                  "Pertumbuhan tanaman berjalan normal sehingga mampu menghasilkan buah dengan baik.",

    "Buah Sehat": "Buah tomat yang sehat memiliki bentuk yang normal, warna cerah merata sesuai tingkat kematangan,"
                  "serta tidak terdapat bercak, luka, atau tanda serangan hama dan penyakit. Tekstur buah juga terasa segar dan tidak lembek."
                  "Kondisi ini menunjukkan bahwa tanaman tumbuh dengan baik dan memiliki produktivitas yang optimal."
}

# == 4.HALAMAN BERANDA ==

if selected == "Beranda":
    st.markdown('<div class="judul-tomat">🍎 Deteksi Penyakit Tanaman Tomat</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-box">
        <h3>Selamat Datang Di Website Deteksi Penyakit Tomat</h3>
        <p>Aplikasi ini menggunakan model Convolutional Neural Network (CNN) untuk mendeteksi penyakit pada daun dan buah tomat secara otomatis.</p>
        <p>Terdapat tiga jenis penyakit dan dua kategori tanaman sehat di bawah ini.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Daftar Kelas")
    keys = list(CLASS_DESCRIPTIONS.keys())

    # -- BARIS 1: (3 Penyakit) --
    cols1 = st.columns(3)
    for i in range(3):
        label = keys[i]
        with cols1[i]:
            st.markdown(f"<h3 class='class-title'>{label}</h3>", unsafe_allow_html=True)
            if os.path.exists(CLASS_IMAGES[label]):
                st.image(CLASS_IMAGES[label], use_container_width=True)
            with st.expander("Lihat Deskripsi"):
                st.markdown(f"<div style='text-align: justify;'>{CLASS_DESCRIPTIONS[label]}</div>", unsafe_allow_html=True)

    # -- BARIS 2: (2 Kotak Sehat)  --
    st.markdown("<br>", unsafe_allow_html=True)
    _, col_mid1, col_mid2, _ = st.columns([0.5, 1, 1, 0.5])
    
    with col_mid1:
        label = keys[3]
        st.markdown(f"<h3 class='class-title'>{label}</h3>", unsafe_allow_html=True)
        if os.path.exists(CLASS_IMAGES[label]):
            st.image(CLASS_IMAGES[label], use_container_width=True)
        with st.expander("Lihat Deskripsi"):
            st.markdown(f"<div style='text-align: justify;'>{CLASS_DESCRIPTIONS[label]}</div>", unsafe_allow_html=True)

    with col_mid2:
        label = keys[4]
        st.markdown(f"<h3 class='class-title'>{label}</h3>", unsafe_allow_html=True)
        if os.path.exists(CLASS_IMAGES[label]):
            st.image(CLASS_IMAGES[label], use_container_width=True)
        with st.expander("Lihat Deskripsi"):
            st.markdown(f"<div style='text-align: justify;'>{CLASS_DESCRIPTIONS[label]}</div>", unsafe_allow_html=True)

# === 5.Judul halaman ===
elif selected == "Deteksi Tanaman":
    st.markdown('<div class="judul-tomat">🍅 Deteksi Penyakit Tanaman Tomat</div>', unsafe_allow_html=True)
    
    MODEL_PATH = os.path.join(BASE_DIR, "model_deteksi_tomat_best.h5")
    model = load_model(MODEL_PATH)
    
    class_names = ["antranoksa", "bercak_daun", "busuk_daun", "sehat"]
    st.write("Upload gambar tomat (buah/daun), lalu sistem akan mendeteksi jenis penyakit yang menyerang tanaman tomat serta menampilkan cara penanganannya.")

# === 6.Upload gambar ===
    uploaded_file = st.file_uploader("Klik tombol di bawah atau seret gambar tomat Anda ke sini untuk di deteksi", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        _, col_img, _ = st.columns([1, 4, 1])
        with col_img:
            st.image(img, caption="📸 Gambar yang diunggah", use_container_width=True)

        img_resized = img.resize((150, 150))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("🔍 Sedang memproses gambar..."):
            time.sleep(0.8)
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]
            pretty = predicted_class.replace("_", " ").title()
        
            gaya_kotak = "sehat-style" if predicted_class == "sehat" else "sakit-style"

            # -- 7.TAMPILAN HASIL PREDIKSI --
            st.markdown(f"""       
                <div class='hasil-box {gaya_kotak}'>
                        Hasil Prediksi: {pretty}
                        </div>
                        """, unsafe_allow_html=True)
            
            # --8. REKOMENDASI PENANGANAN --
            if predicted_class == "antranoksa":
                tips = [
                    "Buang dan bakar bagian tanaman yang terinfeksi.",
                    "Gunakan fungisida berbahan aktif tembaga atau azoksistrobin.",
                    "Jaga kelembapan lahan agar tidak terlalu tinggi."
                ]
            elif predicted_class == "bercak_daun":
                tips = [
                    "Pangkas daun yang menunjukkan gejala bercak.",
                    "Semprotkan fungisida seperti Dithane M-45 atau Benlate.",
                    "Gunakan jarak tanam yang cukup untuk sirkulasi udara."
                ]
            elif predicted_class == "busuk_daun":
                tips = [
                    "Hindari penyiraman berlebihan.",
                    "Gunakan fungisida berbahan aktif mancozeb atau metalaksil.",
                    "Rotasi tanaman setiap musim tanam."
                ]
            else: # Sehat
                tips = [
                    "Pertahankan sanitasi lahan yang bersih.",
                    "Berikan nutrisi/pupuk secara teratur sesuai dosis.",
                    "Lakukan pengecekan rutin seminggu dua kali."
                ]

            # --9. TAMPILAN KOTAK REKOMENDASI --
            rekomendasi_html = "".join([f"<li>{item}</li>" for item in tips])
            st.markdown(f"""
            <div class="rekomendasi-box">
                <p>🩺 <b>Rekomendasi Penanganan untuk {pretty}:</b></p>
                <ul style="margin-bottom: 0;">
                    {rekomendasi_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)
