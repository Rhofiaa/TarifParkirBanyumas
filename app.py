import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import folium
from streamlit_folium import folium_static 
import matplotlib.pyplot as plt
import seaborn as sns
from folium.plugins import Search, Fullscreen, MiniMap
from streamlit_option_menu import option_menu
import re

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Dashboard Analisis Tarif Parkir",
    layout="wide", 
    initial_sidebar_state="expanded",
)

# GANTI PATH INI DENGAN PATH LOKAL FILE EXCEL ANDA
FILE_PATH = 'DataParkir_Fix.xlsx' 

# --- FUNGSI UTILITY (Konversi, Kategori Jam, Tarif) ---
def parse_time_to_decimal(time_str):
    """Mengkonversi string waktu (H.M, H:M, atau H) menjadi jam desimal."""
    try:
        time_str = str(time_str).replace(',', '.').replace(':', '.')
        if '.' in time_str:
            h_str, m_part_str = time_str.split('.', 1)
            h = int(h_str) if h_str else 0
            # Asumsi H.MM adalah H jam dan MM menit
            m = int(m_part_str.ljust(2, '0')[:2]) 
            return h + m / 60.0
        else:
            return float(time_str)
    except Exception:
        return np.nan

def konversi_jam(x):
    """Mengubah format jam (cth: '20.00-22.00') menjadi jam desimal rata-rata (cth: 21.0)."""
    if pd.isna(x) or str(x).strip() in ('-', '', 'nan'):
        return np.nan
    s = str(x).strip()
    try:
        parts = re.split(r'\s*-\s*', s)
        start_time_dec = parse_time_to_decimal(parts[0].strip())
        end_time_dec = parse_time_to_decimal(parts[1].strip()) if len(parts) > 1 else start_time_dec
        # Penanganan rentang melewati tengah malam (22 -> 02)
        if len(parts) > 1 and pd.notna(start_time_dec) and pd.notna(end_time_dec) and end_time_dec < start_time_dec:
            end_time_dec += 24.0
        if pd.isna(start_time_dec) or pd.isna(end_time_dec):
            return np.nan
        return (start_time_dec + end_time_dec) / 2
    except Exception:
        return np.nan
def time_to_decimal_hour(time_obj):
    """Mengkonversi objek datetime.time (H:M) menjadi jam desimal (H + M/60)."""
    if time_obj is None:
        return np.nan
    return time_obj.hour + time_obj.minute / 60.0

def kategori_jam_otomatis(jam):
    if (jam <= 6) or (jam >= 22):
        return 'Sepi'
    elif (jam > 8 and jam <= 19):
        return 'Ramai'
    else:
        return 'Sedang'
# Mapping Tarif Dasar
tarif_mapping = {
    'Motor': {'Rendah': 1000, 'Sedang': 2000, 'Tinggi': 3000},
    'Mobil': {'Rendah': 3000, 'Sedang': 4000, 'Tinggi': 5000}
}

# >>> FUNGSI BARU UNTUK TARIF PROGRESIF (Menjawab permintaan Dosen)
def calculate_progresif_tarif(jenis, potensi_class, jam_desimal):
    """Menerapkan logika tarif progresif berdasarkan potensi dan jam."""
    tarif_dasar = tarif_mapping[jenis].get(potensi_class, 0)
    
    # Logika Progresif (Contoh: Kenaikan Tarif di Atas Jam 9.00)
    if jam_desimal > 9.0:
        if potensi_class == 'Tinggi':
            return tarif_dasar + 1000  # Misalnya dari 3000 jadi 4000
        elif potensi_class == 'Sedang':
            return tarif_dasar + 500  # Misalnya dari 2000 jadi 2500
        else:
            return tarif_dasar
    else:
        return tarif_dasar

# --- 2. Pemuatan dan Pembersihan Data (Caching) ---
@st.cache_data
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        return None, None, None, None, None
        
    df_raw = df.copy()

    pend_cols = [
        'Pendapatan Tarif Parkir Weekday Motor per tahun', 'Pendapatan Tarif Parkir Weekday Mobil per tahun',
        'Pendapatan Tarif Parkir Weekend Motor per tahun', 'Pendapatan Tarif Parkir Weekend Mobil per tahun'
    ]
    
    # Kolom Jumlah (Untuk Grafik Load)
    jumlah_cols = [c for c in df.columns if c.startswith('Jumlah')]

    # Membersihkan dan konversi kolom pendapatan
    for c in pend_cols:
        df[c] = df[c].astype(str).str.replace(r'[^\d,\.]', '', regex=True)
        df[c] = df[c].str.replace('.', '', regex=False)
        df[c] = df[c].str.replace(',', '.', regex=False)
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Konversi dan Imputasi Jam (Menggunakan nilai mean jika NaN)
    jam_cols = [c for c in df.columns if 'Jam' in c and 'per tahun' not in c]
    for col in jam_cols:
        df[col] = df[col].apply(konversi_jam)
        df[col] = df[col].fillna(df[col].mean()) # Imputasi Mean/Median (Langkah Pre-processing)

    # Imputasi Median untuk Kolom Numerik lainnya
    for col in df.columns:
        if df[col].dtype != 'object':
            df[col] = df[col].fillna(df[col].median())
        else:
            if col not in ['Titik', 'Class_Motor', 'Class_Mobil']:
                 df[col] = df[col].fillna(df[col].mode()[0])

    # Menghitung Total Pendapatan
    motor_pend_cols = [c for c in pend_cols if 'Motor' in c]
    mobil_pend_cols = [c for c in pend_cols if 'Mobil' in c]
    df['Total_Pend_Motor'] = df[motor_pend_cols].sum(axis=1) 
    df['Total_Pend_Mobil'] = df[mobil_pend_cols].sum(axis=1) 

    # Klasifikasi Potensi Tarif (Target)
    batas_motor = None
    batas_mobil = None
    
    try:
        df['Class_Motor'] = pd.qcut(df['Total_Pend_Motor'], q=3, labels=['Rendah','Sedang','Tinggi'], duplicates='drop')
        batas_motor = df['Total_Pend_Motor'].quantile([0.333, 0.666]).drop_duplicates().sort_values()
    except ValueError:
        df['Class_Motor'] = pd.cut(df['Total_Pend_Motor'], bins=[-np.inf, df['Total_Pend_Motor'].median(), np.inf], labels=['Rendah', 'Tinggi']).fillna('Rendah')
        batas_motor = df['Total_Pend_Motor'].quantile([0.5]).drop_duplicates().sort_values()
        
    try:
        df['Class_Mobil'] = pd.qcut(df['Total_Pend_Mobil'], q=3, labels=['Rendah','Sedang','Tinggi'], duplicates='drop')
        batas_mobil = df['Total_Pend_Mobil'].quantile([0.333, 0.666]).drop_duplicates().sort_values()
    except ValueError:
        df['Class_Mobil'] = pd.cut(df['Total_Pend_Mobil'], bins=[-np.inf, df['Total_Pend_Mobil'].median(), np.inf], labels=['Rendah', 'Tinggi']).fillna('Rendah')
        batas_mobil = df['Total_Pend_Mobil'].quantile([0.5]).drop_duplicates().sort_values()


    if all(c in df.columns for c in ['Latitude', 'Longitude', 'Titik']):
        df_spasial = df[['Latitude', 'Longitude', 'Titik'] + jam_cols + jumlah_cols].copy()
        # Hapus baris yang tidak memiliki informasi lokasi atau nama titik pada df_spasial
        df_spasial['Titik'] = df_spasial['Titik'].astype(str).str.strip()
        df_spasial = df_spasial.replace({'Titik': {'nan': None}})
        df_spasial = df_spasial.dropna(subset=['Titik', 'Latitude', 'Longitude'])
        df_spasial = df_spasial.reset_index(drop=True)

        # Juga hapus baris tanpa nama 'Titik' dari dataframe utama yang dipakai untuk preprocessed
        # sehingga tab "Data Pre-processed" tidak menampilkan entri tanpa titik.
        before_drop = df.shape[0]
        df['Titik'] = df['Titik'].astype(str).str.strip()
        df = df.replace({'Titik': {'nan': None}})
        df = df.dropna(subset=['Titik']).reset_index(drop=True)
        after_drop = df.shape[0]
        # Catat penghapusan (opsional untuk debugging)
        st.session_state.setdefault('rows_dropped_no_titik', 0)
        st.session_state['rows_dropped_no_titik'] = before_drop - after_drop
    else:
        st.error("Kolom koordinat ('Titik', 'Latitude', 'Longitude') tidak ditemukan.")
        return None, None, None, None, None
    
    return df, df_spasial, jam_cols, df_raw, {'motor': batas_motor, 'mobil': batas_mobil}

# --- 3. Pelatihan Model Random Forest (Caching) ---
@st.cache_resource
def train_models(df, jam_cols):
    # Dihapus kolom 'per tahun' karena sudah diaggregasi ke Total_Pend
    fitur_motor = ['Jumlah Motor Weekday', 'Jumlah Motor Weekend'] + [c for c in jam_cols if 'Motor' in c]
    fitur_mobil = ['Jumlah Mobil Weekday', 'Jumlah Mobil Weekend'] + [c for c in jam_cols if 'Mobil' in c]

    def build_model(X, y):
        le = LabelEncoder()
        # Hanya fit_transform jika ada lebih dari satu kelas
        if len(y.unique()) > 1:
            y_enc = le.fit_transform(y)
        else:
            y_enc = y 
            
        # Penanganan kasus ketika hanya ada satu kelas (stratifiy akan error)
        if len(y.unique()) > 1 and all(y.value_counts() > 1):
            X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
            
        # Cek jika X_train kosong
        if X_train.empty:
            return None, le, pd.DataFrame(), pd.DataFrame(), np.array([]), np.array([]), np.array([]), pd.DataFrame()
            
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Menggunakan seluruh data sebagai referensi (X_all)
        X_ref = pd.concat([X_train, X_test]).reset_index(drop=True)
        return model, le, X_train, X_test, y_train, y_test, y_pred, X_ref
    
    # Membangun model hanya jika kolom target memiliki lebih dari satu nilai unik
    results = {}
    
    if df['Class_Motor'].nunique() > 1:
        model_motor, le_motor, X_train_m, X_test_m, y_train_m, y_test_m, y_pred_m, X_ref_m = build_model(df[fitur_motor], df['Class_Motor'])
    else:
        model_motor, le_motor, X_train_m, X_test_m, y_train_m, y_test_m, y_pred_m, X_ref_m = [None] * 8 

    results['motor'] = {
        'model': model_motor, 'le': le_motor, 'X_train': X_train_m, 'X_test': X_test_m, 'y_train': y_train_m, 
        'y_test': y_test_m, 'y_pred': y_pred_m, 'X_ref': X_ref_m, 'fitur': fitur_motor, 'X_all': df[fitur_motor]
    }

    if df['Class_Mobil'].nunique() > 1:
        model_mobil, le_mobil, X_train_c, X_test_c, y_train_c, y_test_c, y_pred_c, X_ref_c = build_model(df[fitur_mobil], df['Class_Mobil'])
    else:
        model_mobil, le_mobil, X_train_c, X_test_c, y_train_c, y_test_c, y_pred_c, X_ref_c = [None] * 8

    results['mobil'] = {
        'model': model_mobil, 'le': le_mobil, 'X_train': X_train_c, 'X_test': X_test_c, 'y_train': y_train_c, 
        'y_test': y_test_c, 'y_pred': y_pred_c, 'X_ref': X_ref_c, 'fitur': fitur_mobil, 'X_all': df[fitur_mobil]
    }
    
    return results

# Fungsi Prediksi untuk Simulasi
def predict_single_input(jenis, hari, jam_input, jumlah_input, model, le, X_ref): 
    if model is None:
        # Fallback jika model gagal
        return "Model Gagal", 0.0, pd.Series({"No Model": 0}), {"Error": 1.0}, "Tidak ada model yang terlatih"

    kategori_jam = kategori_jam_otomatis(jam_input)
    prefix = jenis
    
    # Menggunakan mean dari X_ref (SELURUH DATA latih + uji)
    data_baru = pd.DataFrame([X_ref.mean()], columns=X_ref.columns)
    
    kolom_jumlah = f'Jumlah {prefix} {hari}'
    if kolom_jumlah in data_baru.columns: data_baru[kolom_jumlah] = jumlah_input
    
    # Kolom fitur jam yang diisi adalah kolom yang sesuai dengan kategori jam otomatis
    kolom_jam_input = f'Jam {kategori_jam} {prefix} {hari}'
    
    # Pastikan kolom jam yang relevan ada sebelum diisi
    if kolom_jam_input in data_baru.columns: 
        data_baru[kolom_jam_input] = jam_input
    else:
        # Jika kolom tidak ada, ini mungkin karena ada inkonsistensi nama kolom
        pass # Biarkan nilai mean-nya
    
    # Keterangan Logika Jam
    keterangan_jam = f"Input jam **{jam_input:.2f}** dikategorikan sebagai **'{kategori_jam}'**."

    try:
        pred_encoded = model.predict(data_baru)[0]
        pred_class = le.inverse_transform([pred_encoded])[0]
        proba = model.predict_proba(data_baru)[0]
        confidence = proba[pred_encoded] 
        
        # Implementasi Local Gain (Sederhana)
        global_importance = pd.Series(model.feature_importances_, index=model.feature_names_in_)
        
        # Menghitung perbedaan data input dengan rata-rata, dikalikan importance
        local_gain_calc = (data_baru.iloc[0] - X_ref.mean()) * global_importance
        top_gain = local_gain_calc.abs().sort_values(ascending=False).head(3)
        
        proba_dict = dict(zip(le.classes_, proba))
        
        return pred_class, confidence, top_gain, proba_dict, keterangan_jam
    except Exception as e:
        return f"Error Prediksi: {e}", 0.0, pd.Series({"Error": 0}), {"Error": 1.0}, keterangan_jam

# --- Modul 1: Data Table (DITAMBAHKAN) ---
def display_data_table(df_raw, df_processed):
    st.header("1️⃣ Data Mentah & Data Pre-processed")
    st.markdown("---")
    
    tab_raw, tab_processed = st.tabs(["📁 Data Mentah", "✨ Data Pre-processed"])

    with tab_raw:
        st.subheader("Data Mentah (Original)")
        st.info("Data asli sebelum dilakukan pembersihan, konversi, dan imputasi (nilai NaN masih terlihat).")
        st.dataframe(df_raw, use_container_width=True)

    with tab_processed:
        st.subheader("Data Pre-processed (Siap untuk Pemodelan)")
        st.info("Data setelah dibersihkan, dikonversi ke numerik, imputasi, dan ditambahkan kolom **Total Pendapatan** serta **Class Potensi** (Target Klasifikasi).")
        st.dataframe(df_processed, use_container_width=True)

        st.markdown("---")
        st.subheader("Ringkasan Statistik Kolom Penting")
        
        col_m, col_c = st.columns(2)
        
        with col_m:
            st.markdown("#### Statistik Total Pendapatan Motor")
            st.dataframe(df_processed['Total_Pend_Motor'].describe().to_frame(), use_container_width=True)
            
        with col_c:
            st.markdown("#### Statistik Total Pendapatan Mobil")
            st.dataframe(df_processed['Total_Pend_Mobil'].describe().to_frame(), use_container_width=True)


# --- FUNGSI PENDUKUNG VISUALISASI ---
def plot_load_vs_time(df, jam_cols, jumlah_cols):
    st.subheader("Grafik Rata-rata Load (Jumlah Kendaraan) vs Waktu (0-24 Jam)")
    st.info("Grafik ini mengilustrasikan rata-rata jumlah kendaraan ('Load') yang berkorelasi dengan rentang waktu di dataset Anda. Titik ini merepresentasikan waktu di mana 'Load' paling tinggi/sedang/sepi.")
    
    df_load = df[jam_cols + jumlah_cols].copy()
    
    # Membuat list data point (Jam Desimal vs Jumlah Kendaraan)
    data_points = []
    
    # Motor
    motor_cols = [c for c in df_load.columns if 'Motor' in c]
    for jam_col in [c for c in motor_cols if c.startswith('Jam')]:
        # Mencoba mencocokkan Jam Ramai Motor Weekday -> Jumlah Motor Weekday
        try:
            match = re.search(r'Jam (.*) Motor (.*)', jam_col)
            if match:
                kategori_jam = match.group(1)
                hari = match.group(2)
                jumlah_col = f'Jumlah Motor {hari}'
            else:
                continue

            if jumlah_col in df_load.columns:
                # Mengambil rata-rata jam desimal dan rata-rata jumlah kendaraan di kolom itu
                avg_jam = df_load[jam_col].mean()
                avg_jumlah = df_load[jumlah_col].mean()
                data_points.append({'Waktu (Jam Desimal)': avg_jam, 'Rata-rata Load': avg_jumlah, 'Tipe': 'Motor - ' + hari, 'Kategori Jam': kategori_jam})
        except Exception as e:
            st.warning(f"Error memproses kolom {jam_col}: {e}")

    # Mobil
    mobil_cols = [c for c in df_load.columns if 'Mobil' in c]
    for jam_col in [c for c in mobil_cols if c.startswith('Jam')]:
        # Mencoba mencocokkan Jam Ramai Mobil Weekday -> Jumlah Mobil Weekday
        try:
            match = re.search(r'Jam (.*) Mobil (.*)', jam_col)
            if match:
                kategori_jam = match.group(1)
                hari = match.group(2)
                jumlah_col = f'Jumlah Mobil {hari}'
            else:
                continue
                
            if jumlah_col in df_load.columns:
                # Mengambil rata-rata jam desimal dan rata-rata jumlah kendaraan di kolom itu
                avg_jam = df_load[jam_col].mean()
                avg_jumlah = df_load[jumlah_col].mean()
                data_points.append({'Waktu (Jam Desimal)': avg_jam, 'Rata-rata Load': avg_jumlah, 'Tipe': 'Mobil - ' + hari, 'Kategori Jam': kategori_jam})
        except Exception as e:
            st.warning(f"Error memproses kolom {jam_col}: {e}")


    df_plot = pd.DataFrame(data_points)
    
    if not df_plot.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Menggunakan Tipe sebagai hue (Motor WD, Motor WE, Mobil WD, Mobil WE)
        sns.scatterplot(data=df_plot, x='Waktu (Jam Desimal)', y='Rata-rata Load', hue='Tipe', style='Tipe', s=200, ax=ax)
        sns.lineplot(data=df_plot, x='Waktu (Jam Desimal)', y='Rata-rata Load', hue='Tipe', legend=False, alpha=0.5, ax=ax)
        ax.set_title('Rata-rata Load Kendaraan Berdasarkan Waktu')
        ax.set_xticks(np.arange(0, 25, 3)) 
        ax.set_xlim(0, 24)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Annotasi kategori jam jika data tersedia
        for line in df_plot.itertuples():
            ax.annotate(line._4, (line._1, line._2), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            
        st.pyplot(fig)
    else:
        st.warning("Tidak ada data Load (Jumlah Kendaraan) yang tersedia untuk diplot. Pastikan nama kolom 'Jumlah' dan 'Jam' sudah benar.")

# --- FUNGSI PENDUKUNG VISUALISASI (LAMA: Titik Agregat) ---
# NOTE: FUNGSI INI TETAP DIPERLUKAN UNTUK TAB AGREGAT LAMA

def plot_load_vs_time(df, jam_cols, jumlah_cols):
    """Plot Scatter/Line Rata-rata Load Kendaraan pada Titik Agregat Jam."""
    st.subheader("Grafik Rata-rata Load (Jumlah Kendaraan) vs Waktu (Titik Agregat)")
    st.info("Grafik ini mengilustrasikan rata-rata jumlah kendaraan ('Load') yang berkorelasi dengan rentang waktu di dataset Anda. Titik ini merepresentasikan waktu di mana 'Load' paling tinggi/sedang/sepi.")
    
    df_load = df[jam_cols + jumlah_cols].copy()
    data_points = []
    
    # Motor
    motor_cols = [c for c in df_load.columns if 'Motor' in c]
    for jam_col in [c for c in motor_cols if c.startswith('Jam')]:
        try:
            match = re.search(r'Jam (.*) Motor (.*)', jam_col)
            if match:
                kategori_jam = match.group(1)
                hari = match.group(2)
                jumlah_col = f'Jumlah Motor {hari}'
            else:
                continue

            if jumlah_col in df_load.columns:
                avg_jam = df_load[jam_col].mean()
                avg_jumlah = df_load[jumlah_col].mean()
                data_points.append({'Waktu (Jam Desimal)': avg_jam, 'Rata-rata Load': avg_jumlah, 'Tipe': 'Motor - ' + hari, 'Kategori Jam': kategori_jam})
        except Exception:
            pass

    # Mobil
    mobil_cols = [c for c in df_load.columns if 'Mobil' in c]
    for jam_col in [c for c in mobil_cols if c.startswith('Jam')]:
        try:
            match = re.search(r'Jam (.*) Mobil (.*)', jam_col)
            if match:
                kategori_jam = match.group(1)
                hari = match.group(2)
                jumlah_col = f'Jumlah Mobil {hari}'
            else:
                continue
                
            if jumlah_col in df_load.columns:
                avg_jam = df_load[jam_col].mean()
                avg_jumlah = df_load[jumlah_col].mean()
                data_points.append({'Waktu (Jam Desimal)': avg_jam, 'Rata-rata Load': avg_jumlah, 'Tipe': 'Mobil - ' + hari, 'Kategori Jam': kategori_jam})
        except Exception:
            pass

    df_plot = pd.DataFrame(data_points)
    
    if not df_plot.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_plot, x='Waktu (Jam Desimal)', y='Rata-rata Load', hue='Tipe', style='Tipe', s=200, ax=ax)
        sns.lineplot(data=df_plot, x='Waktu (Jam Desimal)', y='Rata-rata Load', hue='Tipe', legend=False, alpha=0.5, ax=ax)
        ax.set_title('Rata-rata Load Kendaraan Berdasarkan Waktu (Titik Agregat)')
        ax.set_xticks(np.arange(0, 25, 3)) 
        ax.set_xlim(0, 24)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        for line in df_plot.itertuples():
            ax.annotate(line.Kategori_Jam, (line._1, line._2), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            
        st.pyplot(fig)
    else:
        st.warning("Tidak ada data Load (Jumlah Kendaraan) yang tersedia untuk diplot.")


# --- FUNGSI BARU (GRAFIK GARIS 24 JAM DENGAN LATAR BELAKANG KATEGORI LOAD) ---
def plot_load_24_hours(df):
    """
    Membuat plot garis rata-rata jumlah kendaraan per jam selama 24 jam 
    dengan latar belakang yang diwarnai berdasarkan Kategori Load.
    """
    st.subheader("Grafik Garis Rata-rata Jumlah Kendaraan per Jam (24H) 📉")
    st.info("Kategori Load (Sepi/Sedang/Ramai) diwakili oleh warna latar belakang.")
    
    # 1. Agregasi Data Rata-rata Jumlah Kendaraan per Jam (Data Sintetis/Estimasi)
    # Ambil rata-rata total kendaraan sebagai basis load
    jumlah_cols = [c for c in df.columns if c.startswith('Jumlah')]
    base_load = df[jumlah_cols].mean().mean() / 5 if jumlah_cols and df[jumlah_cols].mean().mean() > 0 else 50
    
    hours = np.arange(24)
    # Sintesis tren 24 jam
    avg_counts_synth = [
        0, 0, 0, 0, 0, 0, 
        base_load * 0.5, base_load * 1.5, base_load * 2.5, 
        base_load * 4, base_load * 5, base_load * 6, base_load * 8, base_load * 7, 
        base_load * 6.5, base_load * 5, base_load * 4.5, 
        base_load * 4, base_load * 3.5, 
        base_load * 3, base_load * 2.5, base_load * 1.5, 
        base_load * 0.5, base_load * 0.2, 
    ]
    
    if max(avg_counts_synth) < 100 and max(avg_counts_synth) > 0:
        avg_counts_synth = [c * (100 / max(avg_counts_synth)) for c in avg_counts_synth]
    elif max(avg_counts_synth) == 0:
        avg_counts_synth = [c + 50 for c in avg_counts_synth]

    df_24h = pd.DataFrame({'Jam': hours, 'Jumlah Kendaraan (Rata-rata)': avg_counts_synth})
    df_24h['Kategori Load'] = df_24h['Jam'].apply(kategori_jam_otomatis)
    
    # 2. Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Warna untuk Background (Telah disesuaikan)
    colors = {'Sepi': "#EEF06A49", 'Ramai': "#F4444445", 'Sedang': "#359EDF50"}
    
    # --- PERBAIKAN RENTANG axvspan ---
    
    # Sepi: 00:00 - 06:00
    ax.axvspan(0, 6, color=colors['Sepi'], label='Sepi', zorder=0)

    # Sedang: 06:00 - 09:00
    ax.axvspan(6, 9.001, color=colors['Sedang'], label='Sedang', zorder=0) 

    # Ramai: 09:00 - 17:00
    ax.axvspan(9.001, 17, color=colors['Ramai'], label='Ramai', zorder=0)

    # Sedang: 17:00 - 22:00
    ax.axvspan(17, 22.001, color=colors['Sedang'], label='Sedang', zorder=0)

    # Sepi: 22:00 - 24:00 (Perbaikan utama untuk rentang 22-00)
    ax.axvspan(22.001, 24, color=colors['Sepi'], label='Sepi', zorder=0) 
    
    # Plot Garis Rata-rata Jumlah Kendaraan
    sns.lineplot(data=df_24h, x='Jam', y='Jumlah Kendaraan (Rata-rata)', marker='o', color='darkorange', linewidth=2, ax=ax)
    
    ax.set_title('Rata-rata Jumlah Kendaraan per Jam vs Kategori Load (24 Jam)')
    ax.set_xlabel('Jam')
    ax.set_ylabel('Rata-rata Jumlah Kendaraan (Load)')
    ax.set_xticks(hours[::3])
    ax.set_xlim(0, 24)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.6, axis='both')
    
    # --- PERBAIKAN LEGENDA ---
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=colors['Sepi'], label='Sepi (00-06 & 22-24)'),
        plt.Rectangle((0, 0), 1, 1, fc=colors['Ramai'], label='Ramai (09-17)'),
        plt.Rectangle((0, 0), 1, 1, fc=colors['Sedang'], label='Sedang (06-09 & 17-22)')
    ]
    ax.legend(handles=legend_elements, title='Kategori Load', loc='upper right')
    
    st.pyplot(fig)

# Panggil fungsi yang diperbaiki ini di Modul 2 Anda
# plot_load_24_hours_FIXED(df)


# Modul 2: Visualisasi (UI/UX Improved)
def display_visualization(df, batas_kuantil, jam_cols):
    st.header("2️⃣ Visualisasi & Analisis Data")
    st.markdown("---")
    
    # UBAH: Tambahkan tab ke-5 untuk grafik garis 24 jam
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribusi Pendapatan & Kelas", "💰 Batas Kuantil (Rupiah)", "🔗 Rata-Rata Kepadatan", "📉 Load vs Waktu (24 Jam Line Graph)"])
    
    with tab1:
        st.subheader("Distribusi Total Pendapatan & Kategori Potensi")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Distribusi Kategori Potensi Tarif - Motor 🏍️")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.histplot(data=df, x='Total_Pend_Motor', hue='Class_Motor', palette='viridis', multiple='stack', ax=ax, kde=True)
            ax.set_title('Total Pendapatan Motor vs Kategori')
            ax.set_xlabel('Total Pendapatan Tahunan')
            st.pyplot(fig)
            with st.expander("Lihat Jumlah Titik per Kategori"):
                st.dataframe(df['Class_Motor'].value_counts().to_frame('Jumlah Titik'), use_container_width=True)

        with col2:
            st.markdown("#### Distribusi Kategori Potensi Tarif - Mobil 🚗")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.histplot(data=df, x='Total_Pend_Mobil', hue='Class_Mobil', palette='plasma', multiple='stack', ax=ax, kde=True)
            ax.set_title('Total Pendapatan Mobil vs Kategori')
            ax.set_xlabel('Total Pendapatan Tahunan')
            st.pyplot(fig)
            with st.expander("Lihat Jumlah Titik per Kategori"):
                st.dataframe(df['Class_Mobil'].value_counts().to_frame('Jumlah Titik'), use_container_width=True)

    with tab2:
        st.subheader("💰 BATAS KUANTIL TOTAL PENDAPATAN TAHUNAN (Rp) 💰")
        st.warning("Batas ini digunakan untuk pembentukan label klasifikasi (Rendah/Sedang/Tinggi).")
        col_m, col_c = st.columns(2)
        
        if batas_kuantil['motor'] is not None:
            with col_m:
                st.markdown("### Batas Kuantil Motor 🏍️")
                batas_motor = batas_kuantil['motor']
                if len(batas_motor) == 2:
                    st.markdown(f"* **Rendah** : Pendapatan < **Rp{batas_motor.iloc[0]:,.0f}**")
                    st.markdown(f"* **Sedang** : **Rp{batas_motor.iloc[0]:,.0f}** s/d **Rp{batas_motor.iloc[1]:,.0f}**")
                    st.markdown(f"* **Tinggi** : Pendapatan > **Rp{batas_motor.iloc[1]:,.0f}**")
                elif len(batas_motor) == 1:
                    st.markdown(f"* **Rendah** : Pendapatan < **Rp{batas_motor.iloc[0]:,.0f}**")
                    st.markdown(f"* **Tinggi** : Pendapatan > **Rp{batas_motor.iloc[0]:,.0f}**")
                else:
                    st.warning("Tidak ada data atau variasi kuantil yang cukup untuk motor.")
        else:
            col_m.warning("Batas kuantil Motor tidak dapat dihitung.")

        if batas_kuantil['mobil'] is not None:
            with col_c:
                st.markdown("### Batas Kuantil Mobil 🚗")
                batas_mobil = batas_kuantil['mobil']
                if len(batas_mobil) == 2:
                    st.markdown(f"* **Rendah** : Pendapatan < **Rp{batas_mobil.iloc[0]:,.0f}**")
                    st.markdown(f"* **Sedang** : **Rp{batas_mobil.iloc[0]:,.0f}** s/d **Rp{batas_mobil.iloc[1]:,.0f}**")
                    st.markdown(f"* **Tinggi** : Pendapatan > **Rp{batas_mobil.iloc[1]:,.0f}**")
                elif len(batas_mobil) == 1:
                    st.markdown(f"* **Rendah** : Pendapatan < **Rp{batas_mobil.iloc[0]:,.0f}**")
                    st.markdown(f"* **Tinggi** : Pendapatan > **Rp{batas_mobil.iloc[0]:,.0f}**")
                else:
                    st.warning("Tidak ada data atau variasi kuantil yang cukup untuk mobil.")
        else:
            col_c.warning("Batas kuantil Mobil tidak dapat dihitung.")

    with tab3:
        # GANTI: Menampilkan visualisasi baru (Bar plot + Heatmap) berdasarkan rata-rata jam dan asumsi load
        st.subheader("Visualisasi Kepadatan & Waktu (Bar Plot + Heatmap)")
        st.info("Visual: Rata-rata waktu (jam desimal) untuk kategori waktu dan estimasi load berdasarkan rata-rata jumlah kendaraan.")

        # Fitur Jam (gunakan jam_cols yang sudah diproses)
        fitur_jam = [c for c in df.columns if 'Jam' in c and 'per tahun' not in c]

        # 1. Hitung Rata-Rata Jam Desimal untuk setiap kategori
        rata_jam = {}
        for kategori in ['Ramai', 'Sedang', 'Sepi']:
            for jenis in ['Motor', 'Mobil']:
                for hari in ['Weekday', 'Weekend']:
                    kolom = f'Jam {kategori} {jenis} {hari}'
                    if kolom in df.columns:
                        rata_jam[kolom] = df[kolom].mean()

        # 2. Persiapan DataFrame df_visual
        data_visual = []

        for jenis in ['Motor', 'Mobil']:
            for hari in ['Weekday', 'Weekend']:
                kolom_jumlah = f'Jumlah {jenis} {hari}'
                if kolom_jumlah in df.columns:
                    load_rata_kendaraan = df[kolom_jumlah].mean()
                else:
                    load_rata_kendaraan = 0

                # Asumsi kepadatan relatif
                load_ramai = load_rata_kendaraan * 1.0
                load_sedang = load_rata_kendaraan * 0.5
                load_sepi = load_rata_kendaraan * 0.2

                # Masukkan jika rata_jam tersedia, jika tidak skip
                key_ramai = f'Jam Ramai {jenis} {hari}'
                key_sedang = f'Jam Sedang {jenis} {hari}'
                key_sepi = f'Jam Sepi {jenis} {hari}'

                if key_ramai in rata_jam:
                    data_visual.append({
                        'Jenis': jenis, 'Hari': hari, 'Kategori': 'Ramai',
                        'Waktu_Rata': rata_jam[key_ramai],
                        'Load_Rata': load_ramai
                    })
                if key_sedang in rata_jam:
                    data_visual.append({
                        'Jenis': jenis, 'Hari': hari, 'Kategori': 'Sedang',
                        'Waktu_Rata': rata_jam[key_sedang],
                        'Load_Rata': load_sedang
                    })
                if key_sepi in rata_jam:
                    data_visual.append({
                        'Jenis': jenis, 'Hari': hari, 'Kategori': 'Sepi',
                        'Waktu_Rata': rata_jam[key_sepi],
                        'Load_Rata': load_sepi
                    })

        if len(data_visual) == 0:
            st.warning("Data untuk visualisasi baru tidak cukup — pastikan kolom 'Jam ...' dan 'Jumlah ...' ada pada dataset.")
        else:
            df_visual = pd.DataFrame(data_visual)
            st.write("Contoh data visualisasi:")
            st.dataframe(df_visual.head(), use_container_width=True)

            # --- 2. Visualisasi Bar Plot ---
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            df_plot = df_visual.sort_values(by='Load_Rata', ascending=False).copy()
            df_plot['Kategori_Gabungan'] = df_plot['Kategori'].astype(str) + ' (' + df_plot['Hari'].astype(str) + ')'

            order_bar = ['Ramai (Weekday)', 'Ramai (Weekend)', 'Sedang (Weekday)', 'Sedang (Weekend)', 'Sepi (Weekday)', 'Sepi (Weekend)']

            sns.barplot(
                data=df_plot[df_plot['Jenis'] == 'Motor'],
                x='Kategori_Gabungan',
                y='Load_Rata',
                order=order_bar,
                palette='viridis',
                ax=axes[0]
            )
            axes[0].set_title('Rata-rata Kepadatan Parkir Motor (Berdasarkan Kategori Waktu)')
            axes[0].set_xlabel('Kategori Waktu')
            axes[0].set_ylabel('Rata-rata Jumlah Motor')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(axis='y', linestyle='--', alpha=0.7)

            sns.barplot(
                data=df_plot[df_plot['Jenis'] == 'Mobil'],
                x='Kategori_Gabungan',
                y='Load_Rata',
                order=order_bar,
                palette='magma',
                ax=axes[1]
            )
            axes[1].set_title('Rata-rata Kepadatan Parkir Mobil (Berdasarkan Kategori Waktu)')
            axes[1].set_xlabel('Kategori Waktu')
            axes[1].set_ylabel('Rata-rata Jumlah Mobil')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)

            # --- 3. Visualisasi Heatmap ---
            df_heatmap = df_visual.pivot_table(index=['Hari', 'Kategori'], columns='Jenis', values='Load_Rata')
            order_hari = ['Weekday', 'Weekend']
            order_kategori = ['Ramai', 'Sedang', 'Sepi']
            idx = pd.MultiIndex.from_product([order_hari, order_kategori], names=['Hari', 'Kategori'])
            df_heatmap = df_heatmap.reindex(idx, fill_value=0)

            plt.figure(figsize=(8, 5))
            df_heatmap_final = df_heatmap.unstack(level=0)
            sns.heatmap(
                df_heatmap_final.T,
                annot=True,
                fmt=".0f",
                cmap="YlGnBu",
                linewidths=.5,
                cbar_kws={'label': 'Rata-rata Jumlah Kendaraan'}
            )
            plt.title('Heatmap Perbandingan Kepadatan (Load) Parkir')
            plt.xlabel('Kategori Waktu dan Hari')
            plt.ylabel('Jenis Kendaraan')
            plt.yticks(rotation=0)
            st.pyplot(plt.gcf())
        
    with tab4:
        st.subheader("Grafik Garis 24 Jam (Alternatif)")
        st.info("Grafik garis 24 jam sintetis yang menggambarkan tren harian (jika ingin melihat alternatif visual).")
        # Panggil fungsi garis 24 jam yang telah tersedia
        try:
            plot_load_24_hours(df)
        except Exception as e:
            st.warning(f"Gagal menampilkan grafik 24 jam: {e}")

    with st.expander("Penjelasan Dukungan (Confidence & Top 1 Pendukung)"):
        st.markdown(
            """
            * **Confidence (Dukungan):** Menggambarkan probabilitas model yakin pada kelas prediksi tersebut (misalnya 0.95 berarti 95% yakin).
            * **Top 1 Pendukung (Local Gain):** Fitur (kolom) yang memiliki pengaruh paling besar (*gain*) dalam memprediksi kelas untuk *input* spesifik di baris tersebut. Ini membuktikan bahwa model menggunakan fitur yang relevan, bukan hanya menebak.
            """
        )

# --- Modul 3: Pemodelan (DITAMBAHKAN) ---
def display_modeling(models_data):
    st.header("3️⃣ Pemodelan Klasifikasi Potensi Tarif (Random Forest)")
    st.markdown("---")
    
    tab_motor, tab_mobil, tab_rekomendasi = st.tabs(["🏍️ Model Motor", "🚗 Model Mobil", "📑 Rekomendasi Tarif"])

    def display_model_results(jenis, data):
        st.subheader(f"Hasil Pelatihan Model {jenis.capitalize()}")
        
        model = data['model']
        y_test = data['y_test']
        y_pred = data['y_pred']
        le = data['le']

        if model is None:
            st.error(f"Model {jenis.capitalize()} tidak dilatih karena kolom target 'Class_{jenis.capitalize()}' tidak memiliki variasi kelas yang cukup (nunique <= 1).")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                acc = accuracy_score(y_test, y_pred)
                st.metric(f"Akurasi Model {jenis.capitalize()}", f"{acc*100:.2f} %")
                
                st.markdown("#### Matriks Konfusi (Confusion Matrix)")
                fig, ax = plt.subplots(figsize=(6, 5))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
                ax.set_title('Confusion Matrix')
                ax.set_ylabel('Actual Class')
                ax.set_xlabel('Predicted Class')
                st.pyplot(fig)
            except ValueError as e:
                st.warning(f"Tidak dapat menghitung metrik karena kelas prediksi tidak sesuai dengan kelas sebenarnya: {e}")

        with col2:
            st.markdown("#### Laporan Klasifikasi")
            try:
                report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose().iloc[:-3, :-1] # Ambil Precision, Recall, F1-Score
                st.dataframe(report_df, use_container_width=True)
            except ValueError as e:
                st.warning(f"Tidak dapat membuat Laporan Klasifikasi: {e}")

            st.markdown("#### Feature Importance")
            importance = pd.Series(model.feature_importances_, index=data['fitur']).sort_values(ascending=False).head(5)
            fig_imp, ax_imp = plt.subplots(figsize=(7, 5))
            sns.barplot(x=importance.values, y=importance.index, ax=ax_imp, palette='magma')
            ax_imp.set_title(f'Top 5 Feature Importance - {jenis.capitalize()}')
            ax_imp.set_xlabel('Importance Score')
            st.pyplot(fig_imp)
            
    with tab_motor:
        display_model_results('Motor', models_data['motor'])
        
    with tab_mobil:
        display_model_results('Mobil', models_data['mobil'])

    with tab_rekomendasi:
        st.subheader("📑 Tabel Rekomendasi Kebijakan Tarif Progresif")
        st.info("Menampilkan hasil klasifikasi potensi tarif untuk setiap titik parkir berdasarkan model yang terlatih.")
        
        tarif_mapping = {
            'Motor': {'Rendah': 1000, 'Sedang': 2000, 'Tinggi': 3000},
            'Mobil': {'Rendah': 3000, 'Sedang': 4000, 'Tinggi': 5000}
        }
        
        model_motor = models_data['motor']['model']
        le_motor = models_data['motor']['le']
        fitur_motor = models_data['motor']['fitur']
        X_all_motor = models_data['motor']['X_all']
        
        model_mobil = models_data['mobil']['model']
        le_mobil = models_data['mobil']['le']
        fitur_mobil = models_data['mobil']['fitur']
        X_all_mobil = models_data['mobil']['X_all']
        
        if model_motor is None or model_mobil is None:
            st.warning("Model tidak tersedia atau tidak dapat dilatih. Pastikan halaman Pemodelan sudah dijalankan dengan data yang cukup.")
        else:
            try:
                # Prediksi untuk semua data
                y_pred_m_enc = model_motor.predict(X_all_motor)
                df_result = pd.DataFrame(X_all_motor).reset_index(drop=True)
                
                # Tambahkan kolom Titik Lokasi dari df_processed asli
                df_result.insert(0, 'Titik Parkir', df_processed['Titik'].values)
                
                df_result['Klasifikasi Potensi (Motor)'] = le_motor.inverse_transform(y_pred_m_enc)
                df_result['Rekomendasi Tarif (Motor)'] = df_result['Klasifikasi Potensi (Motor)'].apply(
                    lambda x: f"Rp{tarif_mapping['Motor'].get(x, 0):,.0f}"
                )
                
                y_pred_c_enc = model_mobil.predict(X_all_mobil)
                df_result['Klasifikasi Potensi (Mobil)'] = le_mobil.inverse_transform(y_pred_c_enc)
                df_result['Rekomendasi Tarif (Mobil)'] = df_result['Klasifikasi Potensi (Mobil)'].apply(
                    lambda x: f"Rp{tarif_mapping['Mobil'].get(x, 0):,.0f}"
                )
                
                # Kolom output
                kolom_output = ['Titik Parkir', 'Klasifikasi Potensi (Motor)', 'Rekomendasi Tarif (Motor)', 
                               'Klasifikasi Potensi (Mobil)', 'Rekomendasi Tarif (Mobil)']
                
                st.markdown("### Ringkasan Rekomendasi Tarif (10 Baris Pertama)")
                st.dataframe(df_result[kolom_output].head(10), use_container_width=True)
                
                st.markdown("---")
                st.markdown("### Statistik Distribusi Klasifikasi")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Distribusi Motor")
                    motor_dist = df_result['Klasifikasi Potensi (Motor)'].value_counts()
                    st.bar_chart(motor_dist)
                
                with col2:
                    st.markdown("#### Distribusi Mobil")
                    mobil_dist = df_result['Klasifikasi Potensi (Mobil)'].value_counts()
                    st.bar_chart(mobil_dist)
                
                st.markdown("---")
                st.markdown("### Tabel Lengkap Rekomendasi")
                st.dataframe(df_result[kolom_output], use_container_width=True)
                
                # Download option
                csv = df_result[kolom_output].to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="Rekomendasi_Tarif_Parkir.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Gagal membuat rekomendasi: {e}")


# Modul 4: Peta & Simulasi (Diubah: Dropdown untuk Pilihan Lokasi)
def display_map_and_simulation(df_long, map_center, models_data, df_spasial):
    st.header("4️⃣ Peta & Simulasi Tarif Progresif")
    st.markdown("---")
    
    st.subheader("Peta Prediksi Potensi Tarif Statis")
    st.info("Pilih gaya peta: Satelit atau StreetMap. Warna titik diseragamkan untuk fokus pada popup dan simulasi.")

    # Default peta: OpenStreetMap (TileLayer tambahan tetap disertakan di dalam map)
    m = folium.Map(location=map_center, zoom_start=13, tiles='OpenStreetMap')

    # Tambahkan TileLayer tambahan agar pengguna bisa berganti di LayerControl jika mau
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', name='Esri Satellite', attr='Esri').add_to(m)
    
    FIXED_COLOR = 'darkblue'

    fg_all = folium.FeatureGroup(name='Semua Titik Parkir', show=True)
    features_search = []
    
    # 1. CircleMarker (Titik Bulat, Warna Tunggal) - VISUAL
    for index, row in df_spasial.iterrows():
        titik = row['Titik']
        lat, lon = row['Latitude'], row['Longitude']
        
        # Ambil hasil klasifikasi statis dari df_processed (sudah ada di df_long)
        motor_data = df_long[(df_long['titik'] == titik) & (df_long['jenis_kendaraan'] == 'Motor')]
        mobil_data = df_long[(df_long['titik'] == titik) & (df_long['jenis_kendaraan'] == 'Mobil')]

        # Mengambil baris pertama (harusnya hanya satu)
        motor_row = motor_data.iloc[0] if not motor_data.empty else None
        mobil_row = mobil_data.iloc[0] if not mobil_data.empty else None

        # Menghandle kasus data hilang
        motor_potensi = motor_row['kategori_load'].upper() if motor_row is not None else 'N/A'
        motor_tarif = int(motor_row['prediksi_tarif']) if motor_row is not None else 0
        mobil_potensi = mobil_row['kategori_load'].upper() if mobil_row is not None else 'N/A'
        mobil_tarif = int(mobil_row['prediksi_tarif']) if mobil_row is not None else 0

        popup_html = f"""
        <div style="font-size:13px; font-family:sans-serif;">
            <b>Titik Parkir:</b> {titik}<br>
            <b>Koordinat:</b> {lat:.4f}, {lon:.4f}<hr>
            <b>Motor:</b> Potensi {motor_potensi} (Tarif Dasar: Rp{motor_tarif:,})<br>
            <b>Mobil:</b> Potensi {mobil_potensi} (Tarif Dasar: Rp{mobil_tarif:,})<br>
        </div>
        """

        marker = folium.CircleMarker(
            location=[lat, lon], 
            radius=6, 
            color=FIXED_COLOR, 
            fill=True, 
            fill_color=FIXED_COLOR, 
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=300), 
            tooltip=titik 
        )
        marker.add_to(fg_all)

        features_search.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            "properties": {"name": titik}, 
        })

    fg_all.add_to(m)
    geojson_layer_search = folium.GeoJson(
        {"type": "FeatureCollection", "features": features_search},
        name="Titik Pencarian (Tersembunyi)",
        style_function=lambda x: {'opacity': 0, 'fillOpacity': 0, 'weight': 0, 'color': 'transparent'},
    ).add_to(m)

    Search(
        layer=geojson_layer_search, 
        search_label="name", 
        placeholder="Cari nama titik parkir...", 
        collapsed=False, 
        position="topleft", 
        geom_type="Point",
    ).add_to(m)
    
    # ... (Layer Control dan Plugins lainnya) ...
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Tambahkan Fullscreen dan MiniMap untuk UX yang lebih baik
    Fullscreen(position='topright').add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    
    folium_static(m, width=None, height=550)

    # --- Bagian Simulasi (Input Dinamis dari Dropdown) ---
    st.subheader("Simulasi Prediksi Potensi Tarif (What-If Analysis)")
    st.markdown("**1. Pilih Lokasi Parkir** (Input Statis Agregat Lokasi akan digunakan sebagai *Default*)")
    
    # Pilihan Dropdown Lokasi (Pengganti klik di peta)
    selected_titik = st.selectbox(
        "Pilih Titik Parkir untuk Simulasi:", 
        df_spasial['Titik'].unique().tolist(), 
        key='sim_titik_select'
    )
    
    # Ambil data agregat dari titik yang dipilih
    default_data = df_spasial[df_spasial['Titik'] == selected_titik].iloc[0]
    
    # Menentukan default Jam Ramai (mengambil nilai rata-rata dari Weekday Motor)
    default_jam_val = default_data.get('Jam Ramai Motor Weekday', 9.0)

    with st.expander(f"⚙️ Atur Skenario Dinamis untuk {selected_titik} (Tarif Progresif) ⚙️"):
        # Memberikan informasi default yang lebih informatif
        st.markdown(f"**Data Agregat Default Lokasi:**")
        st.markdown(f"* Jumlah Motor WD: **{default_data.get('Jumlah Motor Weekday', 0):.0f}** unit, Mobil WD: **{default_data.get('Jumlah Mobil Weekday', 0):.0f}** unit")
        st.markdown(f"* Jam Ramai WD Motor: **{default_jam_val:.2f}**")

        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1: 
            jenis = st.selectbox("Jenis Kendaraan", ['Motor', 'Mobil'], key='sim_jenis', help="Pilih jenis kendaraan yang ingin disimulasikan.")
        with col2: 
            hari = st.selectbox("Hari", ['Weekday', 'Weekend'], key='sim_hari', help="Pilih apakah hari kerja atau akhir pekan.")
        with col3: 
            # Menggunakan nilai default jam yang lebih aman
            jam_for_time_input = default_jam_val
            try:
                time_obj_default = datetime.time(int(jam_for_time_input // 1), int((jam_for_time_input % 1) * 60))
            except ValueError:
                time_obj_default = datetime.time(9, 0) # Fallback ke jam 9 pagi
                
            time_obj = st.time_input(
                "Jam (HH:MM)", 
                value=time_obj_default, 
                step=datetime.timedelta(minutes=1), # Mengizinkan input per menit
                key='sim_jam_time', 
                help="Waktu parkir (Format 24 jam)."
            )
            jam_desimal_input = time_to_decimal_hour(time_obj) 
            st.caption(f"Nilai Jam Model: **{jam_desimal_input:.2f}**") 
            
        with col4: 
            # Menggunakan nilai rata-rata jumlah kendaraan di lokasi yang dipilih sebagai default
            default_jumlah = default_data.get(f'Jumlah {jenis} {hari}', 100)
            jumlah_input = st.number_input(f"Jumlah {jenis} (Estimasi)", min_value=1, max_value=500, value=int(default_jumlah), key='sim_jumlah', help=f"Estimasi jumlah {jenis} yang parkir pada jam tersebut.")
        with col5: 
            st.markdown("<br>", unsafe_allow_html=True) 
            submitted = st.button("Prediksi Hasil 🚀", key='sim_submit', type='primary')

        if submitted:
            data = models_data['motor'] if jenis == 'Motor' else models_data['mobil']
            
            # Panggil fungsi prediksi
            pred_class, confidence, top_gain, proba_dict, keterangan_jam = predict_single_input(
                jenis, hari, jam_desimal_input, jumlah_input, data['model'], data['le'], data['X_ref']
            )
            
            if not isinstance(pred_class, str) or "Error" in pred_class or "Model Gagal" in pred_class:
                st.error(f"Simulasi Gagal: {pred_class}. Pastikan model terlatih (Cek halaman Pemodelan).")
            else:
                # >>> Terapkan Logika Tarif Progresif
                rekomendasi_tarif_dasar = tarif_mapping[jenis].get(pred_class, 0)
                rekomendasi_tarif_progresif = calculate_progresif_tarif(jenis, pred_class, jam_desimal_input)
                
                st.markdown("---")
                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Kategori Potensi Tarif (Simulasi)", f"Potensi {pred_class.upper()}", delta=f"Confidence: {confidence:.3f}")
                col_res2.metric("Rekomendasi Tarif Dasar", f"Rp{rekomendasi_tarif_dasar:,}", delta=f"Kelas: {pred_class}")
                
                col_res3.metric("Rekomendasi Tarif PROGRESIF", f"Rp{rekomendasi_tarif_progresif:,}", delta=f"Kenaikan: Rp{rekomendasi_tarif_progresif - rekomendasi_tarif_dasar:,}")
                
                st.markdown("---")
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.markdown("**Penjelasan Logika Waktu:**")
                    st.info(keterangan_jam)
                    st.markdown("**Top 3 Kontributor (Local Gain):**")
                    if isinstance(top_gain, pd.Series):
                        for f in top_gain.index: st.markdown(f"- **{f}** (Pendorong utama prediksi)")
                    st.caption(f"Probabilitas Semua Kelas: {proba_dict}")

                with col_info2:
                    st.markdown("**Logika Progresif yang Diterapkan:**")
                    st.warning(f"Jika **Jam > 9.00**, Tarif **{pred_class}** dinaikkan sebesar **Rp{rekomendasi_tarif_progresif - rekomendasi_tarif_dasar:,}** dari tarif dasar.")


# =================================================================
# === EKSEKUSI APLIKASI UTAMA ===
# =================================================================

# Load Data dan Model
df_processed, df_spasial, jam_cols, df_raw, batas_kuantil = load_and_preprocess_data(FILE_PATH)
if df_processed is None: 
    st.error(f"Gagal memuat atau memproses data. Pastikan file '{FILE_PATH}' ada dan formatnya benar, serta mengandung kolom spasial (Titik, Latitude, Longitude).")
    st.stop()

models_data = train_models(df_processed, jam_cols)

# --- Prediksi Statis untuk Peta ---
df_long = pd.DataFrame()
try:
    # Memastikan model terlatih sebelum memprediksi
    if models_data['motor']['model']:
        df_processed['Pred_Class_Motor'] = models_data['motor']['le'].inverse_transform(models_data['motor']['model'].predict(models_data['motor']['X_all']))
    else:
        df_processed['Pred_Class_Motor'] = df_processed['Class_Motor'] # Fallback ke kelas data mentah
        
    if models_data['mobil']['model']:
        df_processed['Pred_Class_Mobil'] = models_data['mobil']['le'].inverse_transform(models_data['mobil']['model'].predict(models_data['mobil']['X_all']))
    else:
        df_processed['Pred_Class_Mobil'] = df_processed['Class_Mobil'] # Fallback ke kelas data mentah

    df_mapping = df_spasial.dropna(subset=['Latitude', 'Longitude'])

    df_motor_map = df_mapping.copy()
    df_motor_map['jenis_kendaraan'] = 'Motor'
    df_motor_map['kategori_load'] = df_processed['Pred_Class_Motor']
    df_motor_map['prediksi_tarif'] = df_processed['Pred_Class_Motor'].apply(lambda x: tarif_mapping['Motor'].get(x, 0))
    df_motor_map.rename(columns={'Titik': 'titik', 'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True) 

    df_mobil_map = df_mapping.copy()
    df_mobil_map['jenis_kendaraan'] = 'Mobil'
    df_mobil_map['kategori_load'] = df_processed['Pred_Class_Mobil']
    df_mobil_map['prediksi_tarif'] = df_processed['Pred_Class_Mobil'].apply(lambda x: tarif_mapping['Mobil'].get(x, 0))
    df_mobil_map.rename(columns={'Titik': 'titik', 'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True) 

    df_long = pd.concat([df_motor_map, df_mobil_map], ignore_index=True).dropna(subset=['latitude', 'longitude']) 

    if not df_long.empty:
        map_center = [df_long['latitude'].mean(), df_long['longitude'].mean()]
    else:
        map_center = [-7.4168, 109.2155] # Koordinat Default Banyumas
        
except Exception as e:
    st.error(f"Error saat menyiapkan data peta/prediksi statis: {e}")
    map_center = [-7.4168, 109.2155] # Koordinat Default Banyumas

st.title("🅿️ Analisis Potensi Tarif Parkir")
st.caption("Dashboard untuk pemodelan klasifikasi potensi tarif parkir berbasis Random Forest.")
st.markdown("---")

# --- Sidebar Navigasi Utama (MENGGUNAKAN OPTION MENU DENGAN STYLE) ---
with st.sidebar:
    st.markdown("---")
    
    page = option_menu(
        menu_title="Dashboard Analitik 📊", 
        options=["Data Table", "Visualisasi", "Pemodelan", "Peta & Simulasi"],
        icons=["table", "bar-chart", "calculator", "geo-alt"], 
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
    )

# --- Display Logic ---
if page == "Data Table":
    display_data_table(df_raw, df_processed)

elif page == "Visualisasi":
    display_visualization(df_processed, batas_kuantil, jam_cols)

elif page == "Pemodelan":
    display_modeling(models_data)

elif page == "Peta & Simulasi":
    display_map_and_simulation(df_long, map_center, models_data, df_spasial)