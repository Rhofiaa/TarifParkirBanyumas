import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import folium
from streamlit_folium import folium_static 
import matplotlib.pyplot as plt
import seaborn as sns
from folium.plugins import Search, Fullscreen, MiniMap
from streamlit_option_menu import option_menu
import re

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Parking Tariff Analysis Dashboard",
    layout="wide", 
    initial_sidebar_state="expanded",
)

# CHANGE THIS PATH TO YOUR LOCAL EXCEL FILE PATH
FILE_PATH = 'DataParkir_Fix.xlsx' 

# --- UTILITY FUNCTIONS (Time Conversion, Category, Tariff) ---
def parse_time_to_decimal(time_str):
    """Convert time string (H.M, H:M, or H) to decimal hours."""
    try:
        time_str = str(time_str).replace(',', '.').replace(':', '.')
        if '.' in time_str:
            h_str, m_part_str = time_str.split('.', 1)
            h = int(h_str) if h_str else 0
            # Assume H.MM is H hours and MM minutes
            m = int(m_part_str.ljust(2, '0')[:2]) 
            return h + m / 60.0
        else:
            return float(time_str)
    except Exception:
        return np.nan

def convert_time_range(x):
    """Convert time range format (e.g., '20.00-22.00') to average decimal hours (e.g., 21.0)."""
    if pd.isna(x) or str(x).strip() in ('-', '', 'nan'):
        return np.nan
    s = str(x).strip()
    try:
        parts = re.split(r'\s*-\s*', s)
        start_time_dec = parse_time_to_decimal(parts[0].strip())
        end_time_dec = parse_time_to_decimal(parts[1].strip()) if len(parts) > 1 else start_time_dec
        # Handle midnight crossing (22 -> 02)
        if len(parts) > 1 and pd.notna(start_time_dec) and pd.notna(end_time_dec) and end_time_dec < start_time_dec:
            end_time_dec += 24.0
        if pd.isna(start_time_dec) or pd.isna(end_time_dec):
            return np.nan
        return (start_time_dec + end_time_dec) / 2
    except Exception:
        return np.nan

def time_to_decimal_hour(time_obj):
    """Convert datetime.time object (H:M) to decimal hours (H + M/60)."""
    if time_obj is None:
        return np.nan
    return time_obj.hour + time_obj.minute / 60.0

def kategori_jam_otomatis(jam):
    """Automatically categorize hour into Off-Peak/Moderate/Peak based on time."""
    if (jam >= 0 and jam < 6) or (jam >= 22 and jam < 24):
        return 'Off-Peak'
    elif (jam >= 9 and jam < 19):
        return 'Peak'
    else:
        return 'Moderate'

# Base Tariff Mapping
tarif_mapping = {
    'Motorcycle': {'Low': 1000, 'Medium': 2000, 'High': 3000},
    'Car': {'Low': 3000, 'Medium': 4000, 'High': 5000}
}

# Progressive Tariff Function
def calculate_progresif_tarif(jenis, potensi_class, jam_desimal):
    """Apply progressive tariff logic based on potential class and hour."""
    tarif_dasar = tarif_mapping[jenis].get(potensi_class, 0)
    
    # Progressive Logic: Increase tariff above 9:00
    if jam_desimal > 9.0:
        if potensi_class == 'High':
            return tarif_dasar + 1000  # E.g., from 3000 to 4000
        elif potensi_class == 'Medium':
            return tarif_dasar + 500   # E.g., from 2000 to 2500
        else:
            return tarif_dasar
    else:
        return tarif_dasar

# --- Data Loading and Preprocessing (Cached) ---
@st.cache_data
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        return None, None, None, None, None
    
    # Remove header row (row 0 with NaN)
    df = df.iloc[1:].reset_index(drop=True)
    
    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
    
    # Drop non-annual revenue columns
    df = df.drop(columns=[c for c in df.columns if 'Parking Fee Revenue' in c and 'Annual' not in c], errors='ignore')
    
    df_raw = df.copy()

    # Annual Revenue Columns
    pend_cols = [
        'Annual Parking Fee Revenue (Weekday – Motorcycles)',
        'Annual Parking Fee Revenue (Weekday – Cars)',
        'Annual Parking Fee Revenue (Weekend – Motorcycles)',
        'Annual Parking Fee Revenue (Weekend – Cars)'
    ]
    
    # Number of Vehicles Columns (for Load Graphs)
    jumlah_cols = [c for c in df.columns if c.startswith('Number of')]

    # Clean and convert revenue columns
    for c in pend_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.replace(r'[^\d,\.]', '', regex=True)
            df[c] = df[c].str.replace('.', '', regex=False)
            df[c] = df[c].str.replace(',', '.', regex=False)
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Convert and impute time columns
    jam_cols = [c for c in df.columns if c.startswith(('Peak Hours', 'Moderate Hours', 'Off-Peak Hours'))]
    for col in jam_cols:
        df[col] = df[col].apply(convert_time_range)
        df[col] = df[col].fillna(df[col].mean())  # Imputation: Mean for missing values

    # Impute other numeric columns
    for col in df.columns:
        if df[col].dtype != 'object':
            df[col] = df[col].fillna(df[col].median())
        else:
            if col not in ['Location Point', 'Class_Motorcycle', 'Class_Car']:
                if df[col].mode().empty:
                    df[col] = df[col].fillna('Unknown')
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

    # Calculate total revenue
    moto_pend_cols = [c for c in pend_cols if 'Motorcycle' in c]
    car_pend_cols = [c for c in pend_cols if 'Car' in c]
    df['Total_Revenue_Motorcycle'] = df[moto_pend_cols].sum(axis=1) 
    df['Total_Revenue_Car'] = df[car_pend_cols].sum(axis=1) 

    # Classify tariff potential (Target)
    batas_moto = None
    batas_car = None
    
    try:
        df['Class_Motorcycle'] = pd.qcut(df['Total_Revenue_Motorcycle'], q=3, labels=['Low','Medium','High'], duplicates='drop')
        batas_moto = df['Total_Revenue_Motorcycle'].quantile([0.333, 0.666]).drop_duplicates().sort_values()
    except ValueError:
        df['Class_Motorcycle'] = pd.cut(df['Total_Revenue_Motorcycle'], bins=[-np.inf, df['Total_Revenue_Motorcycle'].median(), np.inf], labels=['Low', 'High']).fillna('Low')
        batas_moto = df['Total_Revenue_Motorcycle'].quantile([0.5]).drop_duplicates().sort_values()
        
    try:
        df['Class_Car'] = pd.qcut(df['Total_Revenue_Car'], q=3, labels=['Low','Medium','High'], duplicates='drop')
        batas_car = df['Total_Revenue_Car'].quantile([0.333, 0.666]).drop_duplicates().sort_values()
    except ValueError:
        df['Class_Car'] = pd.cut(df['Total_Revenue_Car'], bins=[-np.inf, df['Total_Revenue_Car'].median(), np.inf], labels=['Low', 'High']).fillna('Low')
        batas_car = df['Total_Revenue_Car'].quantile([0.5]).drop_duplicates().sort_values()

    if all(c in df.columns for c in ['Latitude', 'Longitude', 'Location Point']):
        df_spasial = df[['Latitude', 'Longitude', 'Location Point'] + jam_cols + jumlah_cols].copy()
        # Remove rows without location information
        df_spasial['Location Point'] = df_spasial['Location Point'].astype(str).str.strip()
        df_spasial = df_spasial.replace({'Location Point': {'nan': None}})
        df_spasial = df_spasial.dropna(subset=['Location Point', 'Latitude', 'Longitude'])
        df_spasial = df_spasial.reset_index(drop=True)

        # Also remove from main df
        before_drop = df.shape[0]
        df['Location Point'] = df['Location Point'].astype(str).str.strip()
        df = df.replace({'Location Point': {'nan': None}})
        df = df.dropna(subset=['Location Point']).reset_index(drop=True)
        after_drop = df.shape[0]
        
        st.session_state.setdefault('rows_dropped_no_location', 0)
        st.session_state['rows_dropped_no_location'] = before_drop - after_drop
    else:
        st.error("Coordinate columns ('Location Point', 'Latitude', 'Longitude') not found.")
        return None, None, None, None, None
    
    return df, df_spasial, jam_cols, df_raw, {'motorcycle': batas_moto, 'car': batas_car}

# --- Train Random Forest Models (Cached) ---
@st.cache_resource
def train_models(df, jam_cols):
    fitur_moto = ['Number of Motorcycles (Weekday)', 'Number of Motorcycles (Weekend)'] + [c for c in jam_cols if 'Motorcycle' in c]
    fitur_car = ['Number of Cars (Weekday)', 'Number of Cars (Weekend)'] + [c for c in jam_cols if 'Car' in c]

    def build_model(X, y):
        le = LabelEncoder()
        if len(y.unique()) > 1:
            y_enc = le.fit_transform(y)
        else:
            y_enc = y 
            
        if len(y.unique()) > 1 and all(y.value_counts() > 1):
            X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
            
        if X_train.empty:
            return None, le, pd.DataFrame(), pd.DataFrame(), np.array([]), np.array([]), np.array([]), pd.DataFrame(), None
            
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Calculate cross-validation score (5-fold)
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
        
        X_ref = pd.concat([X_train, X_test]).reset_index(drop=True)
        return model, le, X_train, X_test, y_train, y_test, y_pred, X_ref, cv_score
    
    results = {}
    
    if df['Class_Motorcycle'].nunique() > 1:
        model_moto, le_moto, X_train_m, X_test_m, y_train_m, y_test_m, y_pred_m, X_ref_m, cv_moto = build_model(df[fitur_moto], df['Class_Motorcycle'])
    else:
        model_moto, le_moto, X_train_m, X_test_m, y_train_m, y_test_m, y_pred_m, X_ref_m, cv_moto = [None] * 9

    results['motorcycle'] = {
        'model': model_moto, 'le': le_moto, 'X_train': X_train_m, 'X_test': X_test_m, 'y_train': y_train_m, 
        'y_test': y_test_m, 'y_pred': y_pred_m, 'X_ref': X_ref_m, 'fitur': fitur_moto, 'X_all': df[fitur_moto], 'cv_score': cv_moto
    }

    if df['Class_Car'].nunique() > 1:
        model_car, le_car, X_train_c, X_test_c, y_train_c, y_test_c, y_pred_c, X_ref_c, cv_car = build_model(df[fitur_car], df['Class_Car'])
    else:
        model_car, le_car, X_train_c, X_test_c, y_train_c, y_test_c, y_pred_c, X_ref_c, cv_car = [None] * 9

    results['car'] = {
        'model': model_car, 'le': le_car, 'X_train': X_train_c, 'X_test': X_test_c, 'y_train': y_train_c, 
        'y_test': y_test_c, 'y_pred': y_pred_c, 'X_ref': X_ref_c, 'fitur': fitur_car, 'X_all': df[fitur_car], 'cv_score': cv_car
    }
    
    return results

# Prediction Function for Simulation
def predict_single_input(jenis, hari, jam_input, jumlah_input, model, le, X_ref): 
    if model is None:
        return "Model Failed", 0.0, pd.Series({"No Model": 0}), {"Error": 1.0}, "No trained model"

    kategori_jam = kategori_jam_otomatis(jam_input)
    prefix = jenis
    
    # Use mean from X_ref
    data_baru = pd.DataFrame([X_ref.mean()], columns=X_ref.columns)
    
    kolom_jumlah = f'Number of {prefix} ({hari})'
    if kolom_jumlah in data_baru.columns: 
        data_baru[kolom_jumlah] = jumlah_input
    
    kolom_jam_input = f'{kategori_jam} Hours for {prefix} ({hari})'
    
    if kolom_jam_input in data_baru.columns: 
        data_baru[kolom_jam_input] = jam_input

    keterangan_jam = f"Input hour **{jam_input:.2f}** is categorized as **'{kategori_jam}'**."

    try:
        pred_encoded = model.predict(data_baru)[0]
        pred_class = le.inverse_transform([pred_encoded])[0]
        proba = model.predict_proba(data_baru)[0]
        confidence = proba[pred_encoded] 
        
        # Local Gain Implementation
        global_importance = pd.Series(model.feature_importances_, index=model.feature_names_in_)
        
        local_gain_calc = (data_baru.iloc[0] - X_ref.mean()) * global_importance
        top_gain = local_gain_calc.abs().sort_values(ascending=False).head(3)
        
        proba_dict = dict(zip(le.classes_, proba))
        
        return pred_class, confidence, top_gain, proba_dict, keterangan_jam
    except Exception as e:
        return f"Prediction Error: {e}", 0.0, pd.Series({"Error": 0}), {"Error": 1.0}, keterangan_jam

# --- Module 1: Data Table ---
def display_data_table(df_raw, df_processed):
    st.header("1️⃣ Raw Data & Preprocessed Data")
    st.markdown("---")
    
    tab_raw, tab_processed = st.tabs(["📁 Raw Data", "✨ Preprocessed Data"])

    with tab_raw:
        st.subheader("Original Raw Data")
        st.info("Original data before cleaning, conversion, and imputation (missing values still visible).")
        st.dataframe(df_raw, use_container_width=True)

    with tab_processed:
        st.subheader("Preprocessed Data (Ready for Modeling)")
        st.info("Data after cleaning, conversion to numeric, imputation, and addition of **Total Revenue** and **Tariff Potential Class** (Classification Target) columns.")
        st.dataframe(df_processed, use_container_width=True)

        st.markdown("---")
        st.subheader("Statistics Summary")
        
        col_m, col_c = st.columns(2)
        
        with col_m:
            st.markdown("#### Statistics: Total Revenue (Motorcycles)")
            st.dataframe(df_processed['Total_Revenue_Motorcycle'].describe().to_frame(), use_container_width=True)
            
        with col_c:
            st.markdown("#### Statistics: Total Revenue (Cars)")
            st.dataframe(df_processed['Total_Revenue_Car'].describe().to_frame(), use_container_width=True)


# --- Visualization Support Functions ---
def plot_load_vs_time(df, jam_cols, jumlah_cols):
    st.subheader("Average Load (Vehicle Count) vs Time (0-24 Hours)")
    st.info("This graph illustrates average vehicle load correlation with time ranges in your dataset.")
    
    df_load = df[jam_cols + jumlah_cols].copy()
    data_points = []
    
    # Motorcycles
    moto_cols = [c for c in df_load.columns if 'Motorcycle' in c]
    for jam_col in [c for c in moto_cols if 'Hours' in c]:
        try:
            match = re.search(r'(.*) Hours for Motorcycle (.*).*', jam_col)
            if match:
                kategori_jam = match.group(1)
                hari = match.group(2)
                jumlah_col = f'Number of Motorcycle ({hari})'
            else:
                continue

            if jumlah_col in df_load.columns:
                avg_jam = df_load[jam_col].mean()
                avg_jumlah = df_load[jumlah_col].mean()
                data_points.append({'Time (Decimal Hours)': avg_jam, 'Average Load': avg_jumlah, 'Type': 'Motorcycle - ' + hari, 'Category': kategori_jam})
        except Exception:
            pass

    # Cars
    car_cols = [c for c in df_load.columns if 'Car' in c]
    for jam_col in [c for c in car_cols if 'Hours' in c]:
        try:
            match = re.search(r'(.*) Hours for Car (.*).*', jam_col)
            if match:
                kategori_jam = match.group(1)
                hari = match.group(2)
                jumlah_col = f'Number of Car ({hari})'
            else:
                continue
                
            if jumlah_col in df_load.columns:
                avg_jam = df_load[jam_col].mean()
                avg_jumlah = df_load[jumlah_col].mean()
                data_points.append({'Time (Decimal Hours)': avg_jam, 'Average Load': avg_jumlah, 'Type': 'Car - ' + hari, 'Category': kategori_jam})
        except Exception:
            pass

    df_plot = pd.DataFrame(data_points)
    
    if not df_plot.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_plot, x='Time (Decimal Hours)', y='Average Load', hue='Type', style='Type', s=200, ax=ax)
        sns.lineplot(data=df_plot, x='Time (Decimal Hours)', y='Average Load', hue='Type', legend=False, alpha=0.5, ax=ax)
        ax.set_title('Average Vehicle Load by Time')
        ax.set_xticks(np.arange(0, 25, 3)) 
        ax.set_xlim(0, 24)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        for line in df_plot.itertuples():
            ax.annotate(line.Category, (line._1, line._2), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            
        st.pyplot(fig)
    else:
        st.warning("No load data available for plotting.")

# 24-Hour Line Graph Function
def plot_load_24_hours(df):
    """Plot 24-hour average vehicle count with load category background."""
    st.subheader("24-Hour Average Vehicle Count Line Graph 📉")
    st.info("Load Categories (Off-Peak/Moderate/Peak) are represented by background colors.")
    
    # Base load calculation
    jumlah_cols = [c for c in df.columns if c.startswith('Number of')]
    base_load = df[jumlah_cols].mean().mean() / 5 if jumlah_cols and df[jumlah_cols].mean().mean() > 0 else 50
    
    hours = np.arange(24)
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

    df_24h = pd.DataFrame({'Hour': hours, 'Average Vehicle Count': avg_counts_synth})
    df_24h['Load Category'] = df_24h['Hour'].apply(kategori_jam_otomatis)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'Off-Peak': "#EEF06A49", 'Peak': "#F4444445", 'Moderate': "#359EDF50"}
    
    # Background shading
    ax.axvspan(0, 6, color=colors['Off-Peak'], label='Off-Peak', zorder=0)
    ax.axvspan(6, 9.001, color=colors['Moderate'], label='Moderate', zorder=0) 
    ax.axvspan(9.001, 19, color=colors['Peak'], label='Peak', zorder=0)
    ax.axvspan(19, 22.001, color=colors['Moderate'], zorder=0)
    ax.axvspan(22.001, 24, color=colors['Off-Peak'], zorder=0) 
    
    # Plot line
    sns.lineplot(data=df_24h, x='Hour', y='Average Vehicle Count', marker='o', color='darkorange', linewidth=2, ax=ax)
    
    ax.set_title('24-Hour Average Vehicle Count vs Load Category')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Average Vehicle Count')
    ax.set_xticks(hours[::3])
    ax.set_xlim(0, 24)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.6, axis='both')
    
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=colors['Off-Peak'], label='Off-Peak (00-06 & 22-24)'),
        plt.Rectangle((0, 0), 1, 1, fc=colors['Peak'], label='Peak (09-19)'),
        plt.Rectangle((0, 0), 1, 1, fc=colors['Moderate'], label='Moderate (06-09 & 19-22)')
    ]
    ax.legend(handles=legend_elements, title='Load Category', loc='upper right')
    
    st.pyplot(fig)


# Module 2: Visualization
def display_visualization(df, batas_kuantil, jam_cols):
    st.header("2️⃣ Visualization & Data Analysis")
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Revenue Distribution & Classes", "💰 Quantile Thresholds (Rupiah)", "🔗 Average Density", "📉 Load vs Time (24-Hour Graph)"])
    
    with tab1:
        st.subheader("Total Revenue Distribution & Tariff Potential Class")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Tariff Potential Distribution - Motorcycles 🏍️")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.histplot(data=df, x='Total_Revenue_Motorcycle', hue='Class_Motorcycle', palette='viridis', multiple='stack', ax=ax, kde=True)
            ax.set_title('Total Motorcycle Revenue vs Class')
            ax.set_xlabel('Annual Total Revenue')
            st.pyplot(fig)
            with st.expander("View Count by Class"):
                st.dataframe(df['Class_Motorcycle'].value_counts().to_frame('Count'), use_container_width=True)

        with col2:
            st.markdown("#### Tariff Potential Distribution - Cars 🚗")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.histplot(data=df, x='Total_Revenue_Car', hue='Class_Car', palette='plasma', multiple='stack', ax=ax, kde=True)
            ax.set_title('Total Car Revenue vs Class')
            ax.set_xlabel('Annual Total Revenue')
            st.pyplot(fig)
            with st.expander("View Count by Class"):
                st.dataframe(df['Class_Car'].value_counts().to_frame('Count'), use_container_width=True)

    with tab2:
        st.subheader("💰 QUANTILE THRESHOLDS FOR ANNUAL REVENUE (Rp) 💰")
        st.warning("These thresholds are used to form classification labels (Low/Medium/High).")
        col_m, col_c = st.columns(2)
        
        if batas_kuantil['motorcycle'] is not None:
            with col_m:
                st.markdown("### Motorcycle Quantile Thresholds 🏍️")
                batas_moto = batas_kuantil['motorcycle']
                if len(batas_moto) == 2:
                    st.markdown(f"* **Low** : Revenue < **Rp{batas_moto.iloc[0]:,.0f}**")
                    st.markdown(f"* **Medium** : **Rp{batas_moto.iloc[0]:,.0f}** to **Rp{batas_moto.iloc[1]:,.0f}**")
                    st.markdown(f"* **High** : Revenue > **Rp{batas_moto.iloc[1]:,.0f}**")
                elif len(batas_moto) == 1:
                    st.markdown(f"* **Low** : Revenue < **Rp{batas_moto.iloc[0]:,.0f}**")
                    st.markdown(f"* **High** : Revenue > **Rp{batas_moto.iloc[0]:,.0f}**")
                else:
                    st.warning("Cannot calculate quantile thresholds for motorcycles.")
        else:
            col_m.warning("Motorcycle quantile thresholds unavailable.")

        if batas_kuantil['car'] is not None:
            with col_c:
                st.markdown("### Car Quantile Thresholds 🚗")
                batas_car = batas_kuantil['car']
                if len(batas_car) == 2:
                    st.markdown(f"* **Low** : Revenue < **Rp{batas_car.iloc[0]:,.0f}**")
                    st.markdown(f"* **Medium** : **Rp{batas_car.iloc[0]:,.0f}** to **Rp{batas_car.iloc[1]:,.0f}**")
                    st.markdown(f"* **High** : Revenue > **Rp{batas_car.iloc[1]:,.0f}**")
                elif len(batas_car) == 1:
                    st.markdown(f"* **Low** : Revenue < **Rp{batas_car.iloc[0]:,.0f}**")
                    st.markdown(f"* **High** : Revenue > **Rp{batas_car.iloc[0]:,.0f}**")
                else:
                    st.warning("Cannot calculate quantile thresholds for cars.")
        else:
            col_c.warning("Car quantile thresholds unavailable.")

    with tab3:
        st.subheader("Average Parking Density Visualization (Bar Plot + Heatmap)")
        st.info("Visual: Average time (decimal hours) for time categories and estimated load based on average vehicle count.")

        fitur_jam = [c for c in df.columns if 'Hours' in c]

        rata_jam = {}
        for jenis in ['Motorcycle', 'Car']:
            for hari in ['Weekday', 'Weekend']:
                for kategori in ['Peak', 'Moderate', 'Off-Peak']:
                    kolom = f'{kategori} Hours for {jenis} ({hari})'
                    if kolom in df.columns:
                        rata_jam[kolom] = df[kolom].mean()

        data_visual = []

        for jenis in ['Motorcycle', 'Car']:
            for hari in ['Weekday', 'Weekend']:
                kolom_jumlah = f'Number of {jenis} ({hari})'
                if kolom_jumlah in df.columns:
                    load_rata_kendaraan = df[kolom_jumlah].mean()
                else:
                    load_rata_kendaraan = 0

                load_peak = load_rata_kendaraan * 1.0
                load_moderate = load_rata_kendaraan * 0.5
                load_offpeak = load_rata_kendaraan * 0.2

                key_peak = f'Peak Hours for {jenis} ({hari})'
                key_moderate = f'Moderate Hours for {jenis} ({hari})'
                key_offpeak = f'Off-Peak Hours for {jenis} ({hari})'

                if key_peak in rata_jam:
                    data_visual.append({
                        'Type': jenis, 'Day': hari, 'Category': 'Peak',
                        'Avg_Time': rata_jam[key_peak],
                        'Avg_Load': load_peak
                    })
                if key_moderate in rata_jam:
                    data_visual.append({
                        'Type': jenis, 'Day': hari, 'Category': 'Moderate',
                        'Avg_Time': rata_jam[key_moderate],
                        'Avg_Load': load_moderate
                    })
                if key_offpeak in rata_jam:
                    data_visual.append({
                        'Type': jenis, 'Day': hari, 'Category': 'Off-Peak',
                        'Avg_Time': rata_jam[key_offpeak],
                        'Avg_Load': load_offpeak
                    })

        if len(data_visual) == 0:
            st.warning("Insufficient data for visualization - ensure 'Hours' and 'Number of' columns exist.")
        else:
            df_visual = pd.DataFrame(data_visual)
            st.write("Sample visualization data:")
            st.dataframe(df_visual.head(), use_container_width=True)

            # Bar Plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            df_plot = df_visual.sort_values(by='Avg_Load', ascending=False).copy()
            df_plot['Combined'] = df_plot['Category'].astype(str) + ' (' + df_plot['Day'].astype(str) + ')'

            order_bar = ['Peak (Weekday)', 'Peak (Weekend)', 'Moderate (Weekday)', 'Moderate (Weekend)', 'Off-Peak (Weekday)', 'Off-Peak (Weekend)']

            sns.barplot(
                data=df_plot[df_plot['Type'] == 'Motorcycle'],
                x='Combined',
                y='Avg_Load',
                order=order_bar,
                palette='viridis',
                ax=axes[0]
            )
            axes[0].set_title('Average Motorcycle Parking Density (By Time Category)')
            axes[0].set_xlabel('Time Category')
            axes[0].set_ylabel('Average Count')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(axis='y', linestyle='--', alpha=0.7)

            sns.barplot(
                data=df_plot[df_plot['Type'] == 'Car'],
                x='Combined',
                y='Avg_Load',
                order=order_bar,
                palette='magma',
                ax=axes[1]
            )
            axes[1].set_title('Average Car Parking Density (By Time Category)')
            axes[1].set_xlabel('Time Category')
            axes[1].set_ylabel('Average Count')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)

            # Heatmap
            df_heatmap = df_visual.pivot_table(index=['Day', 'Category'], columns='Type', values='Avg_Load')
            order_day = ['Weekday', 'Weekend']
            order_kategori = ['Peak', 'Moderate', 'Off-Peak']
            idx = pd.MultiIndex.from_product([order_day, order_kategori], names=['Day', 'Category'])
            df_heatmap = df_heatmap.reindex(idx, fill_value=0)

            plt.figure(figsize=(8, 5))
            df_heatmap_final = df_heatmap.unstack(level=0)
            sns.heatmap(
                df_heatmap_final.T,
                annot=True,
                fmt=".0f",
                cmap="YlGnBu",
                linewidths=.5,
                cbar_kws={'label': 'Average Vehicle Count'}
            )
            plt.title('Heatmap: Parking Density Comparison')
            plt.xlabel('Time Category and Day')
            plt.ylabel('Vehicle Type')
            plt.yticks(rotation=0)
            st.pyplot(plt.gcf())
        
    with tab4:
        st.subheader("24-Hour Line Graph (Alternative View)")
        st.info("24-hour synthetic line graph showing daily trend.")
        try:
            plot_load_24_hours(df)
        except Exception as e:
            st.warning(f"Failed to display 24-hour graph: {e}")

# --- Module 3: Modeling ---
def display_modeling(models_data):
    st.header("3️⃣ Tariff Potential Classification Modeling (Random Forest)")
    st.markdown("---")
    
    tab_moto, tab_car, tab_rekomendasi = st.tabs(["🏍️ Motorcycle Model", "🚗 Car Model", "📑 Tariff Recommendations"])

    def display_model_results(jenis, data):
        st.subheader(f"Training Results: {jenis} Model")
        
        model = data['model']
        y_test = data['y_test']
        y_pred = data['y_pred']
        le = data['le']
        cv_score = data['cv_score']

        if model is None:
            st.error(f"{jenis} model not trained - target class lacks sufficient variation.")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                acc = accuracy_score(y_test, y_pred)
                st.metric(f"{jenis} Model Accuracy", f"{acc*100:.2f} %")
                if cv_score is not None:
                    st.metric(f"Cross-Validation Score (5-Fold)", f"{cv_score*100:.2f} %")
                
                st.markdown("#### Confusion Matrix")
                fig, ax = plt.subplots(figsize=(6, 5))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
                ax.set_title('Confusion Matrix')
                ax.set_ylabel('Actual Class')
                ax.set_xlabel('Predicted Class')
                st.pyplot(fig)
            except ValueError as e:
                st.warning(f"Cannot calculate metrics: {e}")

        with col2:
            st.markdown("#### Classification Report")
            try:
                report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose().iloc[:-3, :-1]
                st.dataframe(report_df, use_container_width=True)
            except ValueError as e:
                st.warning(f"Cannot create classification report: {e}")

            st.markdown("#### Feature Importance")
            importance = pd.Series(model.feature_importances_, index=data['fitur']).sort_values(ascending=False).head(5)
            fig_imp, ax_imp = plt.subplots(figsize=(7, 5))
            sns.barplot(x=importance.values, y=importance.index, ax=ax_imp, palette='magma', hue=importance.index, legend=False)
            ax_imp.set_title(f'Top 5 Feature Importance - {jenis}')
            ax_imp.set_xlabel('Importance Score')
            st.pyplot(fig_imp)
            
    with tab_moto:
        display_model_results('Motorcycle', models_data['motorcycle'])
        
    with tab_car:
        display_model_results('Car', models_data['car'])

    with tab_rekomendasi:
        st.subheader("📑 Tariff Recommendation Policy Table")
        st.info("Displays tariff potential classification for each parking location based on trained models.")
        
        tarif_mapping = {
            'Motorcycle': {'Low': 1000, 'Medium': 2000, 'High': 3000},
            'Car': {'Low': 3000, 'Medium': 4000, 'High': 5000}
        }
        
        model_moto = models_data['motorcycle']['model']
        le_moto = models_data['motorcycle']['le']
        fitur_moto = models_data['motorcycle']['fitur']
        X_all_moto = models_data['motorcycle']['X_all']
        
        model_car = models_data['car']['model']
        le_car = models_data['car']['le']
        fitur_car = models_data['car']['fitur']
        X_all_car = models_data['car']['X_all']
        
        if model_moto is None or model_car is None:
            st.warning("Models unavailable or could not be trained.")
        else:
            try:
                # Predictions
                y_pred_m_enc = model_moto.predict(X_all_moto)
                df_result = pd.DataFrame(X_all_moto).reset_index(drop=True)
                
                df_result.insert(0, 'Parking Location', df_processed['Location Point'].values)
                
                df_result['Tariff Potential (Motorcycle)'] = le_moto.inverse_transform(y_pred_m_enc)
                df_result['Recommended Tariff (Motorcycle)'] = df_result['Tariff Potential (Motorcycle)'].apply(
                    lambda x: f"Rp{tarif_mapping['Motorcycle'].get(x, 0):,.0f}"
                )
                
                y_pred_c_enc = model_car.predict(X_all_car)
                df_result['Tariff Potential (Car)'] = le_car.inverse_transform(y_pred_c_enc)
                df_result['Recommended Tariff (Car)'] = df_result['Tariff Potential (Car)'].apply(
                    lambda x: f"Rp{tarif_mapping['Car'].get(x, 0):,.0f}"
                )
                
                kolom_output = ['Parking Location', 'Tariff Potential (Motorcycle)', 'Recommended Tariff (Motorcycle)', 
                               'Tariff Potential (Car)', 'Recommended Tariff (Car)']
                
                st.markdown("### Tariff Recommendation Summary (First 10 Rows)")
                st.dataframe(df_result[kolom_output].head(10), use_container_width=True)
                
                st.markdown("---")
                st.markdown("### Classification Distribution Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Motorcycle Distribution")
                    moto_dist = df_result['Tariff Potential (Motorcycle)'].value_counts()
                    st.bar_chart(moto_dist)
                
                with col2:
                    st.markdown("#### Car Distribution")
                    car_dist = df_result['Tariff Potential (Car)'].value_counts()
                    st.bar_chart(car_dist)
                
                st.markdown("---")
                st.markdown("### Complete Recommendation Table")
                st.dataframe(df_result[kolom_output], use_container_width=True)
                
                csv = df_result[kolom_output].to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="Tariff_Recommendations.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Failed to create recommendations: {e}")


# Module 4: Map & Simulation
def display_map_and_simulation(df_long, map_center, models_data, df_spasial):
    st.header("4️⃣ Map & Progressive Tariff Simulation")
    st.markdown("---")
    
    st.subheader("Static Tariff Potential Prediction Map")
    st.info("Location markers are fixed color. Use the dropdown below to select a parking location for simulation.")

    m = folium.Map(location=map_center, zoom_start=13, tiles='OpenStreetMap')

    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer(tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', name='Esri Satellite', attr='Esri').add_to(m)
    
    FIXED_COLOR = 'darkblue'

    fg_all = folium.FeatureGroup(name='All Parking Locations', show=True)
    features_search = []
    
    for index, row in df_spasial.iterrows():
        titik = row['Location Point']
        lat, lon = row['Latitude'], row['Longitude']
        
        motor_data = df_long[(df_long['titik'] == titik) & (df_long['jenis_kendaraan'] == 'Motorcycle')]
        mobil_data = df_long[(df_long['titik'] == titik) & (df_long['jenis_kendaraan'] == 'Car')]

        motor_row = motor_data.iloc[0] if not motor_data.empty else None
        mobil_row = mobil_data.iloc[0] if not mobil_data.empty else None

        motor_potensi = motor_row['kategori_load'].upper() if motor_row is not None else 'N/A'
        motor_tarif = int(motor_row['prediksi_tarif']) if motor_row is not None else 0
        mobil_potensi = mobil_row['kategori_load'].upper() if mobil_row is not None else 'N/A'
        mobil_tarif = int(mobil_row['prediksi_tarif']) if mobil_row is not None else 0

        popup_html = f"""
        <div style="font-size:13px; font-family:sans-serif;">
            <b>Parking Location:</b> {titik}<br>
            <b>Coordinates:</b> {lat:.4f}, {lon:.4f}<hr>
            <b>Motorcycle:</b> Potential {motor_potensi} (Base Tariff: Rp{motor_tarif:,})<br>
            <b>Car:</b> Potential {mobil_potensi} (Base Tariff: Rp{mobil_tarif:,})<br>
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
        name="Search Points (Hidden)",
        style_function=lambda x: {'opacity': 0, 'fillOpacity': 0, 'weight': 0, 'color': 'transparent'},
    ).add_to(m)

    Search(
        layer=geojson_layer_search, 
        search_label="name", 
        placeholder="Search parking location...", 
        collapsed=False, 
        position="topleft", 
        geom_type="Point",
    ).add_to(m)
    
    folium.LayerControl(collapsed=False).add_to(m)
    Fullscreen(position='topright').add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    
    folium_static(m, width=None, height=550)

    # --- Simulation Section ---
    st.subheader("Tariff Potential Simulation (What-If Analysis)")
    st.markdown("**1. Select Parking Location** (Location aggregate data will be used as default)")
    
    selected_titik = st.selectbox(
        "Select Parking Location for Simulation:", 
        df_spasial['Location Point'].unique().tolist(), 
        key='sim_titik_select'
    )
    
    default_data = df_spasial[df_spasial['Location Point'] == selected_titik].iloc[0]
    
    default_jam_val = 9.0
    for col in default_data.index:
        if 'Peak Hours' in col and 'Motorcycle' in col and 'Weekday' in col:
            default_jam_val = default_data[col]
            break

    with st.expander(f"⚙️ Configure Scenario for {selected_titik} ⚙️"):
        st.markdown(f"**Location Default Aggregate Data:**")
        st.markdown(f"* Motorcycles Weekday: **{default_data.get('Number of Motorcycles (Weekday)', 0):.0f}** units, Cars Weekday: **{default_data.get('Number of Cars (Weekday)', 0):.0f}** units")
        st.markdown(f"* Peak Hours Weekday Motorcycle: **{default_jam_val:.2f}**")

        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1: 
            jenis = st.selectbox("Vehicle Type", ['Motorcycle', 'Car'], key='sim_jenis')
        with col2: 
            hari = st.selectbox("Day", ['Weekday', 'Weekend'], key='sim_hari')
        with col3: 
            jam_for_time_input = default_jam_val
            try:
                time_obj_default = datetime.time(int(jam_for_time_input // 1), int((jam_for_time_input % 1) * 60))
            except ValueError:
                time_obj_default = datetime.time(9, 0)
                
            time_obj = st.time_input(
                "Time (HH:MM)", 
                value=time_obj_default, 
                step=datetime.timedelta(minutes=1), 
                key='sim_jam_time'
            )
            jam_desimal_input = time_to_decimal_hour(time_obj) 
            st.caption(f"Model Hour Value: **{jam_desimal_input:.2f}**") 
            
        with col4: 
            default_jumlah = default_data.get(f'Number of {jenis} ({hari})', 100)
            jumlah_input = st.number_input(f"Vehicle Count ({jenis})", min_value=1, max_value=500, value=int(default_jumlah), key='sim_jumlah')
        with col5: 
            st.markdown("<br>", unsafe_allow_html=True) 
            submitted = st.button("Predict Result 🚀", key='sim_submit', type='primary')

        if submitted:
            jenis_key = 'motorcycle' if jenis == 'Motorcycle' else 'car'
            data = models_data[jenis_key]
            
            pred_class, confidence, top_gain, proba_dict, keterangan_jam = predict_single_input(
                jenis, hari, jam_desimal_input, jumlah_input, data['model'], data['le'], data['X_ref']
            )
            
            if not isinstance(pred_class, str) or "Error" in pred_class or "Failed" in pred_class:
                st.error(f"Simulation Failed: {pred_class}. Ensure model is trained.")
            else:
                rekomendasi_tarif_dasar = tarif_mapping[jenis].get(pred_class, 0)
                rekomendasi_tarif_progresif = calculate_progresif_tarif(jenis, pred_class, jam_desimal_input)
                
                st.markdown("---")
                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Tariff Potential Class (Simulation)", f"Potential {pred_class.upper()}", delta=f"Confidence: {confidence:.3f}")
                col_res2.metric("Recommended Base Tariff", f"Rp{rekomendasi_tarif_dasar:,}", delta=f"Class: {pred_class}")
                
                col_res3.metric("Recommended PROGRESSIVE Tariff", f"Rp{rekomendasi_tarif_progresif:,}", delta=f"Increase: Rp{rekomendasi_tarif_progresif - rekomendasi_tarif_dasar:,}")
                
                st.markdown("---")
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.markdown("**Time Logic Explanation:**")
                    st.info(keterangan_jam)
                    st.markdown("**Top 3 Contributors (Local Gain):**")
                    if isinstance(top_gain, pd.Series):
                        for f in top_gain.index: 
                            st.markdown(f"- **{f}** (Main predictor)")
                    st.caption(f"Probability All Classes: {proba_dict}")

                with col_info2:
                    st.markdown("**Progressive Logic Applied:**")
                    st.warning(f"If **Hour > 9.00**, **{pred_class}** tariff increases by **Rp{rekomendasi_tarif_progresif - rekomendasi_tarif_dasar:,}** from base tariff.")


# =================================================================
# === MAIN APPLICATION EXECUTION ===
# =================================================================

df_processed, df_spasial, jam_cols, df_raw, batas_kuantil = load_and_preprocess_data(FILE_PATH)
if df_processed is None: 
    st.error(f"Failed to load data from '{FILE_PATH}'. Ensure file exists and contains required spatial columns.")
    st.stop()

models_data = train_models(df_processed, jam_cols)

# Static predictions for map
df_long = pd.DataFrame()
try:
    if models_data['motorcycle']['model']:
        df_processed['Pred_Class_Motorcycle'] = models_data['motorcycle']['le'].inverse_transform(models_data['motorcycle']['model'].predict(models_data['motorcycle']['X_all']))
    else:
        df_processed['Pred_Class_Motorcycle'] = df_processed['Class_Motorcycle']
        
    if models_data['car']['model']:
        df_processed['Pred_Class_Car'] = models_data['car']['le'].inverse_transform(models_data['car']['model'].predict(models_data['car']['X_all']))
    else:
        df_processed['Pred_Class_Car'] = df_processed['Class_Car']

    df_mapping = df_spasial.dropna(subset=['Latitude', 'Longitude'])

    df_moto_map = df_mapping.copy()
    df_moto_map['jenis_kendaraan'] = 'Motorcycle'
    df_moto_map['kategori_load'] = df_processed['Pred_Class_Motorcycle']
    df_moto_map['prediksi_tarif'] = df_processed['Pred_Class_Motorcycle'].apply(lambda x: tarif_mapping['Motorcycle'].get(x, 0))
    df_moto_map.rename(columns={'Location Point': 'titik', 'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True) 

    df_car_map = df_mapping.copy()
    df_car_map['jenis_kendaraan'] = 'Car'
    df_car_map['kategori_load'] = df_processed['Pred_Class_Car']
    df_car_map['prediksi_tarif'] = df_processed['Pred_Class_Car'].apply(lambda x: tarif_mapping['Car'].get(x, 0))
    df_car_map.rename(columns={'Location Point': 'titik', 'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True) 

    df_long = pd.concat([df_moto_map, df_car_map], ignore_index=True).dropna(subset=['latitude', 'longitude']) 

    if not df_long.empty:
        map_center = [df_long['latitude'].mean(), df_long['longitude'].mean()]
    else:
        map_center = [-7.4168, 109.2155]
        
except Exception as e:
    st.error(f"Error preparing map/prediction data: {e}")
    map_center = [-7.4168, 109.2155]

st.title("🅿️ Parking Tariff Potential Analysis")
st.caption("Dashboard for parking tariff potential classification modeling using Random Forest.")
st.markdown("---")

# Main navigation
with st.sidebar:
    st.markdown("---")
    
    page = option_menu(
        menu_title="Analytics Dashboard 📊", 
        options=["Data Table", "Visualization", "Modeling", "Map & Simulation"],
        icons=["table", "bar-chart", "calculator", "geo-alt"], 
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
    )

# Display logic
if page == "Data Table":
    display_data_table(df_raw, df_processed)

elif page == "Visualization":
    display_visualization(df_processed, batas_kuantil, jam_cols)

elif page == "Modeling":
    display_modeling(models_data)

elif page == "Map & Simulation":
    display_map_and_simulation(df_long, map_center, models_data, df_spasial)
