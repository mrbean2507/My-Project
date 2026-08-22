import streamlit as st
import pandas as pd
import pickle
import optuna

# Cấu hình giao diện trang web (wide layout)
st.set_page_config(page_title="Astaxanthin Extraction System", layout="wide")

st.title("🧪 Hệ Thống Dự Đoán & Tối Ưu Hóa Chiết Xuất Astaxanthin")
st.markdown("Nền tảng mô hình hóa quá trình trích ly dựa trên học máy (Machine Learning Surrogate Models).")

# 1. Tải các mô hình .pkl độc lập cho từng phương pháp
@st.cache_resource
def load_models():
    try:
        with open("trained_solvent_model.pkl", "rb") as f:
            model_sol = pickle.load(f)
    except:
        model_sol = None
        
    try:
        with open("trained_ultrasound_model.pkl", "rb") as f:
            model_us = pickle.load(f)
    except:
        model_us = None
        
    return model_sol, model_us

model_solvent, model_ultrasound = load_models()

# Chia ứng dụng thành 3 Tab chức năng riêng biệt
tab_pred_sol, tab_pred_us, tab_opt = st.tabs([
    "🧪 1. Dự Đoán Dung Môi (Solvent Extraction)", 
    "🔊 2. Dự Đoán Siêu Âm (Ultrasound-Assisted Extraction)", 
    "⚙️ 3. Tối Ưu Hóa (Optimization)"
])

# Khai báo biến
Solvent_type_no = [
"Acetone", 
"Ethanol", 
"Hexane", 
"Palm olein", 
"Crude viscera oil (CVO)", 
"Total fatty acids ethyl esters (TFA)",
"Polyunsaturatedfatty acidethylesters (PUFAE)"
]
Shrimp_species_no = [
"Farfantepenaeus subtilis", 
"Parapenaeus longirostris",
"Penaeus semisulcatus",
"Pandalus borealis"
]
Solvent_type_uae = []
# ==========================================
# TAB 1: DỰ ĐOÁN TRÍCH LY DUNG MÔI (SOLVENT EXTRACTION)
# ==========================================
with tab_pred_sol:
    st.header("Nhập thông số cho phương pháp Trích ly Dung môi")
    
    if model_solvent is None:
        st.warning("⚠️ Không tìm thấy tệp mô hình `trained_solvent_model.pkl`.")
    
    col1, col2 = st.columns(2)
    with col1:
        # Các thông số là "tên" dùng nút chọn (selectbox)
        sol_solvent = st.selectbox("Chọn loại dung môi (Solvent):", Solvent_type_no, key="sol_solvent")
        sol_shrimp = st.selectbox("Chọn loài tôm (Shrimp Species):", Shrimp_species_no, key="sol_shrimp")
        
    with col2:
        # Các thông số là giá trị số dùng ô nhập liệu trực tiếp (number_input)
        sol_temp = st.number_input("Nhiệt độ chiết xuất (°C) [Temperature]:", min_value=0.0, max_value=80.0, value=50.0, step=0.5, key="sol_temp")
        sol_time = st.number_input("Thời gian chiết xuất (phút) [Time]:", min_value=0.0, max_value=600.0, value=30.0, step=1.0, key="sol_time")
        sol_ratio = st.number_input("Tỷ lệ dung môi / nguyên liệu [Ratio]:", min_value=0.0, max_value=100.0, value=20.0, step=0.5, key="sol_ratio")
        
    if st.button("🚀 Dự Đoán Hiệu Suất Dung Môi", type="primary"):
        if model_solvent is not None:
            input_sol = pd.DataFrame({
                'Solvent': [sol_solvent],
                'Shrimp_Species': [sol_shrimp],
                'Temperature': [sol_temp],
                'Time': [sol_time],
                'Ratio': [sol_ratio]
            })
            pred_res = model_solvent.predict(input_sol)
            y_pred = pred_res['Yield_pred'].values[0] if isinstance(pred_res, pd.DataFrame) else pred_res[0]
            y_sd = pred_res['Yield_sd'].values[0] if isinstance(pred_res, pd.DataFrame) else 0.0
            
            st.success("Dự đoán hoàn tất!")
            m1, m2 = st.columns(2)
            m1.metric("🎯 Hiệu suất dự đoán (Yield_pred)", value=f"{y_pred:.2f} µg/g")
            m2.metric("📊 Độ bất định (Yield_sd)", value=f"± {y_sd:.2f} µg/g")
        else:
            st.error("Chưa tải được mô hình trích ly dung môi.")

# ==========================================
# TAB 2: DỰ ĐOÁN TRÍCH LY SIÊU ÂM (ULTRASOUND EXTRACTION)
# ==========================================
with tab_pred_us:
    st.header("Nhập thông số cho phương pháp Trích ly Hỗ trợ Siêu âm (UAE)")
    
    if model_ultrasound is None:
        st.warning("⚠️ Không tìm thấy tệp mô hình `trained_ultrasound_model.pkl`.")
        
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        # Các thông số tên dùng nút chọn
        us_solvent = st.selectbox("Chọn loại dung môi (Solvent):", ["Acetone", "Ethanol", "Hexane"], key="us_solvent")
        us_shrimp = st.selectbox("Chọn loài tôm (Shrimp Species):", ["Farfantepenaeus subtilis", "Parapenaeus longirostris"], key="us_shrimp")
        
    with col_u2:
        # Các thông số số học dùng ô nhập liệu trực tiếp
        us_frequency = st.number_input("Tần số (kHz) [Frequency]:", min_value=20.0, max_value=40.0, value=20.0, step=1.0, key="us_frequency")
        us_temp = st.number_input("Nhiệt độ (°C) [Temperature]:", min_value=0.0, max_value=80.0, value=40.0, step=1.0, key="us_temp")
        us_time = st.number_input("Thời gian siêu âm (phút) [Time]:", min_value=0.0, max_value=120.0, value=0.0, step=1.0, key="us_time")
        us_ratio = st.number_input("Tỷ lệ dung môi / nguyên liệu (ml/g) [Ratio]:", min_value=0.0, max_value=100.0, value=0.0, step=0.5, key="us_ratio")
        us_amplitude = st.number_input("Biên độ sóng (%) ", min_value=0.0, max_value=100.0, value=100.0, step=10.0, key="us_amplitude")
    if st.button("🚀 Dự Đoán Hiệu Suất Siêu Âm", type="primary"):
        if model_ultrasound is not None:
            input_us = pd.DataFrame({
                'Solvent': [us_solvent],
                'Shrimp Species': [us_shrimp],
                'Frequency': [us_frequency],
                'Temperature': [us_temp],
                'Time': [us_time],
                'Ratio': [us_ratio],
                'Amplitude': [us_amplitude]
            })
            pred_res_us = model_ultrasound.predict(input_us)
            y_pred_us = pred_res_us['Yield_pred'].values[0] if isinstance(pred_res_us, pd.DataFrame) else pred_res_us[0]
            y_sd_us = pred_res_us['Yield_sd'].values[0] if isinstance(pred_res_us, pd.DataFrame) else 0.0
            
            st.success("Dự đoán siêu âm hoàn tất!")
            mu1, mu2 = st.columns(2)
            mu1.metric("🎯 Hiệu suất dự đoán siêu âm", value=f"{y_pred_us:.2f} µg/g")
            mu2.metric("📊 Độ bất định (Yield_sd)", value=f"± {y_sd_us:.2f} µg/g")
        else:
            st.error("Chưa tải được mô hình trích ly siêu âm.")

# ==========================================
# TAB 3: TỐI ƯU HÓA (OPTIMIZATION)
# ==========================================
with tab_opt:
    st.header("Tối ưu hóa thông số công nghệ (Optuna Optimization)")
    opt_choice = st.radio("Chọn quy trình cần tối ưu:", ["Trích ly Dung môi", "Trích ly Siêu âm"])
    
    n_trials = st.number_input("Số vòng lặp thử nghiệm (n_trials):", min_value=10, max_value=500, value=100, step=10)
    
    if st.button("⚙️ Bắt đầu Tối Ưu Hóa", type="primary"):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective_sol(trial):
            s = trial.suggest_categorical('Solvent', ['Acetone', 'Ethanol', 'Hexane'])
            sh = trial.suggest_categorical('Shrimp_Species', ['Farfantepenaeus subtilis', 'Parapenaeus longirostris'])
            t = trial.suggest_float('Temperature', 30.0, 80.0)
            ti = trial.suggest_float('Time', 10.0, 120.0)
            r = trial.suggest_float('Ratio', 10.0, 50.0)
            df = pd.DataFrame({'Solvent': [s], 'Shrimp_Species': [sh], 'Temperature': [t], 'Time': [ti], 'Ratio': [r]})
            res = model_solvent.predict(df)
            return res['Yield_pred'].values[0] if isinstance(res, pd.DataFrame) else res[0]

        def objective_us(trial):
            s = trial.suggest_categorical('Solvent', ['Acetone', 'Ethanol', 'Hexane'])
            sh = trial.suggest_categorical('Shrimp Species', ['Farfantepenaeus subtilis', 'Parapenaeus longirostris'])
            f = trial.suggest_float('Frequency', 20.0, 40.0)
            t = trial.suggest_float('Temperature', 0, 80.0)
            ti = trial.suggest_float('Time', 0, 120.0)
            r = trial.suggest_float('Ratio', 0, 100.0)
            df = pd.DataFrame({'Solvent': [s], 'Shrimp Species': [sh], 'Frequency': [f], 'Temperature': [t], 'Time': [ti], 'Ratio': [r]})
            res = model_ultrasound.predict(df)
            return res['Yield_pred'].values[0] if isinstance(res, pd.DataFrame) else res[0]

        with st.spinner("Đang chạy thuật toán tối ưu hóa Bayes..."):
            study = optuna.create_study(direction='maximize')
            if opt_choice == "Trích ly Dung môi":
                if model_solvent is not None:
                    study.optimize(objective_sol, n_trials=int(n_trials))
                    best_df = pd.DataFrame([study.best_params])
                    best_pred = model_solvent.predict(best_df)
                else:
                    st.error("Thiếu mô hình dung môi.")
            else:
                if model_ultrasound is not None:
                    study.optimize(objective_us, n_trials=int(n_trials))
                    best_df = pd.DataFrame([study.best_params])
                    best_pred = model_ultrasound.predict(best_df)
                else:
                    st.error("Thiếu mô hình siêu âm.")
            
            if 'study' in locals() and len(study.trials) > 0:
                best_y = best_pred['Yield_pred'].values[0] if isinstance(best_pred, pd.DataFrame) else best_pred[0]
                st.success("Tối ưu hóa thành công!")
                st.metric("🏆 Hiệu suất cực đại tối ưu", value=f"{best_y:.2f} µg/g")
                st.subheader("⚙️ Bộ thông số tối ưu:")
                st.json(study.best_params)