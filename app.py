import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & TỪ ĐIỂN DỊCH THUẬT
# ==========================================
st.set_page_config(page_title="AI Dự báo Thời tiết", page_icon="🌤️", layout="wide")

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🌤️ HỆ THỐNG TRÍ TUỆ NHÂN TẠO DỰ BÁO NHIỆT ĐỘ</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: gray;'>Sử dụng thuật toán học máy XGBoost Regression</p>", unsafe_allow_html=True)
st.divider()

# Từ điển dịch mở rộng (Quét sạch các trường hợp thời tiết)
dict_summary = {
    'Clear': 'Trời quang đãng',
    'Partly Cloudy': 'Có mây rải rác',
    'Mostly Cloudy': 'Trời nhiều mây',
    'Overcast': 'Trời u ám',
    'Foggy': 'Có sương mù',
    'Breezy and Overcast': 'Gió nhẹ & U ám',
    'Breezy and Mostly Cloudy': 'Gió nhẹ & Nhiều mây',
    'Breezy and Partly Cloudy': 'Gió nhẹ & Có mây rải rác',
    'Dry': 'Trời khô hanh',
    'Windy and Partly Cloudy': 'Nhiều gió & Có mây rải rác',
    'Windy and Overcast': 'Nhiều gió & U ám',
    'Breezy': 'Có gió nhẹ',
    'Windy': 'Nhiều gió',
    'Breezy and Foggy': 'Gió nhẹ & Có sương mù',
    'Windy and Foggy': 'Nhiều gió & Có sương mù',
    'Humid and Mostly Cloudy': 'Nồm ẩm & Nhiều mây',
    'Humid and Partly Cloudy': 'Nồm ẩm & Có mây rải rác',
    'Humid and Overcast': 'Nồm ẩm & U ám',
    'Light Rain': 'Mưa nhỏ',
    'Drizzle': 'Mưa phùn',
    'Rain': 'Có mưa',
    'Dangerously Windy and Partly Cloudy': 'Gió giật mạnh & Có mây'
}

dict_precip = {
    'rain': 'Mưa 🌧️',
    'snow': 'Tuyết ❄️'
}

# ==========================================
# 2. DẠY AI TRONG HẬU TRƯỜNG
# ==========================================
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('data/raw/weatherHistory.csv')
    df['Precip Type'] = df['Precip Type'].fillna(df['Precip Type'].mode()[0])
    
    le_sum = LabelEncoder()
    le_prec = LabelEncoder()
    df['Summary_Encoded'] = le_sum.fit_transform(df['Summary'])
    df['Precip_Type_Encoded'] = le_prec.fit_transform(df['Precip Type'])
    
    X = df[['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)', 'Summary_Encoded', 'Precip_Type_Encoded']]
    y = df['Temperature (C)']
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, le_sum, le_prec

model, le_sum, le_prec = load_and_train_model()

# ==========================================
# 3. GIAO DIỆN ĐIỀU KHIỂN BÊN TRÁI
# ==========================================
col1, col2 = st.columns([1, 1])

with col1:
    st.header("⚙️ Nhập thông số thời tiết")
    with st.container(border=True):
        # Đã xóa sạch đuôi tiếng Anh gân vướng mắt
        humidity = st.slider("💧 Độ ẩm", 0.0, 1.0, 0.7)
        wind_speed = st.slider("💨 Tốc độ gió (km/h)", 0.0, 60.0, 15.0)
        pressure = st.number_input("🧭 Áp suất (mbar)", 900.0, 1100.0, 1015.0)
        
        summary = st.selectbox("☁️ Kiểu thời tiết", le_sum.classes_, format_func=lambda x: dict_summary.get(x, x))
        precip_type = st.selectbox("🌧️ Loại lượng mưa", le_prec.classes_, format_func=lambda x: dict_precip.get(x, x))

with col2:
    st.header("🤖 Kết quả từ AI")
    st.markdown("Nhấn nút bên dưới để mô hình XGBoost thực hiện dự báo:")
    
    if st.button("🚀 KÍCH HOẠT AI DỰ BÁO", use_container_width=True, type="primary"):
        sum_enc = le_sum.transform([summary])[0]
        prec_enc = le_prec.transform([precip_type])[0]
        input_data = pd.DataFrame([[humidity, wind_speed, pressure, sum_enc, prec_enc]],
                                  columns=['Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)', 'Summary_Encoded', 'Precip_Type_Encoded'])
        
        prediction = model.predict(input_data)[0]
        
        st.divider()
        st.metric(label="🌡️ Nhiệt độ dự báo lúc này là:", value=f"{prediction:.2f} °C", delta="Mô hình XGBoost")
        st.balloons()