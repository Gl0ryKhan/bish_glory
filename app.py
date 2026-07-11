import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Bish Glory", page_icon="🏠", layout="wide")

# ---------- Стили ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9a9a9a;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .price-card {
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 1px solid #ff8a00;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .price-value {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .price-label {
        color: #9a9a9a;
        font-size: 0.95rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    div[data-testid="stMetric"] {
        background-color: #1a1a2e;
        border-radius: 12px;
        padding: 10px;
    }
    .stButton>button, .stFormSubmitButton>button {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.2rem;
        font-weight: 700;
        width: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 0 0 rgba(229, 46, 113, 0.4);
        animation: pulse 2.5s infinite;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(229, 46, 113, 0.6);
    }
    .stButton>button:active, .stFormSubmitButton>button:active {
        transform: scale(0.98);
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 138, 0, 0.4); }
        70% { box-shadow: 0 0 0 12px rgba(255, 138, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 138, 0, 0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .price-card {
        animation: fadeInUp 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🏠 Bish Glory</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Предсказание рыночной цены квартиры в Бишкеке</p>', unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("final_catboost_model.cbm")
    return model


model = load_model()

CAT_FEATURES = ["Серия", "Отопление", "Состояние", "house_type", "district"]

SERIYA_OPTIONS = [
    "элитка", "индивид. планировка", "104 серия", "105 серия", "106 серия",
    "хрущевка", "108 серия", "малосемейка", "сталинка", "пентхаус", "ост. серии",
]
OTOPLENIE_OPTIONS = ["центральное", "на газе", "не указано", "автономное", "электрическое", "другое"]
SOSTOYANIE_OPTIONS = ["евроремонт", "под самоотделку (псо)", "не указано", "хорошее", "среднее", "не достроено"]
HOUSE_TYPE_OPTIONS = ["монолитный", "кирпичный", "панельный"]

# Ориентиры для быстрого позиционирования на карте (приблизительные координаты)
LANDMARKS = {
    "Центр (Ала-Тоо)": (42.8746, 74.6122),
    "Восток-5": (42.8460, 74.6520),
    "Джал": (42.8580, 74.6350),
    "Асанбай": (42.8890, 74.6050),
    "Аламедин-1": (42.8830, 74.6650),
    "Filaret / Юг-2": (42.8280, 74.5980),
}

COLUMN_ORDER = [
    "lat", "lon", "Серия", "Отопление", "Состояние", "rooms", "square",
    "is_free_layout", "house_type", "build_year",
    "Тип предложения_от агента", "Тип предложения_от собственника",
    "floor_ratio", "is_first_floor", "is_last_floor",
    "doc_ddu", "doc_tech_passport", "doc_red_book", "doc_sale_purchase",
    "district", "build_year_is_missing",
]

if "lat" not in st.session_state:
    st.session_state.lat = 42.8746
if "lon" not in st.session_state:
    st.session_state.lon = 74.6122

# ---------- Карта ----------
st.markdown("### 📍 Укажите расположение квартиры на карте")

quick_col1, quick_col2, quick_col3 = st.columns([2, 1, 1])
with quick_col1:
    landmark = st.selectbox("Быстрый переход к району", ["—"] + list(LANDMARKS.keys()))
    if landmark != "—":
        st.session_state.lat, st.session_state.lon = LANDMARKS[landmark]
with quick_col2:
    manual_lat = st.number_input(
        "Широта (lat)", value=float(st.session_state.lat), format="%.5f", key="lat_input"
    )
with quick_col3:
    manual_lon = st.number_input(
        "Долгота (lon)", value=float(st.session_state.lon), format="%.5f", key="lon_input"
    )

if (manual_lat, manual_lon) != (st.session_state.lat, st.session_state.lon):
    st.session_state.lat, st.session_state.lon = manual_lat, manual_lon
    st.rerun()

m = folium.Map(
    location=[st.session_state.lat, st.session_state.lon],
    zoom_start=13,
    tiles="OpenStreetMap",
)
folium.Marker(
    [st.session_state.lat, st.session_state.lon],
    tooltip="Ваша квартира",
    icon=folium.Icon(color="orange", icon="home", prefix="fa"),
).add_to(m)

map_data = st_folium(m, height=420, use_container_width=True, key="map")

if map_data and map_data.get("last_clicked"):
    new_lat = map_data["last_clicked"]["lat"]
    new_lon = map_data["last_clicked"]["lng"]
    if (new_lat, new_lon) != (st.session_state.lat, st.session_state.lon):
        st.session_state.lat, st.session_state.lon = new_lat, new_lon
        st.rerun()

st.caption(f"Выбрано (lat, lon): {st.session_state.lat:.5f}, {st.session_state.lon:.5f} — кликните по карте или введите вручную")

st.markdown("---")

# ---------- Параметры квартиры ----------
with st.form("prediction_form"):
    tab1, tab2, tab3 = st.tabs(["🏢 Квартира", "🏗️ Дом и этаж", "📄 Документы"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            seriya = st.selectbox("Серия", SERIYA_OPTIONS, index=SERIYA_OPTIONS.index("элитка"))
            sostoyanie = st.selectbox("Состояние", SOSTOYANIE_OPTIONS, index=SOSTOYANIE_OPTIONS.index("евроремонт"))
            rooms = st.number_input("Кол-во комнат", min_value=1, value=2, step=1)
        with c2:
            district = st.text_input("Район (название)", value="Магистраль")
            otoplenie = st.selectbox("Отопление", OTOPLENIE_OPTIONS, index=OTOPLENIE_OPTIONS.index("центральное"))
            square = st.number_input("Площадь (м²)", min_value=1.0, value=60.0)
        is_free_layout = st.checkbox("Свободная планировка")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            house_type = st.selectbox("Тип дома", HOUSE_TYPE_OPTIONS, index=HOUSE_TYPE_OPTIONS.index("монолитный"))
            build_year = st.number_input("Год постройки (0, если неизвестен)", min_value=0, value=2015, step=1)
        with c2:
            floor = st.number_input("Этаж квартиры", min_value=1, value=3, step=1)
            total_floors = st.number_input("Этажность дома", min_value=1, value=9, step=1)
        tip_predlozheniya = st.radio("Тип предложения", ["от агента", "от собственника"], index=1, horizontal=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            doc_ddu = st.checkbox("Есть ДДУ")
            doc_tech_passport = st.checkbox("Есть техпаспорт")
        with c2:
            doc_red_book = st.checkbox("Есть красная книга")
            doc_sale_purchase = st.checkbox("Есть договор купли-продажи")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("💰 Предсказать цену")

# ---------- Предсказание ----------
if submitted:
    with st.spinner("Считаем цену..."):
        if total_floors and total_floors > 0:
            floor_ratio = floor / total_floors
        else:
            floor_ratio = np.nan

        is_first_floor = 1 if floor == 1 else 0
        is_last_floor = 1 if total_floors and floor == total_floors else 0

        build_year_is_missing = 1 if not build_year or build_year <= 0 else 0
        build_year_val = np.nan if build_year_is_missing else build_year

        ot_agenta = 1 if tip_predlozheniya == "от агента" else 0
        ot_sobstvennika = 1 if tip_predlozheniya == "от собственника" else 0

        row = {
            "lat": st.session_state.lat,
            "lon": st.session_state.lon,
            "Серия": str(seriya),
            "Отопление": str(otoplenie),
            "Состояние": str(sostoyanie),
            "rooms": rooms,
            "square": square,
            "is_free_layout": int(is_free_layout),
            "house_type": str(house_type),
            "build_year": build_year_val,
            "Тип предложения_от агента": ot_agenta,
            "Тип предложения_от собственника": ot_sobstvennika,
            "floor_ratio": floor_ratio,
            "is_first_floor": is_first_floor,
            "is_last_floor": is_last_floor,
            "doc_ddu": int(doc_ddu),
            "doc_tech_passport": int(doc_tech_passport),
            "doc_red_book": int(doc_red_book),
            "doc_sale_purchase": int(doc_sale_purchase),
            "district": str(district),
            "build_year_is_missing": build_year_is_missing,
        }

        X = pd.DataFrame([row])[COLUMN_ORDER]
        for col in CAT_FEATURES:
            X[col] = X[col].astype(str)

        pred_log = model.predict(X)[0]
        price_usd = np.expm1(pred_log)
        price_per_m2 = price_usd / square if square else 0

    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">Предсказанная рыночная цена</div>
        <div class="price-value">${price_usd:,.0f}</div>
        <div class="price-label">~${price_per_m2:,.0f} за м²</div>
    </div>
    """, unsafe_allow_html=True)
    st.balloons()
