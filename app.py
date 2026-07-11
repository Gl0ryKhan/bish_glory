import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

st.set_page_config(page_title="Bish Glory", page_icon="🏠")


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

# Порядок колонок должен строго соответствовать тому, на чём обучалась модель
COLUMN_ORDER = [
    "lat", "lon", "Серия", "Отопление", "Состояние", "rooms", "square",
    "is_free_layout", "house_type", "build_year",
    "Тип предложения_от агента", "Тип предложения_от собственника",
    "floor_ratio", "is_first_floor", "is_last_floor",
    "doc_ddu", "doc_tech_passport", "doc_red_book", "doc_sale_purchase",
    "district", "build_year_is_missing",
]

st.title("🏠 Bish Glory — предсказание цены квартиры в Бишкеке")
st.markdown("Введите параметры квартиры, модель CatBoost предскажет рыночную стоимость в USD.")

col1, col2 = st.columns(2)

with col1:
    lat = st.number_input("Широта (lat)", value=42.8746, format="%.6f")
    lon = st.number_input("Долгота (lon)", value=74.6122, format="%.6f")
    seriya = st.selectbox("Серия", SERIYA_OPTIONS, index=SERIYA_OPTIONS.index("элитка"))
    otoplenie = st.selectbox("Отопление", OTOPLENIE_OPTIONS, index=OTOPLENIE_OPTIONS.index("центральное"))
    sostoyanie = st.selectbox("Состояние", SOSTOYANIE_OPTIONS, index=SOSTOYANIE_OPTIONS.index("евроремонт"))
    rooms = st.number_input("Кол-во комнат", min_value=1, value=2, step=1)
    square = st.number_input("Площадь (м²)", min_value=1.0, value=60.0)
    is_free_layout = st.checkbox("Свободная планировка")
    house_type = st.selectbox("Тип дома", HOUSE_TYPE_OPTIONS, index=HOUSE_TYPE_OPTIONS.index("монолитный"))

with col2:
    build_year = st.number_input("Год постройки (0, если неизвестен)", min_value=0, value=2015, step=1)
    tip_predlozheniya = st.radio("Тип предложения", ["от агента", "от собственника"], index=1)
    floor = st.number_input("Этаж квартиры", min_value=1, value=3, step=1)
    total_floors = st.number_input("Этажность дома", min_value=1, value=9, step=1)
    doc_ddu = st.checkbox("Есть ДДУ")
    doc_tech_passport = st.checkbox("Есть техпаспорт")
    doc_red_book = st.checkbox("Есть красная книга")
    doc_sale_purchase = st.checkbox("Есть договор купли-продажи")
    district = st.text_input("Район", value="Магистраль")

if st.button("Предсказать цену", type="primary"):
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
        "lat": lat,
        "lon": lon,
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

    st.success(f"💰 Предсказанная цена: ${price_usd:,.0f}")
