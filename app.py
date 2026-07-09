import gradio as gr
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

model = CatBoostRegressor()
model.load_model("final_catboost_model.cbm")

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


def predict(
    lat, lon, seriya, otoplenie, sostoyanie, rooms, square,
    is_free_layout, house_type, build_year,
    tip_predlozheniya,
    floor, total_floors,
    doc_ddu, doc_tech_passport, doc_red_book, doc_sale_purchase,
    district,
):
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

    return f"💰 Предсказанная цена: ${price_usd:,.0f}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Широта (lat)", value=42.8746),
        gr.Number(label="Долгота (lon)", value=74.6122),
        gr.Dropdown(SERIYA_OPTIONS, label="Серия", value="элитка"),
        gr.Dropdown(OTOPLENIE_OPTIONS, label="Отопление", value="центральное"),
        gr.Dropdown(SOSTOYANIE_OPTIONS, label="Состояние", value="евроремонт"),
        gr.Number(label="Кол-во комнат", value=2),
        gr.Number(label="Площадь (м²)", value=60),
        gr.Checkbox(label="Свободная планировка"),
        gr.Dropdown(HOUSE_TYPE_OPTIONS, label="Тип дома", value="монолитный"),
        gr.Number(label="Год постройки (0, если неизвестен)", value=2015),
        gr.Radio(["от агента", "от собственника"], label="Тип предложения", value="от собственника"),
        gr.Number(label="Этаж квартиры", value=3),
        gr.Number(label="Этажность дома", value=9),
        gr.Checkbox(label="Есть ДДУ"),
        gr.Checkbox(label="Есть техпаспорт"),
        gr.Checkbox(label="Есть красная книга"),
        gr.Checkbox(label="Есть договор купли-продажи"),
        gr.Textbox(label="Район", value="Магистраль"),
    ],
    outputs=gr.Text(label="Результат"),
    title="🏠 Bish Glory — предсказание цены квартиры в Бишкеке",
    description="Введите параметры квартиры, модель CatBoost предскажет рыночную стоимость в USD.",
)

if __name__ == "__main__":
    demo.launch()
