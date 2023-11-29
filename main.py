"""Сервис для предсказания стоимости автомобилей"""
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sklearn.linear_model import Ridge


MODEL_PARAMS_PATH = './ml-vse/model.pkl'
app = FastAPI()


class Item(BaseModel):
    """Класс с данными одного автомобиля"""
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

    def to_dict(self):
        """Возвращает словарь с данными экземпляра класса"""
        return self.__dict__


class Items(BaseModel):
    """Класс для списка автомобилей"""
    objects: List[Item]


def prepare_data(
    df: pd.DataFrame,
    model_data: Dict[str, Any]
) -> pd.DataFrame:
    """
    Метод для предобработки датафрейма перед применением модели
    param: df: pd.Dataframe - датафрейм с данными автомобилей
    model_data: Dict[str, Any] - словарь с данными предобученной модели
    return: pd.DataFrame - обработанный датафрейм
    """

    def get_torque_coef(row: pd.Series) -> float:
        """
        Метод для получения коэффициентов для перевода крутящего момента
        к одной единице измерения (nm)
        param: row: pd.Series - строка датафрейма с данными автомобилей
        return: float - коэффициент для перевода в nm
        """
        if not isinstance(row['torque'], str):
            return 1

        torque_lower = row['torque'].lower()
        if 'nm' in torque_lower or 'kgm' not in torque_lower:
            return 1

        if row['torque_2'] > 100:
            return 1

        return 9.80665

    def get_number_from_roman(emissions_level: str) -> int:
        """
        Метод для получения чисел из римских чисел в уровнях экологичности
        param: emissions_level: str - уровень экологичности римским числом
        return: int - числовой уровень экологичности
        """
        levels = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6}
        return levels.get(emissions_level, 0)

    # удаляем единицы измерения
    columns = ['mileage', 'engine', 'max_power']
    for column in columns:
        df[column] = df[column].str.extract(r'(\d+\.?\d+ )').astype(float)

    # обрабатываем признак torque
    df['max_torque_rpm'] = df['torque'].str.extract(r'\D(\d+,?\d+)\D*$')
    df['max_torque_rpm'] = df['max_torque_rpm'].str.replace(',', '')
    df['max_torque_rpm'] = df['max_torque_rpm'].astype(float)
    df['torque_2'] = df['torque'].str.extract(r'^(\d+\.?\d*)').astype(float)
    torque_coef = df.apply(get_torque_coef, axis=1)
    df['torque_2'] = df['torque_2'] * torque_coef
    df.drop('torque', axis=1, inplace=True)
    df.rename(columns={'torque_2': 'torque'}, inplace=True)

    # заполняем пропуски
    na_counts = df.isnull().sum()
    na_columns = na_counts[na_counts > 0].index
    na_values = model_data.get('na_values', {})
    for column in na_columns:
        df[column] = df[column].fillna(na_values.get(column))

    # преобразуем некоторые колонки к int
    columns = ['engine', 'seats']
    for column in columns:
        df[column] = df[column].astype(int)

    # обработаем признак name
    df['brand'] = df['name'].apply(lambda x: x.split()[0])
    df['model'] = df['name'].apply(lambda x: x.split()[1])
    df['emissions_level'] = df['name'].str.extract(' BS ?-?([IV12345]+)')
    df['emissions_level'] = df['emissions_level'].apply(get_number_from_roman)
    equipment_pattern = '(EX|DX|LX|LTD|GT|TDI|VDI|ZDI|SE)'
    df['equipment_level'] = df['name'].str.extract(equipment_pattern)
    df.drop('name', axis=1, inplace=True)

    # добавляем новые признаки
    df['age_reverse'] = 1/(2024 - df['year'])**0.5
    df['km_driven_reverse'] = 1/df['km_driven']**0.5
    df.drop('year', axis=1, inplace=True)

    # кодируем категориальные признаки
    columns_to_encode = list(df.columns[df.dtypes == 'object'].tolist())
    columns_to_encode.append('seats')
    columns_to_encode.append('emissions_level')
    columns_to_encode.append('equipment_level')
    columns_to_encode.remove('model')
    encoder = model_data.get('encoder')
    col_encoded = encoder.transform(df[columns_to_encode])
    encoder_features = encoder.get_feature_names_out(columns_to_encode)
    df_encoded = pd.DataFrame(col_encoded, columns=encoder_features)
    df = pd.concat([df, df_encoded], axis=1).drop(columns=columns_to_encode)

    bin_columns = ['model']
    bin_enc = model_data.get('bin_encoder')
    df_bin = bin_enc.transform(df[bin_columns])
    df = pd.concat([df, df_bin], axis=1)
    df.drop(bin_columns, axis=1, inplace=True)
    df.drop('selling_price', axis=1, inplace=True)

    return df


def get_predictions(
    df: pd.DataFrame,
    filename_config: str
) -> List[float]:
    """
    Метод для получения предсказаний стоимости автомобилей с помощью
    предобученной модели
    param: df: pd.Dataframe - датафрейм с данными автомобилей, для
    которых нужно получить предсказания
    filename_config: str - имя файла с данными предобученной модели
    return: List[float] - список стоимостей автомобилей
    """
    with open(filename_config, 'rb') as f:
        best_model_data = pickle.load(f)

    prepared_df = prepare_data(df, best_model_data)

    best_model = Ridge(**best_model_data.get('params'))
    best_model.coef_ = np.array(best_model_data.get('weights'))
    best_model.intercept_ = best_model_data.get('intercept')

    if prepared_df.shape[0] == 1:
        prepared_df = prepared_df.iloc[0].values.reshape(1, -1)

    return np.round(best_model.predict(prepared_df), 2)


@app.get("/", summary='Root')
def root():
    """Текстовое описание сервиса"""
    description = """
        Welcome to Service for car price prediction!

        Use one of endpoints:
        - /predict_item (POST) - return prediction for one car
        - /predict_items (POST) - return predictions for list of cars
    """
    return PlainTextResponse(content=description)


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    """Получение предсказания для одного автомобиля"""
    df_cars = pd.DataFrame.from_records([item.to_dict()])
    return get_predictions(df_cars, MODEL_PARAMS_PATH)[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    """Получение списка предсказаний для списка автомобилей"""
    df_cars = pd.DataFrame.from_records(item.to_dict() for item in items)
    return list(get_predictions(df_cars, MODEL_PARAMS_PATH))


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
