import streamlit as st
import joblib
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import random
import pandas as pd
from io import StringIO
import sys
# Важная настройка для корректной настройки pipeline!
import sklearn
sklearn.set_config(transform_output="pandas")

import seaborn as sns
import matplotlib.pyplot as plt




# Загрузка пайплайна
pipeline = joblib.load('pipeline_model.joblib')

# Создание боковой панели для навигации
page = st.sidebar.radio("Выберите страницу:", ("Предсказание", "Отчетность"))

# Контент главной страницы
if page == "Предсказание":
    st.header("Предсказание цены дома")

# Контент второй страницы
elif page == "Отчетность":
    st.header("Процесс работы")

#аббеатуры для категориальных признаков

#Condition2
abbreviation_feature10 = {
    "Adjacent to arterial street": "Artery",
    "Adjacent to feeder street": "Feedr",
    "Normal": "Norm",
    "Within 200' of North-South Railroad": "RRNn",
    "Adjacent to North-South Railroad": "RRAn",
    "Near positive off-site feature--park, greenbelt, etc.":"PosN",
    "Adjacent to postive off-site feature": "PosA",  
    "Within 200' of East-West Railroad": "RRNe",
    "Adjacent to East-West Railroad": "RRAe",  

}

#MSZoning
abbreviation_feature11 = {
    "Agriculture": "A",
    "Commercial": "C",
    "Floating Village Residential": "FV",
    "Industrial": "I",
    "Residential High Density": "RH",
    "Residential Low Density": "RL",  
    "Residential Low Density Park": "RP",
    "Residential Medium Density": "RM",    
}

#Neighborhood
abbreviation_feature12 = {
    "Bloomington Heights": "Blmngtn",
    "Bluestem": "Blueste",
    "Briardale": "BrDale",
    'Brookside':'BrkSide',
    "Clear Creek": "ClearCr",
    "College Creek": "CollgCr",
    "Crawford": "Crawfor",  
    "Edwards": "Edwards",
    "Gilbert": "Gilbert",
    "Iowa DOT and Rail Road": "IDOTRR",   
    "Meadow Village": "MeadowV",   
    "Mitchell": "Mitchel",   
    "North Ames": "Names",   
    "Northridge": "NoRidge",   
    "Northpark Villa": "NPkVill",   
    "Northridge Heights": "NridgHt",   
    "Northwest Ames": "NWAmes",   
    "Old Town": "OldTown",   
    "South & West of Iowa State University": "SWISU",   
    "Sawyer": "Sawyer",   
    "Sawyer West": "SawyerW",  
    "Somerset": "Somerst",  
    "Stone Brook": "StoneBr",  
    "Timberland": "Timber",  
    "Veenker": "Veenker",
}

#SaleCondition
abbreviation_feature13 = {
    "Normal Sale": "Normal",
    "Abnormal Sale - trade, foreclosure, short sale": "Abnorml",
    "Adjoining Land Purchase": "AdjLand",
    "Allocation": "Alloca",
    "Sale between family members": "Family",
    "Home was not completed when last assessed": "Partial",  
}
#BldgType
abbreviation_feature14 = {
    "Single-family Detached": "1Fam",
    "Two-family Conversion; originally built as one-family dwelling": "2FmCon",
    "Duplex": "Duplx",
    "Townhouse End Unit": "TwnhsE",
    "Townhouse Inside Unit": "TwnhsI"
}
#BsmtCond
abbreviation_feature15 = {
    "Excellent": "Ex",
    "Good": "Gd",
    "Typical - slight dampness allowed": "TA",
    "Fair - dampness or some cracking or settling": "Fa",
    "Poor - Severe cracking, settling, or wetness": "Po",
    "No Basement": "NA"
}
#Alley
abbreviation_feature16 = {
    "Gravel": "Grvl",
    "Paved": "Pave",
    "No alley access": "NA"
}

#BsmtExposure
abbreviation_feature17 = {
    "Good Exposure": "Gd",
    "Average Exposure (split levels or foyers typically score average or above)": "Av",
    "Mimimum Exposure": "Mn",
    "No Exposure": "No",
    "No Basement": "NA"
}


if page == "Предсказание":
# Ввод данных от пользователя
    st.sidebar.title("Введите/Выберете данные для вашего дома мечты")
# Ввод данных от пользователя
# Числовые характеристики
    feature1 = st.sidebar.slider("Предпочтительная отделка дома (от 0 (плохо) до 10 (самая лучшая в городе)):", 0, 10, 5)
    feature2 = int(st.sidebar.number_input("Какая площадь подвала вам необходима:"))
    feature3 = int(st.sidebar.number_input("Количество каминов"))
    #feature4 = st.sidebar.number_input("Количество спален", 3)
    feature5 = int(st.sidebar.number_input("Площадь крыльца"))
    feature6 = int(st.sidebar.number_input("Готовые квадратные футы"))
    feature7 = int(st.sidebar.number_input("Готовые квадратные футы 1 этажа"))
    feature8 = int(st.sidebar.number_input("Готовые квадратные футы 2 этажа"))
    feature9 = int(st.sidebar.number_input("Размер участка"))


# Категорианльные характеристики
    feature10 = st.sidebar.selectbox("Выберете близость к различным условиям", 
list(abbreviation_feature10.keys()))
    feature11 = st.sidebar.selectbox("Выберите зону", 
list(abbreviation_feature11.keys()))
    feature12 = st.sidebar.selectbox("Выберите местораположение", 
list(abbreviation_feature12.keys()))
    feature13 = st.sidebar.selectbox("Выберите тип продажи", 
list(abbreviation_feature13.keys()))
    feature14 = st.sidebar.selectbox("Выберите тип жилья", 
list(abbreviation_feature14.keys()))
    feature15 = st.sidebar.selectbox("Выберите состояние подвала", 
list(abbreviation_feature15.keys()))
    feature16 = st.sidebar.selectbox("Выберите выход к аллее", 
list(abbreviation_feature16.keys()))
    feature17 = st.sidebar.selectbox("Выберите желаемую экспозицию", 
list(abbreviation_feature17.keys()))

# Кнопка для отправки данных
    if st.button("Предсказать"):
    # Подготовка данных для модели
        input_data = {
        'OverallCond': [feature1],
        'BsmtFinSF2': [feature2],
        'Fireplaces': [feature3],
        #'Bedroom': [feature4],
        'Condition2': [abbreviation_feature10[feature10]],
        'BldgType': [abbreviation_feature14[feature14]],
        'BsmtCond': [abbreviation_feature15[feature15]],
        'MSZoning': [abbreviation_feature11[feature11]],
        '3SsnPorch': [feature5],
        'BsmtFinSF1': [feature6],
        '1stFlrSF': [feature7],
        '2ndFlrSF': [feature8],
        'Neighborhood': [abbreviation_feature12[feature12]],
        'Alley': [abbreviation_feature16[feature16]],
        'SaleCondition': [abbreviation_feature13[feature13]],
        'LotArea': [feature9],           
        'BsmtExposure': [abbreviation_feature17[feature17]]
        }
        
        df = pd.DataFrame(input_data)
        st.write("Введенные данные:", df)
        #Предсказание
        try:
            prediction = pipeline.predict(df)
            st.write("Предсказание:", prediction)  
        except Exception as e:
            st.error(f"Ошибка предсказания: {e}")     

        
#Разбавим серьезную атмосферу
        images = {
    "image1": "https://www.film.ru/sites/default/files/images/HZ-01.jpg",
    "image2": "https://www.architime.ru/news/airbnb/1.jpg",
    "image3": "https://xn--c1acndtdamdoc1ib.xn--p1ai/upload/iblock/59f/IMG_6382.jpg",
    "image4": "https://static.wikia.nocookie.net/elderscrolls/images/1/1a/%D0%92%D0%B0%D0%B9%D1%82%D1%80%D0%B0%D0%BD_%28Skyrim%29.png/revision/latest/scale-to-width-down/1000?cb=20201019143159&path-prefix=ru",
    "image5": "https://avatars.dzeninfra.ru/get-zen_doc/9662638/pub_64956e41c8fc6f10d7bc073e_64956e6c86f0b87a4a4dd2c8/scale_2400"
        }

# Выбор случайного изображения для отображения
        selected_image_key = random.choice(list(images.keys()))
        selected_image_url = images[selected_image_key]
# Получаем изображение из URL
        response = requests.get(selected_image_url)
        image = Image.open(BytesIO(response.content))

#Вывод результатов
        st.image(image, caption="Ваш дом мечты!", use_column_width=True)


elif page == "Отчетность":
    st.write('Набор данных о жилье доступен на Kaggle в разделе «House Prices: Advanced Regression Techniques». Файл «train.csv» содержит обучающие данные, а «test.csv» — тестовые данные. Обучающие данные содержат данные для 1460 строк, что соответствует данным о 1460 домах, и 80 столбцов, которые соответствуют характеристикам этих домов. Аналогично, тестовые данные содержат данные о 1461 доме и их 79 атрибутах.')
    st.write('Тестовые данные')
    data_test= pd.read_csv('test.csv')
    st.dataframe(data_test)
    st.write('Обучающие данные')
    data_train = pd.read_csv('train.csv')
    st.dataframe(data_train)
    st.write('Посмотрим сводную таблицу по колонке SalePrice', data_train['SalePrice'].describe())
    
    st.write('Построим корреляционную матрицу')
    image_corr = Image.open('corr.png')
    st.image(image_corr, use_column_width=True)

    #График влияния фич на таргет
    st.write('График влияния фич на таргет')
    image = Image.open('shape.png')
    st.image(image, use_column_width=True)


    st.write('Графики распределения фич')
    plt.figure(figsize=(20, 20))
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars','GarageArea', 'TotalBsmtSF','1stFlrSF','FullBath','YearBuilt']
    sns_plot = sns.pairplot(data_train[cols])

    plt.suptitle('Scatter plots between top 9 most corr features', y=1.04, size=25)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

    st.write('Графики ящиков с усами')
    # plot a box plot for categorical feature : Year Built
    fig = plt.figure(figsize=(18,8))

    data = pd.concat([data_train['SalePrice'], data_train['YearBuilt']], axis=1)
    sns.boxplot(x= data_train['YearBuilt'], y="SalePrice", data=data)
    plt.xticks(rotation=90,fontsize= 9)
    st.pyplot(plt)


    st.write('Итоги')
    st.write('после удаления фич в резудьбтате фич аналтьза показатель качества RMSE составиЛ 0,15. так же были проанализированы след модела: Catboost, RandomForest и метод ближайших соседей. лучший результат у линейной модели Lasso, в обоих случаях R^2 0,85%. При фиче анализе былипроделаны след операции: рассмотрена корр матрица, построен shape и ящик с усами')

    #Итоговое место на Kaggle
    st.write('Итоговое место на Kaggle, который был получен для безлайн модели, далее был произведен анализ для упрощения модели, которые не показал значительных успехов, поэтому оставляем первыйц полученный результат')
    image_kg = Image.open('kaggle.png')
    st.image(image_kg, use_column_width=True)


