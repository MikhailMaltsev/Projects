#!/usr/bin/env python
# coding: utf-8

# # Загрузка Pandas и просмотр данных

# In[1]:


import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
from itertools import combinations
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import ttest_ind


# In[2]:


df = pd.read_csv('main_task.csv')
df_original = pd.read_csv('main_task.csv')
df.head(5)


# In[3]:


# Посмотрим какие данные представлены в датасете
df.info()


# У нас есть 3 столбца с числовыми значениями, остальные со строковыми. Столбец Price Range содержит интервальные данные, позже мы их преобразуем. Из столбца Reviews также сможем выделить числовые значения, такие как "длина комментариев" и "дата". Из столбца Cuisine Slyle получим числовое значение кол-ва представленных кухонь.

# In[4]:


# Посмотрим кол-во пропусков по столбцам
df.isna().sum()


# # Работа с попусками

# In[5]:


# Дальше мы будем менять категоральные признаки на числовые, поэтому Price Range я заполняю 0
df['Price Range'].fillna(0, inplace=True) 

# По условию задачи, не отмеченная кухня в ресторане считается единичной, дадим ей собственную категорию
df['Cuisine Style'] = df['Cuisine Style'].fillna("['Unknown']") 

# Здесь я подразумеваю, что отзывов нет, поэтому также заполняю 0
df['Number of Reviews'].fillna(0, inplace=True) 

# В столбце Reviews есть пустые значения, их обработка будет производится дальше


# # Обработка признаков

# In[6]:


# Значения кухонь приведем к списку
df['Cuisine Style'] = [x[1:-1].split(', ') for x in df['Cuisine Style']]

# Создадим новый столбец с кол-вом кухонь у ресторана и заполним его
df['Cuisine Quantity'] = [len(x) for x in df['Cuisine Style']]


# In[7]:


# Получим спиcок всех дат и запишем разницу в днях между ними в новый столбец.
# Среднюю длину отзывов запишем также в новый столбец

# Разделяем значения Reviews на отзывы и даты
reviews = [0 if len(x) <= 8 else (x[2:-2].split('], [')[0]).split(', ') for x in df['Reviews']]
dates = [0 if len(x) <= 8 else (x[2:-2].split('], [')[1]).split(', ') for x in df['Reviews']]

# Приводим даты к datetime
dates_to_dates = ['0' if x == 0 else pd.to_datetime(x, format = "'%m/%d/%Y'") for x in dates]

# Получаем разницу дней между отзывами, записываем в новый столбец
df['Reviews Interval'] = [0 if len(x) == 1 else (x[0] - x[1]).days for x in dates_to_dates]

# Получаем среднюю длину отзывов, записываем в новый столбец
df['Reviews Lenght'] = [x if x == 0 else sum(map(len,x))/len(x) for x in reviews]


# In[8]:


# Заменим значения Price Range на числовые
price_dict = {'$': 1, '$$ - $$$': 2, '$$$$': 3}
df['Price Range'] = df['Price Range'].replace(to_replace=price_dict)


# In[9]:


# Создадим dummy variables для видов кухонь
dummies = pd.get_dummies(df['Cuisine Style'].explode())
dummies = dummies.groupby(dummies.index).sum().reset_index()
dummies.drop(['index'], axis='columns', inplace=True)
df = df.join(dummies)


# In[10]:


# Тоже самое сделаем для городов
df = pd.get_dummies(df, columns=[ 'City',], dummy_na=True)


# In[11]:


# Посмотрим, что получается
pd.set_option('display.max_columns', None)
df.head(5)


# # Обогащение

# In[12]:


# В приступе отсутствия лени, добавим в датасет информацию о населении городов.
# Для этого используем скачанный на Kaggle датасет World Cities 2019 года выпуска.
cities = pd.read_csv('worldcities.csv')
cities


# In[13]:


# Проверим, все ли наши города присутствуют в нем.
city_to_check = pd.DataFrame(df_original['City'].unique())
result = city_to_check.isin(cities.city_ascii.unique())
city_to_check[result == False].dropna() # Посмотрим кто отуствует


# In[14]:


# Постотрим сколько раз названия наших городов втсречаются в датасете WC
for i in range(len(city_to_check[0])):
    if city_to_check[0][i] in cities.city_ascii.unique():
        display(cities[cities.city_ascii == city_to_check[0][i]])


# In[15]:


# Видно что дубликаты городов: 28 из US, 1 из VE, 1 из CA.
# Можно смело минусануть все города из данных стран, т.к. наших городов в этих странах нет.
to_del = cities[cities.iso2.isin(['US','VE','CA'])]
cities = cities[~cities.index.isin(to_del.index)]


# In[16]:


# Получим значения population для наших городов
pop = []
our_cities = city_to_check[0].tolist()

for i in range(len(city_to_check[0])):
    if city_to_check[0][i] in cities.city_ascii.unique():
        pop.append(int(cities[cities.city_ascii == city_to_check[0][i]].population.values[0]))
    else:
        pop.append(0)

population_dict = dict(zip(our_cities, pop))

# Заполним пропуски данными из Википедии
missed_cities = {'Oporto': 214349,'Copenhagen': 602481}
population_dict.update(missed_cities)


# In[17]:


# Добавим в рабочий датафрейм данные о населении
df['City Population'] = [population_dict.get(x) for x in df_original['City']]


# In[18]:


# Финальная проверка на пропуски
df.isna().sum().sum()


# # Анализ данных

# In[19]:


# Проверим распределение признака Ranking на нормальность
# Проведем тест Шапиро-Уилка
import scipy
stat, p = scipy.stats.shapiro(df_original['Ranking'])
print('Statistics=%.3f, p-value=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Нормальное распределение признака')
else:
    print('Ненормальное распределение признака')


# In[20]:


# Посмотрим на его распределение в ТОП1 городе 
df_original['Ranking'][df_original['City'] =='London'].hist(bins=100)


# In[21]:


# Посмотрим на его распределение в ТОП10 городах
for top10cities in df_original['City'].value_counts()[0:10].index:
    df_original['Ranking'][df_original['City'] == top10cities].hist(bins=100)
plt.show()


# Получается, что Ranking имеет нормальное распределение, просто в больших городах больше ресторанов, из-за мы этого имеем смещение. (Списал с ноутубка Kaggle)

# In[22]:


# Сделаем тест Стьюдента
def get_stat_dif(column):
    cols = df.loc[:, column].value_counts().index[:50]
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(df.loc[df.loc[:, column] == comb[0], 'Rating'],
                     df.loc[df.loc[:, column] == comb[1], 'Rating']).pvalue \
                <= 0.05/len(combinations_all):  # Учли поправку Бонферони
            print('Найдены статистически значимые различия для колонки', column)
            break


# In[23]:


for col in ['Ranking', 'Price Range', 'Number of Reviews', 'Cuisine Quantity', 'Reviews Interval', 'Reviews Lenght']:
    get_stat_dif(col)


# Забираем эти столбцы в модель.

# In[24]:


# Соберем модель для корреляции, исключая кухни (их очень много)
corr_model = df[df.columns[:12]].join(df[df.columns[-32:]])


# In[25]:


plt.rcParams['figure.figsize'] = (12,8)
sns.heatmap(corr_model.corr(),cmap='coolwarm')


# Из тепловой карты корреляции становится понятно, что:<br>
# 
# 1) Самые высокий Ranking (= больше низкорейтинговых заведений) в Лондоне и Париже (потомучто больше всего данных именно оттуда)<br>
# 2) Чем больше кухонь представлено в ресторане, тем выше у него диапозон цены<br>
# 3) У ресторанов с большим кол-вом кухонь большее кол-во отзывов и меньше Ranking (что лучше)<br>
# 4) Чем город больше, тем больше в нем ресторанов (что не удивительно) и выше Ranking (что хуже)<br><br>
# Выводы:<br>
# В крупных городах численно больше заведений с маленьким набором кухонь, низким Ranking'ом. Подозреваю что это связано с туристическим потоком в этих городах. Но проверять гипотезу уже нет сил. Простите.
# 
# 

# # Подготовка модели

# In[26]:


# Удалим столбцы типа Object
model = df
model.drop(['Restaurant_id','Cuisine Style','Reviews','URL_TA','ID_TA'], axis='columns', inplace=True)


# In[27]:


# Нормализуем все признаки и целевую переменную

scaler = MinMaxScaler()

def normalizer(df):
    for i in range(0,len(df.columns)):
        if type(df[df.columns[i]][0]) != type('str'):
            to_norm = np.array(df[df.columns[i]]).reshape(-1, 1)
            df[df.columns[i]] = scaler.fit_transform(to_norm)  
        else: 
            continue

normalizer(model)
model


# # Разбиваем датафрейм на части, необходимые для обучения и тестирования модели

# In[28]:


# Х - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)
X = model.drop(['Rating'], axis = 1)
y = model['Rating']


# In[29]:


# Загружаем специальный инструмент для разбивки:
from sklearn.model_selection import train_test_split


# In[30]:


# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.

RANDOM_SEED = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


# # Создаём, обучаем и тестируем модель

# In[31]:


# Импортируем необходимые библиотеки:
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели


# In[32]:


# Создаём модель
regr = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)

# Обучаем модель на тестовом наборе данных
regr.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = regr.predict(X_test)


# In[33]:


# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.
MAE = metrics.mean_absolute_error(y_test, y_pred)
print('MAE:', MAE)


# In[34]:


# Оценка влияния признаков на качество модели
plt.rcParams['figure.figsize'] = (10,6)
feat_importances = pd.Series(regr.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')

