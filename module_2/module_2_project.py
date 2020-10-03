#!/usr/bin/env python
# coding: utf-8

# # Введение

# Нас (возможно и вас) пригласили поучаствовать в одном из проектов UNICEF — международного подразделения ООН, чья миссия состоит в повышении уровня благополучия детей по всему миру. 
# 
# Суть проекта — отследить влияние условий жизни учащихся в возрасте от 15 до 22 лет на их успеваемость по математике, чтобы на ранней стадии выявлять студентов, находящихся в группе риска.

# In[1046]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind
import scipy.stats
import itertools as it
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import ttest_ind


# # Ознакомление
# Для начала загрузим файл в датафрейм (ДФ), переименуем некоторые колонки для удобства и посмотрим развернутый ДФ.

# In[1047]:


df = pd.read_csv('stud_math.csv')
df.rename(columns={'studytime, granular': 'studytime_granular',
                         'Pstatus': 'pstatus', 'Medu': 'medu', 'Fedu': 'fedu',
                         'Mjob': 'mjob', 'Fjob': 'fjob'}, inplace=True)
pd.set_option('max_columns', None)


# In[1048]:


df.sample(10)


# #### Значения колонок датафрейма
# 
# school — аббревиатура школы, в которой учится ученик
# 
# sex — пол ученика ('F' - женский, 'M' - мужской)
# 
# age — возраст ученика (от 15 до 22)
# 
# address — тип адреса ученика ('U' - городской, 'R' - за городом)
# 
# famsize — размер семьи('LE3' <= 3, 'GT3' >3)
# 
# Pstatus — статус совместного жилья родителей ('T' - живут вместе 'A' - раздельно)
# 
# Medu — образование матери (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)
# 
# Fedu — образование отца (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)
# 
# Mjob — работа матери ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба, 'at_home' - не работает, 'other' - другое)
# 
# Fjob — работа отца ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба, 'at_home' - не работает, 'other' - другое)
# 
# reason — причина выбора школы ('home' - близость к дому, 'reputation' - репутация школы, 'course' - образовательная программа, 'other' - другое)
# 
# guardian — опекун ('mother' - мать, 'father' - отец, 'other' - другое)
# 
# traveltime — время в пути до школы (1 - <15 мин., 2 - 15-30 мин., 3 - 30-60 мин., 4 - >60 мин.)
# 
# studytime — время на учёбу помимо школы в неделю (1 - <2 часов, 2 - 2-5 часов, 3 - 5-10 часов, 4 - >10 часов)
# 
# failures — количество внеучебных неудач (n, если 1<=n<=3, иначе 0)
# 
# schoolsup — дополнительная образовательная поддержка (yes или no)
# 
# famsup — семейная образовательная поддержка (yes или no)
# 
# paid — дополнительные платные занятия по математике (yes или no)
# 
# activities — дополнительные внеучебные занятия (yes или no)
# 
# nursery — посещал детский сад (yes или no)
# 
# higher — хочет получить высшее образование (yes или no)
# 
# internet — наличие интернета дома (yes или no)
# 
# romantic — в романтических отношениях (yes или no)
# 
# famrel — семейные отношения (от 1 - очень плохо до 5 - очень хорошо)
# 
# freetime — свободное время после школы (от 1 - очень мало до 5 - очень мого)
# 
# goout — проведение времени с друзьями (от 1 - очень мало до 5 - очень много)
# 
# health — текущее состояние здоровья (от 1 - очень плохо до 5 - очень хорошо)
# 
# absences — количество пропущенных занятий
# 
# score — баллы по госэкзамену по математике

# In[1049]:


df.info()


# In[1050]:


# Посмотрим кол-во пропусков в колонках
df.isna().sum()


# В датасете всего 30 колонок: 13 числовых колонок и 17 строковых. 
# Датасете содержит данные данные о 395 учениках.
# Во всех колонках, кроме school, sex, age есть пустые значения.

# ### Предобработка

# In[1051]:


# Функция для получения быстрой справки о данных в числовых колонках
def info_dig(x):
    print(pd.DataFrame(x.value_counts()))
    print('Пропущенных значений -', x.isnull().values.sum())
    x.hist(figsize = [6,4])


# In[1052]:


# Функция для получения быстрой справки о данных в текстовых колонках
def info_object(smth):
    print(pd.DataFrame(smth.value_counts()))
    print('Пропущенных значений -', smth.isnull().values.sum())
    plt.rcParams['figure.figsize'] = (6,4)
    sns.boxplot(x=smth, y='score', data=df)


# ### Просмотр числовыx столбцов

# In[1053]:


info_dig(df.age)


# In[1054]:


info_dig(df.medu)


# In[1055]:


# Заполним пропуски средним значением
df.medu.fillna(round(df.medu.mean()), inplace=True)


# In[1056]:


info_dig(df.fedu)


# In[1057]:


# Исправим опечатку значения 40.0, вероятно имелось ввиду 4.
df.fedu = df.fedu.apply(lambda x: x/10 if x > 9 else x)


# In[1058]:


# Заполним пропуски средним значением
df.fedu.fillna(round(df.fedu.mean()), inplace=True)


# In[1059]:


info_dig(df.traveltime)


# In[1060]:


# Заполним пропуски средним значением
df.traveltime.fillna(round(df.traveltime.mean()), inplace=True)


# In[1061]:


info_dig(df.studytime)


# In[1062]:


# Заполним пропуски средним значением
df.studytime.fillna(round(df.studytime.mean()), inplace=True)


# In[1063]:


info_dig(df.failures)


# In[1064]:


# Заполним пропуски средним значением
df.failures.fillna(round(df.failures.mean()), inplace=True)


# In[1065]:


info_dig(df.studytime_granular)


# In[1066]:


# Заполним пропуски самым частоврстречаемым значением
df.studytime_granular.fillna(-6.0, inplace=True)


# In[1067]:


info_dig(df.famrel)


# In[1068]:


# Исправим опечатку значения -1.0, вероятно имелось ввиду 1.0
df.famrel = df.famrel.apply(lambda x: abs(x) if x < 0 else x)


# In[1069]:


# Заполним пропуски средним значением
df.famrel.fillna(round(df.famrel.mean()), inplace=True)


# In[1070]:


info_dig(df.freetime)


# In[1071]:


# Заполним пропуски средним значением
df.freetime.fillna(round(df.freetime.mean()), inplace=True)


# In[1072]:


info_dig(df.goout)


# In[1073]:


# Заполним пропуски средним значением
df.goout.fillna(round(df.goout.mean()), inplace=True)


# In[1074]:


info_dig(df.health)


# In[1075]:


# Заполним пропуски средним значением
df.health.fillna(round(df.health.mean()), inplace=True)


# In[1076]:


info_dig(df.absences)


# In[1077]:


# Вероятно, значения 212 и 385 являются ошибками, т.к. выходят за рамки кол-ва учебных дней и кол-ва дней в году соответственно.
# Удалим их из датасета.
df = df[~df.absences.isin([212.0,385.0])]


# In[1078]:


# Заполним пропуски медианным значением, т.к. достаточно большой разброс
df.absences.fillna(round(df.absences.median()), inplace=True)


# In[1079]:


# Посмотрим еще раз не получившийся результат
info_dig(df.absences)


# In[1080]:


# Определим выбросы и удалим их. Выбросами считаем значения больше 30, 
# т.к. пропуск более 30 учебных дней скорее всего приведет к переводу ученика на домашнее обучение

median = df.absences.median()
IQR = df.absences.quantile(0.75) - df.absences.quantile(0.25)
perc25 = df.absences.quantile(0.25)
perc75 = df.absences.quantile(0.75)
print('25-й перцентиль: {},'.format(perc25), 
      '75-й перцентиль: {},'.format(perc75), 
      "IQR: {}, ".format(IQR),
      "Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))
df.absences.loc[df.absences.between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)].hist(bins = 16,
                                                                              range = (0, 30),
                                                                              label = 'IQR',
                                                                              figsize = [6,4])
plt.legend();


# In[1081]:


# Удалим выбросы

df = df[df.absences.between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)]


# In[1082]:


info_dig(df.score)


# In[1083]:


# Пропуски в целевой переменной можно удалить
df = df[~df.score.isna()]

# Также можно удалить нулевые значения, вероятно ученик просто не явился на экзамен
df = df[df.score > 0.0]


# ### Просмотр строковых столбцов

# In[1084]:


info_object(df.school)


# In[1085]:


info_object(df.sex)


# In[1086]:


info_object(df.address)


# In[1087]:


# Чтобы заполнить пропуски найдем среднее время до школы для городских и загородных учеников
df.groupby(['address'])['traveltime'].mean().reset_index()


# In[1088]:


# Т.к. у нас круглые значения в traveltime - будем считать, что 2 и более - это загородные ученики. 

pd.set_option('mode.chained_assignment', None) # Чтобы не ругался

df = df.reset_index() # Сбросим индексы учеников, иначе алерт
df.address.fillna(0, inplace=True) # Заполним пропуски 0 для удобства

# Заполняем пропуски
for i in range(0, len(df)):
    if df.address[i] == 0:
        if df.traveltime[i] > 1.0:
            df.address[i] = 'R'
        else:
            df.address[i] = 'U'


# In[1089]:


info_object(df.famsize)


# In[1090]:


# Заполним пропуски самым частовстречаемым значением
# Логическое предположение: родители живут вместе - больше семья
# Для этого обратимся к параметру Pstatus (Совместное проживание родителей)
df.groupby(['pstatus'])['famsize'].value_counts() 


# In[1091]:


# Предположение не оправдалось, во всех группах преобладает значение GT3, им и заполним пропуски
df.famsize.fillna('GT3', inplace=True)


# In[1092]:


info_object(df.pstatus)


# In[1093]:


# Заполним пропуски самым частовстречаемым значением
df.pstatus.fillna('T', inplace=True)


# In[1094]:


info_object(df.mjob)


# In[1095]:


# Заполним пропуски неопределенным значением
df.mjob.fillna('other', inplace=True)


# In[1096]:


info_object(df.fjob)


# In[1097]:


# Заполним пропуски неопределенным значением
df.fjob.fillna('other', inplace=True)


# In[1098]:


info_object(df.reason)


# In[1099]:


# Заполним пропуски самым частовстречаемым значением
df.reason.fillna('course', inplace=True)


# In[1100]:


info_object(df.guardian)


# In[1101]:


# Заполним пропуски самым частовстречаемым значением
df.guardian.fillna('mother', inplace=True)


# In[1102]:


info_object(df.schoolsup)


# In[1103]:


# Заполним пропуски самым частовстречаемым значением
df.schoolsup.fillna('no', inplace=True)


# In[1104]:


info_object(df.famsup)


# In[1105]:


# Заполним пропуски самым частовстречаемым значением
df.famsup.fillna('yes', inplace=True)


# In[1106]:


info_object(df.paid)


# In[1107]:


# Заполним пропуски самым частовстречаемым значением
df.paid.fillna('no', inplace=True)


# In[1108]:


info_object(df.activities)


# In[1109]:


# Значения близкие, хочется распределить пропуски равномерно
# Заполним пропуски поочереди каждым значением

df.activities.fillna(0, inplace=True) # Сначала заполним пропуски 0 для убоства

# Поочереди заполняем пропуски значениями
for i in range(0, len(df), 2): 
    if df.activities[i] == 0:
        df.activities[i] = 'yes'
        
for i in range(0, len(df)):
    if df.activities[i] == 0:
        df.activities[i] = 'no'


# In[1110]:


info_object(df.nursery)


# In[1111]:


# Заполним пропуски самым частовстречаемым значением
df.nursery.fillna('yes', inplace=True)


# In[1112]:


info_object(df.higher)


# In[1113]:


# Заполним пропуски самым частовстречаемым значением
df.higher.fillna('yes', inplace=True)


# In[1114]:


info_object(df.internet)


# In[1115]:


# Заполним пропуски самым частовстречаемым значением
df.internet.fillna('yes', inplace=True)


# In[1116]:


info_object(df.romantic)


# In[1117]:


# Заполним пропуски самым частовстречаемым значением
df.romantic.fillna('no', inplace=True)


# In[1118]:


# Финальная проверка на пропуски
df.isna().sum().sum()


# # Поиск зависимостей
# Посмотрим корреляцию числовых значений

# In[1119]:


plt.rcParams['figure.figsize'] = (12,8)
sns.heatmap(df.corr(),cmap='coolwarm')


# Полная корреляция столбцов studytime и studytime_granular позволяют не брать последний в рассчет.
# 
# Образование родителей (medu и fedu) и кол-во внеучебных неудач (failures) больше других оказывают влияние на успеваемость. Т.е. чем выше образование родителей, тем более успешный ребенок в жизни в целом, в том числе и в учебе. 
# 
# Также можно увидеть корреляцию возраста (age) и кол-ва внеучебных неудач (failures). Чем страше ребенок, тем хуже успеваемость. Можно предположить, что более молодые родители имеют лучшее образование, либо растущие дети начинают больше времени уделять друзьям (gout) а не учебе (studytime).
# 
# Присутствует также логическая связь влияния пропусков занятий (absences) на успеваемость, эти данные мы тоже возьмем в модель.

# Еще раз построим графики и посмотрим на распределения категоральных данных

# In[1120]:


def get_boxplot(column):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=column, y='score',
                data=df.loc[df.loc[:, column].isin(df.loc[:, column].value_counts().index[:10])], ax=ax)
    plt.xticks(rotation=0)
    ax.set_title('Boxplot for ' + column)
    plt.show()


# In[1121]:


for col in ['school', 'address', 'famsize', 'pstatus', 'mjob', 'fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 
            'activities', 'nursery', 'higher', 'sex', 'internet', 'romantic']:
    get_boxplot(col)


# По условию задачи поиска группы риска, оцениваем факторы влияющие на ухудшение успеваемости.
# Из графиков мы видим, что минимальных значений больше у учащихся:
# <ul>
#     <li>школы GP</li>
#     <li>живущих за городом (address)</li>
#     <li>с размеров семьи больше 3 человек (famsize)</li>
#     <li>родители которых проживают совместно (pstatus)
#     <li>опекуном которых является мать (guardian) - статистическое большинство
#     <li>без дополнительной образовательной поддержки
#     <li>имеющих учебную поддержку дома (famsup)
#     <li>не занимающихся дополнительно (платно) (paid)
#     <li>посещавших детский сад
#     <li>не собирающихся получать высшее образование (higher) - сильно влияет
#     <li>женского пола
#     <li>имеющих доступ к интернету
#     <li>состоящих в романтических отношениях
# </ul>

# Сделаем тест Стюдента

# In[1122]:


def get_stat_dif(column):
    cols = df.loc[:, column].value_counts().index[:20]
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(df.loc[df.loc[:, column] == comb[0], 'score'],
                     df.loc[df.loc[:, column] == comb[1], 'score']).pvalue \
                <= 0.05/len(combinations_all):  # Учли поправку Бонферони
            print('Найдены статистически значимые различия для колонки', column)
            break


# In[1123]:


for col in ['school', 'address', 'famsize', 'pstatus', 'mjob', 'fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid',
            'activities', 'nursery', 'higher', 'sex', 'internet', 'romantic']:
    get_stat_dif(col)


# ### Собираем датасет для модели
# Немного выводов:
# <ol>
#     <li>В целом, сильно влияет образование родителей. Они коррелируют достаточно сильно, но я решил не объединять, чтобы выявить будущие зависимости от пола родителей. Образование отца сильнее влияет на успеваемость, но также выделяется позиция матери преподавателя.</li>
#     <li>Также сильно влияют на успеваемость планы учащегося поступать в высшее учебное заведение. Если таких планов нет - то и мотивация соответствующая.</li>
#     <li>Остальные параметры в меньшей степени влияют на результат, но их сумма может оказать решающее воздействие.</li>
# </ol>

# In[1124]:


model = df[['school','age','sex','address', 'medu', 'fedu', 'mjob', 'fjob', 'studytime', 'failures','romantic','schoolsup','higher', 'goout', 'absences', 'score']]
model


# # Нас не просили, но все же...

# In[1125]:


# Преобразуем категоральные данные в dummy
model = pd.get_dummies(model, columns=['school','sex','address','mjob','fjob','romantic','schoolsup','higher'])
model


# In[1126]:


# Нормализуем все признаки, кроме целевой переменной
scaler = MinMaxScaler()

def normalizer(df):
    for i in range(0,len(df.columns)):
        if df.columns[i] != 'score':
            to_norm = np.array(df[df.columns[i]]).reshape(-1, 1)
            df[df.columns[i]] = scaler.fit_transform(to_norm)  
        else: 
            continue

normalizer(model)
model


# # Разбиваем датафрейм на части, необходимые для обучения и тестирования модели

# In[1127]:


# Х - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)
X = model.drop(['score'], axis = 1)
y = model['score']


# In[1128]:


# Загружаем специальный инструмент для разбивки:
from sklearn.model_selection import train_test_split


# In[1129]:


# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.
# Для тестирования мы будем использовать 100% от исходного датасета.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)


# # Создаём, обучаем и тестируем модель

# In[1130]:


# Импортируем необходимые библиотеки:
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели


# In[1131]:


# Создаём модель
regr = RandomForestRegressor(n_estimators=100)

# Обучаем модель на тестовом наборе данных
regr.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = regr.predict(X_test)


# In[1132]:


# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.
MAE = metrics.mean_absolute_error(y_test, y_pred)
print('MAE:', MAE)


# In[1133]:


# Оценка влияния признаков на качество модели
plt.rcParams['figure.figsize'] = (10,4)
feat_importances = pd.Series(regr.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')

