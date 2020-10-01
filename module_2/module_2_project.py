#!/usr/bin/env python
# coding: utf-8

# # Введение

# Нас (возможно и вас) пригласили поучаствовать в одном из проектов UNICEF — международного подразделения ООН, чья миссия состоит в повышении уровня благополучия детей по всему миру. 
# 
# Суть проекта — отследить влияние условий жизни учащихся в возрасте от 15 до 22 лет на их успеваемость по математике, чтобы на ранней стадии выявлять студентов, находящихся в группе риска.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind
import scipy.stats
import itertools as it


# # Ознакомление
# Для начала загрузим файл в датафрейм (ДФ), переименуем некоторые колонки для удобства и посмотрим развернутый ДФ.

# In[2]:


df = pd.read_csv('stud_math.csv')
df.rename(columns={'studytime, granular': 'studytime_granular',
                         'Pstatus': 'pstatus', 'Medu': 'medu', 'Fedu': 'fedu',
                         'Mjob': 'mjob', 'Fjob': 'fjob'}, inplace=True)
pd.set_option('max_columns', None)


# In[3]:


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

# In[4]:


df.info()


# In[5]:


# Посмотрим кол-во пропусков в колонках
df.isna().sum()


# В датасете всего 30 колонок: 13 числовых колонок и 17 строковых. 
# Датасете содержит данные данные о 395 учениках.
# Во всех колонках, кроме school, sex, age есть пустые значения.

# ### Предобработка

# In[6]:


# Функция для получения быстрой справки о данных в числовых колонках
def info_dig(x):
    print(pd.DataFrame(x.value_counts()))
    print('Пропущенных значений -', x.isnull().values.sum())
    x.hist()


# In[7]:


# Функция для получения быстрой справки о данных в текстовых колонках
def info_object(smth):
    print(pd.DataFrame(smth.value_counts()))
    print('Пропущенных значений -', smth.isnull().values.sum())
    sns.boxplot(x=smth, y='score', data=df)


# ### Просмотр числовыx столбцов

# In[8]:


info_dig(df.age)


# In[9]:


info_dig(df.medu)


# In[10]:


# Заполним пропуски средним значением
df.medu.fillna(round(df.medu.mean()), inplace=True)


# In[11]:


info_dig(df.fedu)


# In[12]:


# Исправим опечатку значения 40.0, вероятно имелось ввиду 4.
df.fedu = df.fedu.apply(lambda x: x/10 if x > 9 else x)


# In[13]:


# Заполним пропуски средним значением
df.fedu.fillna(round(df.fedu.mean()), inplace=True)


# In[14]:


info_dig(df.traveltime)


# In[15]:


# Заполним пропуски средним значением
df.traveltime.fillna(round(df.traveltime.mean()), inplace=True)


# In[16]:


info_dig(df.studytime)


# In[17]:


# Заполним пропуски средним значением
df.studytime.fillna(round(df.studytime.mean()), inplace=True)


# In[18]:


info_dig(df.failures)


# In[19]:


# Заполним пропуски средним значением
df.failures.fillna(round(df.failures.mean()), inplace=True)


# In[20]:


info_dig(df.studytime_granular)


# In[21]:


# Заполним пропуски средним значением
df.studytime_granular.fillna(round(df.studytime_granular.mean()), inplace=True)


# In[22]:


info_dig(df.famrel)


# In[23]:


# Исправим опечатку значения -1.0, вероятно имелось ввиду 1.0
df.famrel = df.famrel.apply(lambda x: abs(x) if x < 0 else x)


# In[24]:


# Заполним пропуски средним значением
df.famrel.fillna(round(df.famrel.mean()), inplace=True)


# In[25]:


info_dig(df.freetime)


# In[26]:


# Заполним пропуски средним значением
df.freetime.fillna(round(df.freetime.mean()), inplace=True)


# In[27]:


info_dig(df.goout)


# In[28]:


# Заполним пропуски средним значением
df.goout.fillna(round(df.goout.mean()), inplace=True)


# In[29]:


info_dig(df.health)


# In[30]:


# Заполним пропуски средним значением
df.health.fillna(round(df.health.mean()), inplace=True)


# In[31]:


info_dig(df.absences)


# In[32]:


# Вероятно, значения 212 и 385 являются ошибками, т.к. выходят за рамки кол-ва учебных дней и кол-ва дней в году соответственно.
# Удалим их из датасета.
df = df[~df.absences.isin([212.0,385.0])]


# In[33]:


# Заполним пропуски медианным значением, т.к. достаточно большой разброс
df.absences.fillna(round(df.absences.median()), inplace=True)


# In[34]:


# Посмотрим еще раз не получившийся результат
info_dig(df.absences)


# In[35]:


# Определим выбросы и удалим их
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
                                                                              label = 'IQR')
plt.legend();


# In[36]:


# Выбросами считаем значения больше 30, 
# т.к. пропуск более 30 учебных дней скорее всего приведет к переводу ученика на домашнее обучение.
# Удалим выбросы

df = df[df.absences.between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)]


# In[37]:


info_dig(df.score)


# In[38]:


# Пропуски в целевой переменной можно удалить
df = df[~df.score.isna()]


# ### Просмотр строковых столбцов

# In[39]:


info_object(df.school)


# In[40]:


info_object(df.sex)


# In[41]:


info_object(df.address)


# In[42]:


# Чтобы заполнить пропуски найдем среднее время до школы для городских и загородных учеников
df.groupby(['address'])['traveltime'].mean().reset_index()


# In[43]:


# Т.к. у нас круглые значения в traveltime - будем считать, что 2 и более - это загородные ученики. 
# Заполним пропуски следующим образом:

pd.set_option('mode.chained_assignment', None) # Чтобы не ругался

df = df.reset_index() # Сбросим индекс учеников, иначе алерт
df.address.fillna(0, inplace=True) # Заполним пропуски 0

for i in range(0, len(df)):
    if df.address[i] == 0:
        if df.traveltime[i] > 1.0:
            df.address[i] = 'R'
        else:
            df.address[i] = 'U'


# In[44]:


info_object(df.famsize)


# In[45]:


# Заполним пропуски самым частовстречаемым значением
# Логическое предположение: родители живут вместе - больше семья
# Для этого обратимся к параметру Pstatus (Совместное проживание родителей)
df.groupby(['pstatus'])['famsize'].value_counts() 


# In[46]:


# Предположение не оправдалось, во всех группах преобладает значение GT3, им и заполним пропуски
df.famsize.fillna('GT3', inplace=True)


# In[47]:


info_object(df.pstatus)


# In[48]:


# Заполним пропуски самым частовстречаемым значением
df.pstatus.fillna('T', inplace=True)


# In[49]:


info_object(df.mjob)


# In[50]:


# Заполним пропуски неопределенным значением
df.mjob.fillna('other', inplace=True)


# In[51]:


info_object(df.fjob)


# In[52]:


# Заполним пропуски неопределенным значением
df.fjob.fillna('other', inplace=True)


# In[53]:


info_object(df.reason)


# In[54]:


# Заполним пропуски самым частовстречаемым значением
df.reason.fillna('course', inplace=True)


# In[55]:


info_object(df.guardian)


# In[56]:


# Заполним пропуски самым частовстречаемым значением
df.guardian.fillna('mother', inplace=True)


# In[57]:


info_object(df.schoolsup)


# In[58]:


# Заполним пропуски самым частовстречаемым значением
df.schoolsup.fillna('no', inplace=True)


# In[59]:


info_object(df.famsup)


# In[60]:


# Заполним пропуски самым частовстречаемым значением
df.famsup.fillna('yes', inplace=True)


# In[61]:


info_object(df.paid)


# In[62]:


# Заполним пропуски самым частовстречаемым значением
df.paid.fillna('no', inplace=True)


# In[63]:


info_object(df.activities)


# In[64]:


# Значения близкие, хочется распределить пропуски равномерно
# Заполним пропуски поочереди каждым значением

df.activities.fillna(0, inplace=True) # Заполним пропуски 0

for i in range(0, len(df)):
    counter = 0
    if df.activities[i] == 0:
        if counter % 2 == 0:
            df.activities[i] = 'yes'
            counter += 1
        else:
            df.activities[i] = 'no'
            counter += 1


# In[65]:


info_object(df.nursery)


# In[66]:


# Заполним пропуски самым частовстречаемым значением
df.nursery.fillna('yes', inplace=True)


# In[67]:


info_object(df.higher)


# In[68]:


# Заполним пропуски самым частовстречаемым значением
df.higher.fillna('yes', inplace=True)


# In[69]:


info_object(df.internet)


# In[70]:


# Заполним пропуски самым частовстречаемым значением
df.internet.fillna('yes', inplace=True)


# In[71]:


info_object(df.romantic)


# In[72]:


# Заполним пропуски самым частовстречаемым значением
df.romantic.fillna('no', inplace=True)


# In[73]:


# Финальная проверка на пропуски
df.isna().sum()


# # Поиск зависимостей
# Посмотрим корреляцию числовых значений

# In[74]:


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

# In[75]:


def get_boxplot(column):
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.boxplot(x=column, y='score',
                data=df.loc[df.loc[:, column].isin(df.loc[:, column].value_counts().index[:10])], ax=ax)
    plt.xticks(rotation=0)
    ax.set_title('Boxplot for ' + column)
    plt.show()


# In[76]:


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

# In[77]:


def get_stat_dif(column):
    cols = df.loc[:, column].value_counts().index[:20]
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(df.loc[df.loc[:, column] == comb[0], 'score'],
                     df.loc[df.loc[:, column] == comb[1], 'score']).pvalue \
                <= 0.05/len(combinations_all):  # Учли поправку Бонферони
            print('Найдены статистически значимые различия для колонки', column)
            break


# In[78]:


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

# In[79]:


model = df[['school','age','sex','address', 'medu', 'fedu', 'mjob', 'fjob', 'studytime', 'failures','romantic','schoolsup','higher', 'goout', 'absences', 'score']]
model


# In[ ]:




