#!/usr/bin/env python
# coding: utf-8

# # Введение

# Нас (возможно и вас) пригласили поучаствовать в одном из проектов UNICEF — международного подразделения ООН, чья миссия состоит в повышении уровня благополучия детей по всему миру. 
# 
# Суть проекта — отследить влияние условий жизни учащихся в возрасте от 15 до 22 лет на их успеваемость по математике, чтобы на ранней стадии выявлять студентов, находящихся в группе риска.

# In[52]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind


# # Ознакомление
# Для начала загрузим файл в датафрейм (ДФ), переименуем некоторые колонки для удобства и посмотрим развернутый ДФ.

# In[79]:


students = pd.read_csv('stud_math.csv')
students.rename(columns={'studytime, granular': 'studytime_granular',
                         'Pstatus': 'pstatus', 'Medu': 'medu', 'Fedu': 'fedu',
                         'Mjob': 'mjob', 'Fjob': 'fjob'}, inplace=True)
pd.set_option('max_columns', None)


# In[54]:


students.sample(10)


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

# In[55]:


students.info()


# В датасете всего 30 колонок: 13 числовых колонок и 17 строковых. 
# Датасете содержит данные данные о 395 учениках.
# Во всех колонках, кроме school, sex, age есть пустые значения.

# In[106]:


type(students.sex)


# # Очистка данных

# In[110]:


# Функция для получения быстрой справки о данных в колонках
def info(x):
    print(pd.DataFrame(x.value_counts()))
    print('Пропущенных значений -', x.isnull().values.sum())
    x.hist()


# ### Просмотр числовыx столбцов

# In[111]:


info(students.age)


# In[112]:


info(students.medu)


# In[59]:


info(students.fedu)


# Исправим опечатку значения 40.0, вероятно имелось ввиду 4.

# In[60]:


students.fedu = students.fedu.apply(lambda x: x/10 if x > 9 else x)


# In[61]:


info(students.traveltime)


# In[62]:


info(students.studytime)


# In[63]:


info(students.failures)


# In[64]:


info(students.studytime_granular)


# In[65]:


info(students.famrel)


# Исправим опечатку значения -1.0, вероятно имелось ввиду 1.0

# In[66]:


students.famrel = students.famrel.apply(lambda x: abs(x) if x < 0 else x)


# In[67]:


info(students.freetime)


# In[68]:


info(students.goout)


# In[69]:


info(students.health)


# In[70]:


info(students.absences)


# Вероятно, значения 212 и 385 являются ошибками, т.к. выходят за рамки кол-ва учебных дней и кол-ва дней в году соответственно. Удалим их из датасета.

# In[71]:


students = students[~students.absences.isin([212.0,385.0])]
len(students)


# Посмотрим еще раз не получившийся результат? определим выбросы и удалим их.

# In[72]:


info(students.absences)


# In[73]:


median = students.absences.median()
IQR = students.absences.quantile(0.75) - students.absences.quantile(0.25)
perc25 = students.absences.quantile(0.25)
perc75 = students.absences.quantile(0.75)
print('25-й перцентиль: {},'.format(perc25), 
      '75-й перцентиль: {},'.format(perc75), 
      "IQR: {}, ".format(IQR),
      "Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))
students.absences.loc[students.absences.between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)].hist(bins = 16, 
                                                                                              range = (0, 30),
                                                                                              label = 'IQR')
plt.legend();


# Выбросами считаем значения больше 30, т.к. пропуск более 30 учебных дней скорее всего приведет к переводу ученика на домашнее обучение.

# In[74]:


students = students.absences.loc[students.absences.between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)]
len(students)


# In[80]:


info(students.score)


# Числовые колонки содержат достаточно чистые данные, мало ошибок и пропущенных значений.

# ### Просмотр строковых столбцов

# In[81]:


info(students.school)


# In[82]:


info(students.sex)


# In[83]:


info(students.address)


# In[84]:


info(students.famsize)


# In[85]:


info(students.pstatus)


# In[86]:


info(students.mjob)


# In[87]:


info(students.fjob)


# In[88]:


info(students.reason)


# In[89]:


info(students.guardian)


# In[90]:


info(students.schoolsup)


# In[91]:


info(students.famsup)


# In[92]:


info(students.paid)


# In[93]:


info(students.activities)


# In[94]:


info(students.nursery)


# In[95]:


info(students.higher)


# In[96]:


info(students.internet)


# In[97]:


info(students.romantic)


# Строковые данные содержат большее кол-во пропусков.

# In[ ]:


# Можно было бы заменить все NaN на None, как обсуждалось в группе по проекту, но в таком случае перестает работать корреляция.
# students = students.where(pd.notnull(students), None)


# # Поиск зависимостей
# Посмотрим корреляцию числовых значений

# In[113]:


students.corr()


# Полная корреляция столбцов studytime и studytime_granular позволяют не брать последний в рассчет.
# 
# Образование родителей (medu и fedu) и кол-во внеучебных неудач (failures) больше других оказывают влияние на успеваемость. Т.е. чем выше образование родителей, тем более успешный ребенок в жизни в целом, в том числе и в учебе. 
# 
# Также можно увидеть корреляцию возраста (age) и кол-ва внеучебных неудач (failures). Чем страше ребенок, тем хуже успеваемость. Можно предположить, что более молодые родители имеют лучшее образование, либо растущие дети начинают больше времени уделять друзьям (gout) а не учебе (studytime).
# 
# Присутствует также логическая связь влияния пропусков занятий (absences) на успеваемость, эти данные мы тоже возьмем в модель.

# Построим графики и посмотрим на распределения строковых данных

# In[114]:


def get_boxplot(column):
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.boxplot(x=column, y='score',
                data=students.loc[students.loc[:, column].isin(students.loc[:, column].value_counts().index[:10])], ax=ax)
    plt.xticks(rotation=0)
    ax.set_title('Boxplot for ' + column)
    plt.show()


# In[115]:


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

# In[116]:


def get_stat_dif(column):
    cols = students.loc[:, column].value_counts().index[:20]
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(students.loc[students.loc[:, column] == comb[0], 'score'],
                     students.loc[students.loc[:, column] == comb[1], 'score']).pvalue \
                <= 0.05/len(combinations_all):  # Учли поправку Бонферони
            print('Найдены статистически значимые различия для колонки', column)
            break


# In[117]:


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

# In[118]:


model = students[['school','age','sex','address', 'medu', 'fedu', 'mjob', 'fjob', 'studytime', 'failures','romantic','schoolsup','higher', 'goout', 'absences', 'score']]
model


# In[ ]:




