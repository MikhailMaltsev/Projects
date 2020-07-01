#!/usr/bin/env python
# coding: utf-8

# In[70]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


# In[71]:


data = pd.read_csv('movie_bd_v5.csv')
data.sample(5)


# In[72]:


data.describe()


# # Предобработка

# In[73]:


answers = {} # создадим словарь для ответов

# добавим столбец profit в датафрейм
data['profit'] = data['revenue'] - data['budget'] # посчитаем прибыль фильмов
data.append(data['profit']) # добавим стобец с прибылью в датафрейм

# функция счетчик для разделения слитых позиций
def counter(df, x): 
    joined = df[x].str.cat(sep='|') # склеиваем все значения столбца в строку для приведения к единому виду
    splited = pd.Series(joined.split('|')) # разделяем значения и кладем в Series
    counted = splited.value_counts(ascending=False)
    return counted


# # 1. У какого фильма из списка самый большой бюджет?

# Использовать варианты ответов в коде решения запрещено.    
# Вы думаете и в жизни у вас будут варианты ответов?)

# In[74]:


answers['1'] = '723. Pirates of the Caribbean: On Stranger Tides (tt1298650)'


# In[75]:


data[data.budget == data.budget.max()]


# ВАРИАНТ 2

# In[76]:


data.loc[data.budget.sort_values(ascending = False).keys()[0]]


# # 2. Какой из фильмов самый длительный (в минутах)?

# In[77]:


answers['2'] = '1157. Gods and Generals (tt0279111)'


# In[78]:


data[data.runtime == data.runtime.max()]


# # 3. Какой из фильмов самый короткий (в минутах)?
# 
# 
# 
# 

# In[79]:


answers['3'] = '768. Winnie the Pooh (tt1449283)'


# In[80]:


data[data.runtime == data.runtime.min()]


# # 4. Какова средняя длительность фильмов?
# 

# In[81]:


answers['4'] = 110


# In[82]:


round(data.runtime.mean())


# # 5. Каково медианное значение длительности фильмов? 

# In[83]:


answers['5'] = 107


# In[84]:


round(data.runtime.median())


# # 6. Какой самый прибыльный фильм?
# #### Внимание! Здесь и далее под «прибылью» или «убытками» понимается разность между сборами и бюджетом фильма. (прибыль = сборы - бюджет) в нашем датасете это будет (profit = revenue - budget) 

# In[85]:


answers['6'] = '239. Avatar (tt0499549)'


# In[86]:


# добавление столбца profit в блоке предобработки
data[data.profit == data.profit.max()]


# # 7. Какой фильм самый убыточный? 

# In[87]:


answers['7'] = '1245. The Lone Ranger (tt1210819)'


# In[88]:


data[data.profit == data.profit.min()]


# # 8. У скольких фильмов из датасета объем сборов оказался выше бюджета?

# In[89]:


answers['8'] = 1478


# In[90]:


len(data.query('profit > 0'))


# # 9. Какой фильм оказался самым кассовым в 2008 году?

# In[91]:


answers['9'] = '599. The Dark Knight (tt0468569)'


# In[92]:


data[data.profit == data[data.release_year == 2008].profit.max()]


# # 10. Самый убыточный фильм за период с 2012 по 2014 г. (включительно)?
# 

# In[93]:


answers['10'] = '1245. The Lone Ranger (tt1210819)'


# In[94]:


data[data.profit == data[(data.release_year >= 2012) & (data.release_year <= 2014)].profit.min()]


# # 11. Какого жанра фильмов больше всего?

# In[95]:


answers['11'] = 'Drama'


# In[96]:


counter(data,'genres').keys()[0]


# ВАРИАНТ 2

# In[ ]:





# # 12. Фильмы какого жанра чаще всего становятся прибыльными? 

# In[97]:


answers['12'] = 'Drama'


# In[98]:


#разделяем скленные жанры по | и передаем в новый датафрейм
pd.DataFrame(data[data.profit > 0].genres.str.split('|').tolist()).stack().value_counts().keys()[0]


# # 13. У какого режиссера самые большие суммарные кассовые сборы?

# In[99]:


answers['13'] = 'Peter Jackson'


# In[100]:


data.groupby(['director'])['revenue'].sum().sort_values(ascending=False).keys()[0]


# # 14. Какой режисер снял больше всего фильмов в стиле Action?

# In[101]:


answers['14'] = 'Robert Rodriguez'


# In[102]:


action_films = data[data['genres'].map(lambda x: True if 'Action' in x else False)]
counter(action_films,'director').keys()[0] # передаем в функцию счетчик (код в предобработке)


# # 15. Фильмы с каким актером принесли самые высокие кассовые сборы в 2012 году? 

# In[103]:


answers['15'] = 'Chris Hemsworth'


# In[104]:


films2012 = pd.DataFrame(data[data.release_year == 2012]).reset_index()
actors = pd.DataFrame(films2012['cast'].str.split('|').tolist()).stack().to_frame().reset_index()
actors.columns = ['index', 'index_1', 'name']
films2012_revenue = films2012['revenue'].reset_index()
final = actors.merge(films2012_revenue, on = 'index', how = 'left')
final[['name', 'revenue']].groupby('name')[['revenue']].sum().sort_values('revenue', ascending = False).head(1)


# # 16. Какой актер снялся в большем количестве высокобюджетных фильмов?

# In[105]:


answers['16'] = 'Matt Damon'


# In[106]:


counter(data[data['budget'] > data.budget.mean()],'cast').keys()[0]


# # 17. В фильмах какого жанра больше всего снимался Nicolas Cage? 

# In[107]:


answers['17'] = 'Action'


# In[108]:


NC_films = data[data['cast'].map(lambda x: True if 'Nicolas Cage' in x else False)]
counter(NC_films,'genres').keys()[0]


# # 18. Самый убыточный фильм от Paramount Pictures

# In[109]:


answers['18'] = '925. K-19: The Widowmaker (tt0267626)'


# In[110]:


PP_films = data[data['production_companies'].map(lambda x: True if 'Paramount Pictures' in x else False)]
PP_films[PP_films.profit == PP_films.profit.min()]


# # 19. Какой год стал самым успешным по суммарным кассовым сборам?

# In[111]:


answers['19'] = 2015


# In[112]:


data.groupby(['release_year'])['revenue'].sum().sort_values(ascending=False).keys()[0]


# # 20. Какой самый прибыльный год для студии Warner Bros?

# In[113]:


answers['20'] = 2014


# In[114]:


WB_films = data[data.production_companies.map(lambda x: True if 'Warner Bros' in x else False)]
WB_films.groupby(['release_year'])['profit'].sum().sort_values(ascending=False).keys()[0]


# # 21. В каком месяце за все годы суммарно вышло больше всего фильмов?

# In[115]:


answers['21'] = 'Сентябрь'


# In[116]:


df_month = pd.DataFrame(data.release_date.str.split('/',2).map((lambda x: x[0])))
df_month.release_date.value_counts().keys()[0]


# # 22. Сколько суммарно вышло фильмов летом? (за июнь, июль, август)

# In[117]:


answers['22'] = 450


# In[118]:


df_month = pd.DataFrame(data.release_date.str.split('/',2).map((lambda x: x[0])))
data[df_month.release_date.isin(['6','7','8'])].release_date.value_counts().sum()


# # 23. Для какого режиссера зима – самое продуктивное время года? 

# In[119]:


answers['23'] = 'Peter Jackson'


# In[120]:


df_month = pd.DataFrame(data.release_date.str.split('/',2).map(lambda x: x[0]))
df_month['director'] = data['director']
winter = df_month[df_month.release_date.isin(['1','2','12'])]
counter(winter,'director').keys()[0]


# # 24. Какая студия дает самые длинные названия своим фильмам по количеству символов?

# In[121]:


answers['24'] = 'Four By Two Productions'


# In[122]:


data['title_lenght'] = pd.DataFrame(data.original_title.map(lambda x: len(x)))
new_data = pd.DataFrame(data['production_companies'].str.split('|').tolist()).stack().to_frame().reset_index()
new_data.columns = ['index', 'index_1', 'company']
data_1 = data.reset_index()
data_join = data_1.merge(new_data, on = 'index', how = 'left')
data_join[['company', 'title_lenght']].groupby('company').mean().sort_values('title_lenght', ascending = False).head(1)


# # 25. Описание фильмов какой студии в среднем самые длинные по количеству слов?

# In[123]:


answers['25'] = 'Four By Two Productions'


# In[124]:


data['overview_lenght'] = pd.DataFrame(data.overview.map(lambda x: len(x)))
new_data = pd.DataFrame(data['production_companies'].str.split('|').tolist()).stack().to_frame().reset_index()
new_data.columns = ['index', 'index_1', 'company']
data_1 = data.reset_index()
data_join = data_1.merge(new_data, on = 'index', how = 'left')
data_join[['company', 'overview_lenght']].groupby('company').mean().sort_values('overview_lenght', ascending = False).head(1)


# # 26. Какие фильмы входят в 1 процент лучших по рейтингу? 
# по vote_average

# In[125]:


answers['26'] = 'Inside Out, The Dark Knight, 12 Years a Slave'


# In[126]:


top1p = data[['original_title','vote_average']].sort_values('vote_average', ascending = False).head(round(len(data.index)/100))
input_str = input()
test = input_str.split(', ')
test_count = 0
for i in range(0,len(test)):
    for j in range(0,len(top1p)):
        if test[i] in top1p.original_title.iloc[j]: 
            test_count += 1
        else: 
            continue
if test_count == len(test):
    print('Эти фильмы входят в 1 процент лучших по рейтингу')
else: print('Эти фильмы НЕ входят в 1 процент лучших по рейтингу')


# # 27. Какие актеры чаще всего снимаются в одном фильме вместе?
# 

# In[131]:


answers['27'] = 'Daniel Radcliffe Rupert Grint'


# In[134]:


from itertools import combinations
actor_list = data.cast.str.split('|').tolist()
combo_list = []
for i in actor_list:
    for j in combinations(i, 2):
        combo_list.append(' '.join(j))
combo_list = pd.DataFrame(combo_list)
combo_list.columns = ['actor_combinations']
combo_list.actor_combinations.value_counts().head(1).keys()[0]


# # Submission

# In[132]:


answers


# In[130]:


len(answers)

