import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Список стоп-слов для русского языка
stopwords_rus = stopwords.words('russian')

# df_rus = pd.read_csv("dataset_with_syn.txt")
df_rus = pd.read_csv("dataset.txt")
df_rus = df_rus.dropna()

for i in range(df_rus.shape[0]):
  try: 
    # Приводим текст к нижнему регистру
    string = str(df_rus['Ответ'][i])
    string = string.lower()

    # Уберём неинформативные данные (оставим только текст)
    string = re.sub("([^0-9A-Za-zА-ЯЁа-яё \t])|(\w+:\/\/\S+)", "", string)
    # print(f"удаление неинформативных данных\n: {string}")

    # Токенизируем текст
    string = word_tokenize(string)
    # print(f"Токенизация текста\n: {string}")

    # Удалим стоп-слова
    string_withoutstop = [word for word in string if word not in stopwords_rus]
    # print(f"Удаление стоп-слов\n: {string_withoutstop}")

    # Лемматизируем (приведем к исходной форме) слова
    string = [WordNetLemmatizer().lemmatize(word) for word in string_withoutstop]
    # print(f"Лемматизация текста\n: {string}")

    df_rus['Ответ'][i] = str(string)

  except:
    pass
print(df_rus['Ответ'])
df_rus['Ключевое слово'] = df_rus['Ключевое слово'].str.replace('"', '')
df_rus['Ключевое слово'] = df_rus['Ключевое слово'].str.replace("'", '')

 # Разделение набора данных на тренировочные и тестовые части
test_size = 0.2
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(df_rus['Ответ'], df_rus['Ключевое слово'], test_size=test_size, random_state=seed)

# Настройка параметров оценивания алгоритма
num_folds = 10
n_estimators = 100
scoring = 'accuracy'

# Векторизуем
vectorizer = CountVectorizer(ngram_range=(1, 3))
# векторизуем обучающую выборку
X_train = vectorizer.fit_transform(X_train)
# и тестовую
X_test = vectorizer.transform(X_test)

# Оценивание эффективности выполнения каждого алгоритма
scores = []
names = []
results = []
predictions = []
msg_row = []

# Обучение модели. Линейный метод опорных векторов
model = LinearSVC()
kfold = KFold(n_splits=num_folds)
cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
m_fit = model.fit(X_train, Y_train)
m_predict = model.predict(X_test)
m_score = model.score(X_test, Y_test)
msg = "%s: train = %.3f (%.3f) / test = %.3f" % ("LinearSVC", cv_results.mean(), cv_results.std(), m_score)
print(msg)

# Сохранение модели 
filename = 'LSVC_model.sav'
pickle.dump(m_fit, open(filename, 'wb')) 
  
# Загрузка модели
load_model = pickle.load(open(filename, 'rb')) 

X_input = str(input())
lst_of_frases = X_input.split(";")
df = pd.DataFrame(lst_of_frases)

for i in range(df.shape[0]):
  try: 
    # Приводим текст к нижнему регистру
    string = str(df[0][i])
    string = string.lower()

    # Уберём неинформативные данные (оставим только текст)
    string = re.sub("([^0-9A-Za-zА-ЯЁа-яё \t])|(\w+:\/\/\S+)", "", string)
    # print(f"удаление неинформативных данных\n: {string}")

    # Токенизируем текст
    string = word_tokenize(string)
    # print(f"Токенизация текста\n: {string}")

    # Удалим стоп-слова
    string_withoutstop = [word for word in string if word not in stopwords_rus]
    # print(f"Удаление стоп-слов\n: {string_withoutstop}")

    # Лемматизируем (приведем к исходной форме) слова
    string = [WordNetLemmatizer().lemmatize(word) for word in string_withoutstop]
    # print(f"Лемматизация текста\n: {string}")

    df[0][i] = str(string)

  except:
    pass

# Векторизуем входящие данные
X_input = vectorizer.transform(df[0])

# Используем модель для предсказания ключевых слов
y_input = pd.DataFrame(load_model.predict(X_input))
# print(y_input)

# Создаём ранжированный словарь
ranged_dict = dict(Counter(y_input))
print(f"Словарь с частотностью ключевых слов: {ranged_dict}")

comment_words = ''           
values = y_input[0].values

for val in values: 
    val = str(val) 
    tokens = val.split() 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
    comment_words += ' '.join(tokens)+' '

facecolor = 'black'
wordcloud = WordCloud(width=1000, height=600, 
            background_color=facecolor,
            min_font_size=10).generate(comment_words)
            
plt.figure(figsize=(10,6), facecolor=facecolor) 
plt.imshow(wordcloud) 
plt.axis('off') 
plt.tight_layout(pad=2)
plt.show()
