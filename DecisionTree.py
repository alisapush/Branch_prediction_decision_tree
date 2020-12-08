#подключение библиотек
from sklearn import tree
import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 

#test_records_count = 33000
#чтение таблицы
data = pd.read_csv('/Users/alisapushnova/Repos/MIPT-MIPS/mipt-mips/build/logsss4.txt') #, skiprows = 0, nrows = test_records_count)

#в переменной X передаем признаки с вопросами
X = data[['pc','target','jump_double_b_predictor','jump_streak']]

#в y - ответ для обучения
y = data['is_taken']

#разбиваем на тестовые и тренировочные признаки и ответы в соотношении 67 к 33.
#random_state - задает зерно случайности при выборке строк. 
#если его не задать, то при каждом запуске в тест и train будут попадать разные строки.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 0)

print('Train records count:', len(X_train))
print('Test  records count:', len(X_test))
#print(type(y_test))

#создаем объект - решающее дерево из библиотеки sklearn
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 20, random_state = 0)

#производим обучение дерева на заготовленных вопросах и ответах:
clf.fit(X_train, y_train)

#calculate train score
train_results_score = clf.score(X_train, y_train) 

#calculate TEST score
test_results_score = clf.score(X_test, y_test) 

#PREDICT test records
predicts = clf.predict(X_test)

#handmade counting correctly predicted values
y_test_list = y_test.tolist()
same = 0
for i in range(len(predicts)):
     same += 1 if predicts[i] == y_test_list[i] else 0

print('Train score:', train_results_score)
print('Test score: ', test_results_score)

print('Success prediction score: ', same, '/', len(predicts))

#посмотрим на получившееся дерево
tree.plot_tree(clf, feature_names = list(X),
     filled = True)
plt.show()
