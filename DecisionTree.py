#подключение библиотек
from sklearn import tree
import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
import seaborn as sns

#чтение таблицы
data = pd.read_csv('/Users/alisapush0va/Repos/MIPT-MIPS/mipt-mips/build/logsss4.txt')

#в переменной X передаем признаки с вопросами
X = data[['jump_double_b_predictor','jump_streak','pc','target']]

#в y - ответ для обучения
y = data['is_taken']

#создаем объект - решающее дерево из библиотеки sklearn
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_leaf_0des = 20, random_state = 0)

#производим обучение дерева на заготовленных вопросах и ответах:
clf.fit(X, y)

#посмотрим на получившееся дерево
tree.plot_tree(clf, feature_names = list(X),
     # class_names = ['0', '1'],
     filled = True)
plt.show()


