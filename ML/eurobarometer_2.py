import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import Imputer


df = pd.read_csv('/Users/carlosarcila/Dropbox/CARLOS ARCILA DOCS/USAL/Investigación/ML Refugees Acceptance/ZA6788/ZA6788_v1-2-0.csv', header=0, sep= ';')

#Trabajamos con cada una de las variables, definimos los NaN y vemos sus valores únicos
print('Variables:', df.columns)
#Pais
print('Países:', df['country'].unique())
#Tipo de población
df['type_community'] = df['type_community'].replace('DK', np.NaN)
print('Tipo de población:', df['type_community'].unique())
#Estado civil
df['marital_status'] = df['marital_status'].replace('Single: without children', 'Single')
df['marital_status'] = df['marital_status'].replace('(Re-)Married: children this marriage', 'Married')
df['marital_status'] = df['marital_status'].replace('Widow: with children', 'Widow')
df['marital_status'] = df['marital_status'].replace('Single liv w partner: childr this union', 'Partnership')
df['marital_status'] = df['marital_status'].replace('Single liv w partner: without children', 'Partnership')
df['marital_status'] = df['marital_status'].replace('Divorced/Separated: without children', 'Divorced/Separated')
df['marital_status'] = df['marital_status'].replace('(Re-)Married: without children', 'Married')
df['marital_status'] = df['marital_status'].replace('Widow: without children', 'Widow')
df['marital_status'] = df['marital_status'].replace('Divorced/Separated: with children', 'Divorced/Separated')
df['marital_status'] = df['marital_status'].replace('(Re-)Married: children prev marriage', 'Married')
df['marital_status'] = df['marital_status'].replace('Single liv w partner: childr this/prev union', 'Partnership')
df['marital_status'] = df['marital_status'].replace('Single: with children', 'Single')
df['marital_status'] = df['marital_status'].replace('Single liv w partner: childr prev union', 'Single')
df['marital_status'] = df['marital_status'].replace('(Re-)Married: children this/prev marriage', 'Married')
df['marital_status'] = df['marital_status'].replace('Other (SPONT.)', np.NaN)
print('Estado civil:', df['marital_status'].unique())
#Años de educación
df['educatational'] = df['educatational'].replace('0', np.NaN)
df['educatational'] = df['educatational'].replace('97', np.NaN)
df['educatational'] = df['educatational'].replace('98', np.NaN)
df['educatational'] = df['educatational'].replace('99', np.NaN)
#Imputamos medias a los NaN de esta variable para no perder tantos casos
imr = Imputer(missing_values='NaN', strategy='mean', axis=1)
imr = imr.fit(df['educatational'])
imputed_data = imr.transform(df['educatational'].values).T
df["educatational"]=imputed_data

print('Años de educación:', df['educatational'].unique())
#Género
print('Género:', df['gender'].unique())
#Edad
print('Edad:', df['age'].unique())
#Ocupación
df['occupation'] = df['occupation'].replace('Student', 'Not active')
df['occupation'] = df['occupation'].replace('Employed position, at desk', 'Employed')
df['occupation'] = df['occupation'].replace('Unskilled manual worker, etc.', 'Employed')
df['occupation'] = df['occupation'].replace('Retired, unable to work', 'Not active')
df['occupation'] = df['occupation'].replace('Employed position, service job', 'Employed')
df['occupation'] = df['occupation'].replace('Unemployed, temporarily not working', 'Unemployed')
df['occupation'] = df['occupation'].replace('Owner of a shop, craftsmen, etc.', 'Employed')
df['occupation'] = df['occupation'].replace('Middle management, etc.', 'Employed')
df['occupation'] = df['occupation'].replace('Skilled manual worker', 'Employed')
df['occupation'] = df['occupation'].replace('Employed professional (employed doctor, etc.)', 'Employed')
df['occupation'] = df['occupation'].replace('General management, etc.', 'Employed')
df['occupation'] = df['occupation'].replace('Employed position, travelling', 'Employed')
df['occupation'] = df['occupation'].replace('Business proprietors, etc.', 'Employed')
df['occupation'] = df['occupation'].replace('Supervisor', 'Employed')
df['occupation'] = df['occupation'].replace('Professional (lawyer, etc.)', 'Employed')
df['occupation'] = df['occupation'].replace('Responsible for ordinary shopping, etc.', 'Employed')
df['occupation'] = df['occupation'].replace('Farmer', 'Employed')
df['occupation'] = df['occupation'].replace('Fisherman', 'Employed')
print('Ocupación:', df['occupation'].unique())

#Para contar los casos de cada categoría en una variable:
#print(pd.value_counts(df['occupation']))

#Número de miembros en el hogar
df['household_compostion'] = df['household_compostion'].replace('99', np.NaN)
print('Número de miembros en el hogar:', df['household_compostion'].unique())

#Apoyo a los refugiados
#Agrupo los valores de la taget en 3: 1=Apoyan, 2=No apoyan, 3=No saben, no hay respuesta
df['support_refugees'] = df['support_refugees'].replace('2', 1)
df['support_refugees'] = df['support_refugees'].replace('3', 2)
df['support_refugees'] = df['support_refugees'].replace('4', 2)
df['support_refugees'] = df['support_refugees'].replace('5', np.NaN)
df['support_refugees'] = df['support_refugees'].replace('9', np.NaN)
print('Apoyo a los refugiados:', df['support_refugees'].unique())


#Quitamos todos los casos con algón valor perdido
print('Valores perdidos por variable:\n', df.isnull().sum()) #Si queremos saber el número de casos con algún valor perdido de cada variable
df = df.dropna()
print('Total de casos válidos: ', len(df))

#Convertimos las variables categóricas en variables dummy 0-1, salvo la target
df = pd.get_dummies(df[['country', 'type_community', 'marital_status', 'educatational','gender', 'age', 'occupation', 'household_compostion', 'support_refugees']])
df = df[[
       'country_BALGARIJA', 'country_BELGIQUE', 'country_CESKA REPUBLIKA',
       'country_DANMARK', 'country_DEUTSCHLAND OST',
       'country_DEUTSCHLAND WEST', 'country_EESTI', 'country_ELLADA',
       'country_ESPANA', 'country_FRANCE', 'country_GREAT BRITAIN',
       'country_HRVATSKA', 'country_IRELAND', 'country_ITALIA',
       'country_KYPROS', 'country_LATVIA', 'country_LIETUVA',
       'country_LUXEMBOURG', 'country_MAGYARORSZAG', 'country_MALTA',
       'country_NEDERLAND', 'country_NORTHERN IRELAND', 'country_POLSKA',
       'country_PORTUGAL', 'country_ROMANIA', 'country_SLOVENIJA',
       'country_SLOVENSKA REPUBLIC', 'country_SUOMI', 'country_SVERIGE',
       'country_ÖSTERREICH', 'type_community_Large town',
       'type_community_Rural area or village',
       'type_community_Small/middle town', 'age', 'educatational', 'household_compostion', 'marital_status_Divorced/Separated',
       'marital_status_Married', 'marital_status_Partnership',
       'marital_status_Single', 'marital_status_Widow', 'gender_Man',
       'gender_Woman', 'occupation_Employed', 'occupation_Not active',
       'occupation_Unemployed', 'support_refugees']]
#Organizo las columnas y pido me dé el listado de feautures para ver la indexación de cada una# :
features = [df.columns]
features = pd.DataFrame(features).T
print(features)

print('Número de casos', len(df))
print('Número de casos por clase de la variable target', df['support_refugees'].value_counts())

#Pasamos a matrix
df = df.as_matrix()

#creamos los vectores para crear predictores y target
#informative = range(0, 46)
#informative = list(informative)
#print(informative) #to generate the list
X = df[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]]
y = df[:, 46]

#Dividimos en training dataset y testing dataset
#from sklearn.cross_validation import train_test_split #He cambiado el modulo porque ha cambiado en la última versión de sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#print(y_train, y_test, X_train, X_test)




from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train) #Entiendo que se ajustan los parámetros con con solo array para poder comparar
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#print(X_train_std, X_test_std)


#Creamos un modelo de ML con el algortimo de Regresión Logística
from sklearn.linear_model import LogisticRegression
#El parámetro C es inverso a la regularización del modelo: Subir C es bajar la fuerza de la regularización y viceversa.
#Entre más se regularice (baje C) más disminuyen los pesos en el modelo
#Con C=1000.0 me da una precisión de 98%, con C=1.0 baja a 80%, con C=10000 sigue dando 98%...
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
print('---')
print('ML Regresión logística')
#print(lr.fit(X_train_std, y_train))

y_pred = lr.predict(X_test_std)
from sklearn.metrics import accuracy_score
print('Número de predicciones: %d' % len(y_test))
print('Número de sujetos mal clasificados: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


#Estimamos coeficiencies
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
print('Coefficients: \n', lr.coef_)
coef = lr.coef_
print(type(coef))



#Comparamos los resultados predecidos
#print(*y_pred, sep=',')
#Con los del test dataset
#print(*y_test, sep=',')
#O podemos predecir las probabilidades de un caso concreto
#print(lr.predict_proba(X_test_std[0]))


#Creamos un modelo de ML con el algortimo de SVM
print('---')
print('ML SVM')
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=0) #Activamos este parámetro si queremos probabilidades probability= True
svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)
print(svm.fit(X_train_std, y_train))
print('Número de predicciones: %d' % len(y_test))
print('Número de sujetos mal clasificados: %d' % (y_test != y_pred) .sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#Creamos un modelo de ML con SVM y KERNELS

#Primero con gamma=0,2
print('--')
print('ML SVM con Kernel y gamma=0.2')
from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)
print(svm.fit(X_train_std, y_train))
print('Número de predicciones: %d' % len(y_test))
print('Número de sujetos mal clasificados: %d' % (y_test != y_pred) .sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#Luego con gamma=100
print('--')
print('ML SVM con Kernel y gamma=100')
from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=0, gamma=100, C=1.0)
svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)
print(svm.fit(X_train_std, y_train))
print('Número de predicciones: %d' % len(y_test))
print('Número de sujetos mal clasificados: %d' % (y_test != y_pred) .sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


#Comparamos los resultados predecidos
#print(*y_pred, sep=',')
#Con los del test dataset
#print(*y_test, sep=',')

#Creamos un modelo de ML con el algortimo de Árboles de Decisión
#No hacer falta usar variables transformadas
print('---')
print('ML Árboles de Decisión. Con entropia 3')
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print(tree.fit(X_train, y_train))
print('Número de predicciones: %d' % len(y_test))
print('Número de sujetos mal clasificados: %d' % (y_test != y_pred) .sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#creamos el gráfico
#from sklearn.tree import export_graphviz
#export_graphviz(tree, out_file='tree.dot', feature_names=['País','Sexo', 'Edad'])
#Para convertirlo en png instalar grapfviz y ejecutar desde la terminal > dot -Tpng tree.dot -o tree.png



#Creamos un modelo de ML con el algortimo de Árboles de Decisión con Ensamblaje, RamdonForest
#No hacer falta usar variables transformadas
print('---')
print('ML Ramdon Forest. Con 10 árboles')
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
print(forest.fit(X_train, y_train))
print('Número de predicciones: %d' % len(y_test))
print('Número de sujetos mal clasificados: %d' % (y_test != y_pred) .sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


#Creamos un modelo de ML con el algortimo de KNN
print('---')
print('ML KNN. Con 5 vecinos y distancia euclidea')
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test)

print(knn.fit(X_train_std, y_train))
print('Número de predicciones: %d' % len(y_test))
print('Número de sujetos mal clasificados: %d' % (y_test != y_pred) .sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#Predicciones
#Con  LR que ha dado el mejor resultado

sujeto1 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 50, 10, 2, 0, 1, 0, 0, 0,  0, 1, 0, 1, 0,])
sujeto1_std = sc.transform(sujeto1)
print(lr.predict_proba(sujeto1_std))

sujeto2 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 50, 30, 2, 0, 1, 0, 0, 0,  0, 1, 0, 1, 0,])
sujeto2_std = sc.transform(sujeto2)
print(lr.predict_proba(sujeto2_std))