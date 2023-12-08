## Importam librariile si functiile de care o sa avem nevoie

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

## Am ales datasetul de aici: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
## Este vorba despre predictia statusului acordarii de imprumut bancar catre o persoana, in functie de mai multe atribute.
## Am inclus fisierul de train de care avem nevoie, dar o sa fie incarcat si pe moodle.

df = pd.read_csv('/content/train_u6lujuX_CVtuZ9i.csv')

print(df.head(5).T)
print("\n<------------------------------------------------> Inainte de completare: ")
print(df.isnull().sum())

## Dupa cum vedem, avem foarte multe coloane care au valori nule, deci putem fie sa eliminam inregistrarile fie sa completam cu niste
# date care nu o sa afecteze.
# Am rulat programul in ambele cazuri si avem valori foarte similare pentru acuratete, deci o sa aleg cazul in care competam cu date.


## Completam datele numerice cu media coloanei respective.

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())

## Completam datele care nu sunt numerice cu cea mai comuna valoare care apare in coloana respectiva.

df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

print("\n<------------------------------------------------> Dupa completare: ")
print(df.isnull().sum())

## Eliminam coloanele de care nu avem nevoie in clasificare.

cols = ['CoapplicantIncome','Loan_ID']
df = df.drop(columns = cols,axis= 1)

## Convertim aici valorile care nu sunt numerice in valori numerice.
# Exemplu : Y --> 1 si N --> 0

cols = ['Gender', 'Married', 'Education','Self_Employed', 'Property_Area','Loan_Status','Dependents']
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])

## Ilustram proportionalitatea celor doua tipuri de inregistrari.

df['Loan_Status'].plot(kind='hist', edgecolor='black')

## Specificam intrarile pentru algoritmii de mai jos.

y = df['Loan_Status']
X = df.drop(columns=['Loan_Status'], axis = 1)

## Normalizam setul de date.

X = preprocessing.normalize(X)

## Scalam setul de date.

stdsc = StandardScaler()
X_std = stdsc.fit_transform(X)

## Impartim setul de date in train si test, deoarece avem fisier separat de test, dar nu avem statusul, deci nu putem verifica.

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.15, random_state=0)

print("\n<------------------------------------------------>\n")
print("Train Shape =======> ",X_train.shape)
print("Test Shape  =======> ",X_test.shape)

## Folosim SVM pentru clasificare.

SVM = svm.SVC(C=1.5, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_train,y_train)
predictions_SVM = SVM.predict(X_test)

## Folosim LogisticRegression pentru clasificare.

log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
predictions_LR = log_reg.predict(X_test)

## Folosim DecisionTree pentru clasificare.

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
predictions_CLF = clf.predict(X_test)

## Calculam acuratetea clasificarii.

print("\n<------------------------------------------------>\n")
print("SVM ================> ", accuracy_score(predictions_SVM, y_test)*100)
print("\n<------------------------------------------------>\n")
print("LogisticRegression => ", accuracy_score(predictions_LR, y_test)*100)
print("\n<------------------------------------------------>\n")
print("DecisionTree =======> ", accuracy_score(predictions_CLF, y_test)*100)
print("\n<------------------------------------------------>\n")



