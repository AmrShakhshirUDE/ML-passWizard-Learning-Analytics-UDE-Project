#flask
from flask import Flask, jsonify, request, json,send_file,redirect,url_for,session
from flask_pymongo import PyMongo

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#FeatureSelection using chi2
from sklearn.feature_selection import SelectKBest, chi2
#FeatureSelection using recursive
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
#FeatureSelection using SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#Naive-Bais
from sklearn.naive_bayes import GaussianNB
#LinearSVC "Support Vector Classifier"
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
#K-Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#Decision Tree
from sklearn import tree

#Vis. Comparision
from yellowbrick.classifier import ClassificationReport
#Evaluating
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score

#statistical measures
from sklearn.feature_selection import SelectKBest, chi2

# f=open('numericalValues.csv')
# f.readline()  # skip the header
# data = np.loadtxt(f)

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/LA"
mongo = PyMongo(app)


url="./numValuesComb.csv"
# dataPro=pd.read_csv(url,error_bad_lines=False)
dataPro=pd.read_csv(url, sep=' ')

# # newValues
# aData="./newData.csv"
# dataApp=pd.read_csv(aData, sep=' ')

#Naive-Bais
def naive_bayes(X_train, X_test, y_train, y_test):
    #create an object of the type GaussianNB
    gnb = GaussianNB()
    #train the algorithm on training data and predict using the testing data
    pred = gnb.fit(X_train, y_train.values.ravel()).predict(X_test)
    #print(pred.tolist())
    #print the accuracy score of the model
    print("Naive-Bayes accuracy : ",accuracy_score(y_test, pred, normalize = True))
    # #Precision Score
    # print('precision_support: ',precision_score(y_test, pred, average='micro'))
    # #F1 scores
    # print('precision_recall_fscore_support: ',precision_recall_fscore_support(y_test, pred, average='macro'))
    # print(f1_score(y_test,pred, average=None))

#Support Vector Classification LinearSVC
def SVC(X_train, X_test, y_train, y_test):
    #create an object of type LinearSVC
    svc_model = LinearSVC(random_state=0,dual=False) #instruction to the built-in random number generator to shuffle the data in a specific order
    #train the algorithm on training data and predict using the testing data
    pred = svc_model.fit(X_train, y_train.values.ravel()).predict(X_test)
    #print the accuracy score of the model
    print("LinearSVC accuracy : ",accuracy_score(y_test, pred, normalize = True))
    #Precision Score
    print('precision_support: ',precision_score(y_test, pred, average=None))
    # #F1 scores
    # print(f1_score(y_test,pred, average=None))

#K-Neighbors Classifier
def kNeigh(X_train,X_test,y_train, y_test):
    #create object of the classifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    #Train the algorithm
    neigh.fit(X_train, y_train.values.ravel())
    # predict the response
    pred = neigh.predict(X_test)
    # evaluate accuracy
    print ("KNeighbors accuracy score : ",accuracy_score(y_test, pred))

# Decision Tree
def decTree(X_train,X_test,y_train, y_test):
    #create object of the classifier
    dTree = tree.DecisionTreeClassifier()
    #Train the algorithm
    dTree = dTree.fit(X_train, y_train.values.ravel())
    # predict the response
    pred = dTree.predict(X_test)
    # evaluate accuracy
    print ("Decision Tree accuracy score : ",accuracy_score(y_test, pred))
    #Precision Score
    print('precision_support: ',precision_score(y_test, pred, average=None), dTree.classes_)
    # print(target.head(18))
    # print(target.tail(13))

#encoding attributes
le = preprocessing.LabelEncoder()
dataPro['school']       =le.fit_transform(dataPro['school'])
dataPro['failures']     =le.fit_transform(dataPro['failures'])
dataPro['higher']       =le.fit_transform(dataPro['higher'])
dataPro['Dalc']         =le.fit_transform(dataPro['Dalc'])
dataPro['absences']     =le.fit_transform(dataPro['absences'])
dataPro['alcStudyTime'] =le.fit_transform(dataPro['alcStudyTime'])
dataPro['WalcAbsence']  =le.fit_transform(dataPro['WalcAbsence'])
dataPro['parentsEdu']   =le.fit_transform(dataPro['parentsEdu'])
dataPro['GN1']          =le.fit_transform(dataPro['GN1'])
dataPro['GN2']          =le.fit_transform(dataPro['GN2'])
dataPro['GN3']          =le.fit_transform(dataPro['GN3'])
dataPro['PF1']          =le.fit_transform(dataPro['PF1'])
dataPro['PF2']          =le.fit_transform(dataPro['PF2'])
dataPro['PF3']          =le.fit_transform(dataPro['PF3'])
# dataPro['sex']          =le.fit_transform(dataPro['sex'])
# dataPro['age']          =le.fit_transform(dataPro['age'])
# dataPro['address']      =le.fit_transform(dataPro['address'])
# dataPro['famsize']      =le.fit_transform(dataPro['famsize'])
# dataPro['Pstatus']      =le.fit_transform(dataPro['Pstatus'])
# dataPro['Medu']         =le.fit_transform(dataPro['Medu'])
# dataPro['Fedu']         =le.fit_transform(dataPro['Fedu'])
# dataPro['Mjob']         =le.fit_transform(dataPro['Mjob'])
# dataPro['Fjob']         =le.fit_transform(dataPro['Fjob'])
# dataPro['reason']       =le.fit_transform(dataPro['reason'])
# dataPro['guardian']     =le.fit_transform(dataPro['guardian'])
# dataPro['traveltime']   =le.fit_transform(dataPro['traveltime'])
# dataPro['studytime']    =le.fit_transform(dataPro['studytime'])
# dataPro['schoolsup']    =le.fit_transform(dataPro['schoolsup'])
# dataPro['famsup']       =le.fit_transform(dataPro['famsup'])
# dataPro['paid']         =le.fit_transform(dataPro['paid'])
# dataPro['activities']   =le.fit_transform(dataPro['activities'])
# dataPro['nursery']      =le.fit_transform(dataPro['nursery'])
# dataPro['internet']     =le.fit_transform(dataPro['internet'])
# dataPro['romantic']     =le.fit_transform(dataPro['romantic'])
# dataPro['famrel']       =le.fit_transform(dataPro['famrel'])
# dataPro['freetime']     =le.fit_transform(dataPro['freetime'])
# dataPro['goout']        =le.fit_transform(dataPro['goout'])
# dataPro['Walc']         =le.fit_transform(dataPro['Walc'])
# dataPro['health']       =le.fit_transform(dataPro['health'])
# dataPro['FailHigh']     =le.fit_transform(dataPro['FailHigh'])

# cols=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason',
# 'guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities',
# 'nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences',
# 'studytime','WalcAbsence','parentsEdu','GN1','GN2']


################################## 3 CATEGORIES [FAIL-MODERATE-HIGH] RATIO 80/20 ################################################

colsPF1=['school','absences','higher','alcStudyTime','failures','WalcAbsence']#PF1~ ch2 2 categories 0.8846 [0.8636 0.8888]
# colsG1=['school','absences','higher','alcStudyTime','failures','WalcAbsence']#G1~ ch2 3 categories 0.7538 [0.8260 0.7383 0.0000]

colsPF2=['failures','higher','absences','WalcAbsence','parentsEdu']#PF2~ ch2 2 categories 0.8153 [0.9166 0.8050]
# cols=['school','failures','higher','absences','WalcAbsence','parentsEdu']#G2~ ch2 3 categories 0.7384 [0.8888 0.7142 0.0000]
colsG2=['failures','higher','absences','WalcAbsence','parentsEdu','PF1','GN1']#G2~ ch2 3 categories 0.84615 [0.78125 0.8780 0.8125]

colsPF3=['Dalc','school','absences','higher','parentsEdu','failures','WalcAbsence']#PF3~ ch2 2 categories 0.8307 [0.5555 0.8512]
# cols=['GN1','GN2']#PF3~ 2 categories 0.8846 [0.6666 0.9339]
# cols=['Dalc','school','absences','higher','parentsEdu','failures','WalcAbsence']#G3~ ch2 3 categories 0.6923 [0.6363 0.6974 0.0000]
colsG3=['PF1','PF2','GN1','GN2']#G3~ ch2 3 categories 0.8461 [0.6666 0.8764 0.9411] 

################################################### END OF 3 CATEGORIES #################################################################

dataPF1 = dataPro[colsPF1]
dataPF2 = dataPro[colsPF2]
dataG2  = dataPro[colsG2]
dataPF3 = dataPro[colsPF3]
dataG3  = dataPro[colsG3]

targetPF1   = dataPro['PF1']
targetPF2   = dataPro['PF2']
targetG2    = dataPro['GN2']
targetPF3   = dataPro['PF3']
targetG3    = dataPro['GN3']

##########################################################PF1 model######################################################################
X_train, X_test, y_train, y_test = train_test_split(dataPF1, targetPF1, test_size=0.2, random_state = 10) #'random_state' just ensures that we get reproducible results every time.
#create an object of type LinearSVC
svc_model = LinearSVC(random_state=0,dual=False) #instruction to the built-in random number generator to shuffle the data in a specific order
#train the algorithm on training data and predict using the testing data
modelPF1 = svc_model.fit(X_train, y_train.values.ravel())
predFP1 = modelPF1.predict(X_test)
print ("SVC PF1 accuracy score : ",accuracy_score(y_test, predFP1))
#Precision Score
print('precision_support: ',precision_score(y_test, predFP1, average=None))
##########################################################End PF1 model##################################################################



##########################################################PF2 model######################################################################
X_train, X_test, y_train, y_test = train_test_split(dataPF2, targetPF2, test_size=0.2, random_state = 10) #'random_state' just ensures that we get reproducible results every time.
#create an object of type LinearSVC
svc_model = LinearSVC(random_state=0,dual=False) #instruction to the built-in random number generator to shuffle the data in a specific order
#train the algorithm on training data and predict using the testing data
modelPF2 = svc_model.fit(X_train, y_train.values.ravel())
predFP2 = modelPF2.predict(X_test)
print ("SVC PF2 accuracy score : ",accuracy_score(y_test, predFP2))
#Precision Score
print('precision_support: ',precision_score(y_test, predFP2, average=None))
##########################################################End PF2 model##################################################################


##########################################################G2 model######################################################################
X_train, X_test, y_train, y_test = train_test_split(dataG2, targetG2, test_size=0.2, random_state = 10) #'random_state' just ensures that we get reproducible results every time.
#create an object of type LinearSVC
svc_model = LinearSVC(random_state=0,dual=False) #instruction to the built-in random number generator to shuffle the data in a specific order
#train the algorithm on training data and predict using the testing data
modelG2 = svc_model.fit(X_train, y_train.values.ravel())
predG2 = modelG2.predict(X_test)
print ("SVC G2 accuracy score : ",accuracy_score(y_test, predG2))
#Precision Score
print('precision_support: ',precision_score(y_test, predG2, average=None))
##########################################################End G2 model##################################################################


##########################################################PF3 model######################################################################
X_train, X_test, y_train, y_test = train_test_split(dataPF3, targetPF3, test_size=0.2, random_state = 10) #'random_state' just ensures that we get reproducible results every time.
#create an object of type LinearSVC
svc_model = LinearSVC(random_state=0,dual=False) #instruction to the built-in random number generator to shuffle the data in a specific order
#train the algorithm on training data and predict using the testing data
modelPF3 = svc_model.fit(X_train, y_train.values.ravel())
predFP3 = modelPF3.predict(X_test)
print ("SVC PF3 accuracy score : ",accuracy_score(y_test, predFP3))
#Precision Score
print('precision_support: ',precision_score(y_test, predFP3, average=None))
##########################################################End PF3 model##################################################################

##########################################################G3 model######################################################################
X_train, X_test, y_train, y_test = train_test_split(dataG3, targetG3, test_size=0.2, random_state = 10) #'random_state' just ensures that we get reproducible results every time.
#create an object of type LinearSVC
svc_model = LinearSVC(random_state=0,dual=False) #instruction to the built-in random number generator to shuffle the data in a specific order
#train the algorithm on training data and predict using the testing data
modelG3 = svc_model.fit(X_train, y_train.values.ravel())
predG3 = modelG3.predict(X_test)
print ("SVC G3 accuracy score : ",accuracy_score(y_test, predG3))
#Precision Score
print('precision_support: ',precision_score(y_test, predG3, average=None))
##########################################################End G3 model##################################################################

@app.route('/predic/portuguese',methods=['POST'])
def predict():
    failures    = request.json['failures']
    higher      = request.json['higher']
    Dalc        = request.json['Dalc']
    Walc        = request.json['Walc']
    studytime   = request.json['studytime']
    WalcAbsence = request.json['WalcAbsence']
    school      = request.json['school']
    absences    = request.json['absences']
    parentsEdu  = request.json['parentsEdu']
    PF1         = request.json['PF1']
    PF2         = request.json['PF2']

    sTimeTemp   = (1-(studytime/3)) * 199
    DalcTemp    = Dalc * 200
    alcSudyTime = sTimeTemp + DalcTemp

    WalcTemp    = Walc * 199
    AbsenceTemp = absences * 200
    WalcAbsence = WalcTemp + AbsenceTemp

@app.route('/')  #check connectivity
def connected():
    return'''
    <html>
        <h1>Connected to LAPro-Group Project Backend Side!!!</h1>
        <h4>Team members in alphabetical order</h4>
        <ul>
        <li>Amr Shakhshir</li>
        <li>Baohui Deng</li>
        <li>Hessamoddin Heidarzadeh</li>
        <li>Tannaz Vahidi</li>
        </ul>
    </html
    '''


if __name__ == '__main__':
    app.run(debug =True )