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
dataPro['sex']          =le.fit_transform(dataPro['sex'])
dataPro['age']          =le.fit_transform(dataPro['age'])
dataPro['address']      =le.fit_transform(dataPro['address'])
dataPro['famsize']      =le.fit_transform(dataPro['famsize'])
dataPro['Pstatus']      =le.fit_transform(dataPro['Pstatus'])
dataPro['Medu']         =le.fit_transform(dataPro['Medu'])
dataPro['Fedu']         =le.fit_transform(dataPro['Fedu'])
dataPro['Mjob']         =le.fit_transform(dataPro['Mjob'])
dataPro['Fjob']         =le.fit_transform(dataPro['Fjob'])
dataPro['reason']       =le.fit_transform(dataPro['reason'])
dataPro['guardian']     =le.fit_transform(dataPro['guardian'])
dataPro['traveltime']   =le.fit_transform(dataPro['traveltime'])
dataPro['studytime']    =le.fit_transform(dataPro['studytime'])
dataPro['failures']     =le.fit_transform(dataPro['failures'])
dataPro['schoolsup']    =le.fit_transform(dataPro['schoolsup'])
dataPro['famsup']       =le.fit_transform(dataPro['famsup'])
dataPro['paid']         =le.fit_transform(dataPro['paid'])
dataPro['activities']   =le.fit_transform(dataPro['activities'])
dataPro['nursery']      =le.fit_transform(dataPro['nursery'])
dataPro['higher']       =le.fit_transform(dataPro['higher'])
dataPro['internet']     =le.fit_transform(dataPro['internet'])
dataPro['romantic']     =le.fit_transform(dataPro['romantic'])
dataPro['famrel']       =le.fit_transform(dataPro['famrel'])
dataPro['freetime']     =le.fit_transform(dataPro['freetime'])
dataPro['goout']        =le.fit_transform(dataPro['goout'])
dataPro['Dalc']         =le.fit_transform(dataPro['Dalc'])
dataPro['Walc']         =le.fit_transform(dataPro['Walc'])
dataPro['health']       =le.fit_transform(dataPro['health'])
dataPro['absences']     =le.fit_transform(dataPro['absences'])
dataPro['FailHigh']     =le.fit_transform(dataPro['FailHigh'])
dataPro['alcStudyTime'] =le.fit_transform(dataPro['alcStudyTime'])
dataPro['WalcAbsence']  =le.fit_transform(dataPro['WalcAbsence'])
dataPro['parentsEdu']   =le.fit_transform(dataPro['parentsEdu'])
dataPro['GN1']          =le.fit_transform(dataPro['GN1'])
dataPro['GN2']          =le.fit_transform(dataPro['GN2'])
dataPro['GN3']          =le.fit_transform(dataPro['GN3'])
dataPro['PF1']          =le.fit_transform(dataPro['PF1'])
dataPro['PF2']          =le.fit_transform(dataPro['PF2'])
dataPro['PF3']          =le.fit_transform(dataPro['PF3'])

# cols=['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason',
# 'guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities',
# 'nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences',
# 'studytime','WalcAbsence','parentsEdu','GN1','GN2']


################################## 3 CATEGORIES [FAIL-MODERATE-HIGH] RATIO 80/20 ################################################

# cols=['school','absences','higher','alcStudyTime','failures','WalcAbsence']#PF1~ ch2 2 categories 0.8846 [0.8636 0.8888]
# cols=['school','absences','higher','alcStudyTime','failures','WalcAbsence']#G1~ ch2 3 categories 0.7538 [0.8260 0.7383 0.0000]

# cols=['failures','higher','absences','WalcAbsence','parentsEdu']#PF2~ ch2 2 categories 0.7384 [0.9166 0.8050]
# cols=['school','failures','higher','absences','WalcAbsence','parentsEdu']#G2~ ch2 3 categories 0.7384 [0.8888 0.7142 0.0000]
# cols=['failures','higher','absences','WalcAbsence','parentsEdu','PF1','GN1']#G2~ ch2 3 categories 0.84615 [0.78125 0.8780 0.8125]

# cols=['Dalc','school','absences','higher','parentsEdu','failures','WalcAbsence']#PF3~ ch2 2 categories 0.8307 [0.5555 0.8512]
# cols=['GN1','GN2']#PF3~ 2 categories 0.8846 [0.6666 0.9339]
# cols=['Dalc','school','absences','higher','parentsEdu','failures','WalcAbsence']#G3~ ch2 3 categories 0.6923 [0.6363 0.6974 0.0000]
cols=['PF1','PF2','GN1','GN2']#G3~ ch2 3 categories 0.8461 [0.6666 0.8764 0.9411] 

################################################### END OF 3 CATEGORIES #################################################################

# cols=['school','age','Fedu','studytime','failures','higher','WalcAbsence']



# cols=["school",'failures','higher','absences','WalcAbsence']
# cols=['higher','failures','school','age','absences']# highest for SVC 72.30

# cols3=['PF1','PF2']

# cols=['failures','WalcAbsence','PF1','PF2']

# cols1=["school",'failures','higher','absences','WalcAbsence','parentsEdu','PF1']#LinearSVC accuracy :  0.6923076923076923
# cols1=['WalcAbsence','PF1']#LinearSVC accuracy :  0.6923076923076923

# cols1=["school",'failures','higher','absences','WalcAbsence']#LinearSVC accuracy :  80.77% pass/fail PF1

# cols1=['failures','absences','WalcAbsence','parentsEdu','PF1']#LinearSVC accuracy :   pass/fail PF2

# cols1=['PF1']

# cols=['WalcAbsence']

# cols1=['higher','failures','school','age','absences']# highest for SVC 72.30
# cols2=['failures','absences','WalcAbsence','parentsEdu','GN1'] #GN2 SVC ~ 72.30 / 70.77 for 70:30 ratio
# cols3=['GN1','GN2']# highest for Decision Tree ~ 83.08/ 82.56 KNeighbors 70:30
# cols3=['failures','WalcAbsence','PF1','PF2','PF3']# highest for Decision Tree ~ 69.62

# data = dataPro[cols]
data = dataPro[cols]
# data = dataPro[cols2]
# data = dataPro[cols3]
#assigning the Oppurtunity Result column as target
# tarCol = [' GC1',' GC2',' GC3']
tarCol1= ['GN3']
# tarCol2= ['PF2']
# # tarCol3= ['PF3']
# # tarCol1= ['GN1']
# # tarCol2= ['GN2']
# # tarCol3= ['GN3']
target = dataPro[tarCol1]
# target = dataPro[tarCol2]
# # target = dataPro[tarCol3]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state = 10) #'random_state' just ensures that we get reproducible results every time.

# # # chi2 method of selecting features
# sel = SelectKBest(chi2,4)
# sel.fit(X_train,y_train)
# print(sel.get_support())
# print(sel.scores_)

# # #get accuracy for each attributes individually
# for x in cols:
#     print('Attribyte name:',x)
#     col=[x]
#     data = dataPro[col]
#     X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state = 10) #'random_state' just ensures that we get reproducible results every time.
#     target = dataPro['GN1']
#     # SVC(X_train, X_test, y_train, y_test)
#     decTree(X_train, X_test, y_train, y_test)
# # #     kNeigh(X_train, X_test, y_train, y_test)

naive_bayes(X_train,X_test,y_train, y_test)
SVC(X_train,X_test,y_train, y_test)
kNeigh(X_train,X_test,y_train, y_test)
decTree(X_train,X_test,y_train, y_test)

colsPF1=["school",'failures','higher','absences','WalcAbsence','parentsEdu']#LinearSVC accuracy :  80.77% pass/fail PF1
data = dataPro[colsPF1]
target = dataPro['PF1']
# separatione data into train & test
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state = 10) #'random_state' just ensures that we get reproducible results every time.
#create an object of type LinearSVC
svc_model = LinearSVC(random_state=0,dual=False) #instruction to the built-in random number generator to shuffle the data in a specific order
#train the algorithm on training data and predict using the testing data
modelPF1 = svc_model.fit(X_train, y_train.values.ravel())
predFP1 = modelPF1.predict(X_test)
print ("SVC PF1 accuracy score : ",accuracy_score(y_test, predFP1))
#Precision Score
print('precision_support: ',precision_score(y_test, predFP1, average=None))

colsPF2 = ['WalcAbsence','PF1']
data = dataPro[colsPF2]
target = dataPro['PF2']
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state = 10)
svc_model = LinearSVC(random_state=0,dual=False) #instruction to the built-in random number generator to shuffle the data in a specific order
#train the algorithm on training data and predict using the testing data
modelPF2 = svc_model.fit(X_train, y_train.values.ravel())
predFP2 = modelPF2.predict(X_test)
print ("SVC PF2 accuracy score : ",accuracy_score(y_test, predFP2))
#Precision Score
print('precision_support: ',precision_score(y_test, predFP2, average=None))

colsPF3 = ['PF1','PF2']
data = dataPro[colsPF3]
target = dataPro['PF3']
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state = 10)
modelPF3 = svc_model.fit(X_train, y_train.values.ravel())
predFP3 = modelPF3.predict(X_test)
print ("SVC PF3 accuracy score : ",accuracy_score(y_test, predFP3))
#Precision Score
print('precision_support: ',precision_score(y_test, predFP3, average=None))

# colsGN3 = ['failures','WalcAbsence','PF1','PF2','PF3']
# colsGN3 = ['failures','WalcAbsence','school','higher','absences','parentsEdu','PF1','PF2']
# colsGN3 = ['failures','WalcAbsence','school','higher','absences','parentsEdu']
colsGN3 = ['PF1','PF2','GN1','GN2']

data = dataPro[colsGN3]
target = dataPro['GN3']
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state = 10)
dTree = tree.DecisionTreeClassifier()
modelGN3 = dTree.fit(X_train, y_train.values.ravel())
# modelGN3 = svc_model.fit(X_train, y_train.values.ravel())
GN3Pred = modelGN3.predict(X_test)
# GN3Pred = modelGN3.predict([[0,1000,0,1,0,11]])
print ("Decision Tree accuracy score : ",accuracy_score(y_test, GN3Pred))
# print ("Decision Tree raw accuracy score : ",GN3Pred)
#Precision Score
print('precision_support: ',precision_score(y_test, GN3Pred, average=None), modelGN3.classes_)
print(predFP3.shape)
data = np.hstack((dataPro[colsGN3][389:649],predFP3.reshape(260,1)))
target = dataPro['GN3'][389:649]
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state = 10)
# dTree = tree.DecisionTreeClassifier()
# modelGN3 = svc_model.fit(X_train, y_train.values.ravel())
modelGN3 = dTree.fit(X_train, y_train.values.ravel())
modelGN3 = modelGN3.fit(X_train, y_train.values.ravel())
GN3Pred = modelGN3.predict(X_test)
# evaluate accuracy
print ("Decision Tree accuracy score : ",accuracy_score(y_test, GN3Pred))
#Precision Score
print('precision_support: ',precision_score(y_test, GN3Pred, average=None), modelGN3.classes_)

# naive_bayes(X_train,X_test,y_train, y_test)
# SVC(X_train,X_test,y_train, y_test)
# kNeigh(X_train,X_test,y_train, y_test)
# decTree(X_train,X_test,y_train, y_test)

# #encoding New attributes
# le = preprocessing.LabelEncoder()
# dataApp['school']=le.fit_transform(dataApp['school'])
# dataApp['sex']=le.fit_transform(dataApp['sex'])
# dataApp['age']=le.fit_transform(dataApp['age'])
# dataApp['address']=le.fit_transform(dataApp['address'])
# dataApp['famsize']=le.fit_transform(dataApp['famsize'])
# dataApp['Pstatus']=le.fit_transform(dataApp['Pstatus'])
# dataApp['Medu']=le.fit_transform(dataApp['Medu'])
# dataApp['Fedu']=le.fit_transform(dataApp['Fedu'])
# dataApp['Mjob']=le.fit_transform(dataApp['Mjob'])
# dataApp['Fjob']=le.fit_transform(dataApp['Fjob'])
# dataApp['reason']=le.fit_transform(dataApp['reason'])
# dataApp['guardian']=le.fit_transform(dataApp['guardian'])
# dataApp['traveltime']=le.fit_transform(dataApp['traveltime'])
# dataApp['studytime']=le.fit_transform(dataApp['studytime'])
# dataApp['failures']=le.fit_transform(dataApp['failures'])
# dataApp['schoolsup']=le.fit_transform(dataApp['schoolsup'])
# dataApp['famsup']=le.fit_transform(dataApp['famsup'])
# dataApp['paid']=le.fit_transform(dataApp['paid'])
# dataApp['activities']=le.fit_transform(dataApp['activities'])
# dataApp['nursery']=le.fit_transform(dataApp['nursery'])
# dataApp['higher']=le.fit_transform(dataApp['higher'])
# dataApp['internet']=le.fit_transform(dataApp['internet'])
# dataApp['romantic']=le.fit_transform(dataApp['romantic'])
# dataApp['famrel']=le.fit_transform(dataApp['famrel'])
# dataApp['freetime']=le.fit_transform(dataApp['freetime'])
# dataApp['goout']=le.fit_transform(dataApp['goout'])
# dataApp['Dalc']=le.fit_transform(dataApp['Dalc'])
# dataApp['Walc']=le.fit_transform(dataApp['Walc'])
# dataApp['health']=le.fit_transform(dataApp['health'])
# dataApp['absences']=le.fit_transform(dataApp['absences'])
# dataApp['FailHigh']=le.fit_transform(dataApp['FailHigh'])
# dataApp['WalcAbsence']=le.fit_transform(dataApp['WalcAbsence'])
# dataApp['parentsEdu']=le.fit_transform(dataApp['parentsEdu'])
# dataApp['GN3']=le.fit_transform(dataApp['GN3'])
# print(predFP1.shape, predFP2.shape, predFP3.shape)

# colsGN3 = ['failures','WalcAbsence']
# data1 = dataApp[colsGN3]
# data1 = np.hstack((data1[71:260],predFP1.reshape(189, 1)))
# data1 = np.hstack((data1,predFP2[71:260].reshape(189, 1)))
# data1 = np.hstack((data1,predFP3[71:260].reshape(189, 1)))
# # data = np.insert(data, 0, predFP3)
# # data = np.insert(data, 0, predFP2)
# # data = np.insert(data, 0, predFP1)
# target1 = dataApp['GN3']
# # X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state = 10)
# GN3Pred = modelGN3.predict(data1)
# print(data1.shape,predFP1.shape, predFP2.shape, predFP3.shape,GN3Pred.shape,target1.shape)
# # print(GN3Pred.head(15))
# # evaluate accuracy
# print ("Decision Tree accuracy score : ",accuracy_score(target1[71:260], GN3Pred))
# #Precision Score
# print('precision_support: ',precision_score(target1[71:260], GN3Pred, average=None), modelGN3.classes_)

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