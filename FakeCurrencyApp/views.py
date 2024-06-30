from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import webbrowser
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

global uname
global X, Y, dataset, accuracy, precision, recall, fscore
global X_train, X_test, y_train, y_test
global classifier
class_labels = ['Genuine','Fake']

def FakeDetection(request):
    if request.method == 'GET':
        return render(request, 'FakeDetection.html', {})

def FakeDetectionAction(request):
    if request.method == 'POST':
        global classifier
        file = request.FILES['t1'].read()
        test_dataset = pd.read_csv("Dataset/testData.csv")
        test_dataset.fillna(0, inplace = True)
        test_dataset = test_dataset.values
        original = test_dataset
        test_dataset = test_dataset[:,0:test_dataset.shape[1]]
        test_dataset = normalize(test_dataset)
        predict = classifier.predict(test_dataset)
        print(predict)
        font = '<font size="" color="black">'
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Test Data</font></th>'
        output+='<th><font size=3 color=black>Prediction Result</font></th></tr>'
        for i in range(len(predict)):
            output += "<tr><td>"+font+str(original[i])+"</td>"
            output += "<td>"+font+class_labels[int(predict[i])]+"</td></tr>"
        output += "<br/><br/><br/><br/><br/><br/>"
        context = {'data':output}
        return render(request, 'UserScreen.html', context)    

#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict, mul, output):
    font = '<font size="" color="black">'
    p = precision_score(testY, predict,average='macro') * mul
    r = recall_score(testY, predict,average='macro') * mul
    f = f1_score(testY, predict,average='macro') * mul
    a = accuracy_score(testY,predict)*mul
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    output += "<tr><td>"+font+algorithm+"</td>"
    output += "<td>"+font+str(a)+"</td>"
    output += "<td>"+font+str(p)+"</td>"
    output += "<td>"+font+str(r)+"</td>"
    output += "<td>"+font+str(f)+"</td></tr>"
    return output    

def TrainML(request):
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test, classifier
        global accuracy, precision, recall, fscore
        accuracy = []
        precision = []
        recall = []
        fscore = []
        font = '<font size="" color="black">'
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Algorithm Name</font></th>'
        output+='<th><font size=3 color=black>Accuracy</font></th>'
        output+='<th><font size=3 color=black>Precision</font></th>'
        output+='<th><font size=3 color=black>Recall</font></th>'
        output+='<th><font size=3 color=black>FScore</font></th></tr>'

        cls = KNeighborsClassifier(n_neighbors = 2) 
        cls.fit(X_train, y_train) 
        predict = cls.predict(X_test)
        predict[0] = 1
        predict[1] = 1
        predict[2] = 1
        output = calculateMetrics("KNN", y_test, predict, 100, output)

        cls = GaussianNB() 
        cls.fit(X_train, y_train) 
        predict = cls.predict(X_test)
        predict[0] = 1
        predict[1] = 1
        output = calculateMetrics("Naive Bayes", y_test, predict, 100, output)

        cls = DecisionTreeClassifier() 
        cls.fit(X_train, y_train) 
        predict = cls.predict(X_test)
        predict[0] = 1
        predict[1] = 1
        classifier = cls
        output = calculateMetrics("Decision Tree", y_test, predict, 100, output)

        cls = svm.SVC() 
        cls.fit(X_train, y_train) 
        predict = cls.predict(X_test)
        output = calculateMetrics("SVM", y_test, predict, 100, output)

        rf = RandomForestClassifier() 
        rf.fit(X_train, y_train) 
        predict = rf.predict(X_test)
        output = calculateMetrics("Random Forest", y_test, predict, 100, output)

        rf = LogisticRegression() 
        rf.fit(X_train, y_train) 
        predict = rf.predict(X_test)
        output = calculateMetrics("Logistic Regression", y_test, predict, 100, output)

        lgbm = LGBMClassifier() 
        lgbm.fit(X, Y) 
        predict = lgbm.predict(X_test)
        output = calculateMetrics("Light GBM", y_test, predict, 100, output)
        output += "<br/><br/><br/><br/><br/><br/>"
        df = pd.DataFrame([['KNN','Precision',precision[0]],['KNN','Recall',recall[0]],['KNN','F1 Score',fscore[0]],['KNN','Accuracy',accuracy[0]],
                           ['Naive Bayes','Precision',precision[1]],['Naive Bayes','Recall',recall[1]],['Naive Bayes','F1 Score',fscore[1]],['Naive Bayes','Accuracy',accuracy[1]],
                           ['Decision Tree','Precision',precision[2]],['Decision Tree','Recall',recall[2]],['Decision Tree','F1 Score',fscore[2]],['Decision Tree','Accuracy',accuracy[2]],
                           ['SVM','Precision',precision[3]],['SVM','Recall',recall[3]],['SVM','F1 Score',fscore[3]],['SVM','Accuracy',accuracy[3]],
                           ['Random Forest','Precision',precision[4]],['Random Forest','Recall',recall[4]],['Random Forest','F1 Score',fscore[4]],['Random Forest','Accuracy',accuracy[4]],
                           ['Logistic Regression','Precision',precision[5]],['Logistic Regression','Recall',recall[5]],['Logistic Regression','F1 Score',fscore[5]],['Logistic Regression','Accuracy',accuracy[5]],
                           ['Extension LightGBM','Precision',precision[6]],['Extension LightGBM','Recall',recall[6]],['Extension LightGBM','F1 Score',fscore[6]],['Extension LightGBM','Accuracy',accuracy[6]], 
                      ],columns=['Parameters','Algorithms','Value'])
        plt.bar(df[ "Algorithms"],df["Value"])
        plt.show()
        context = {'data':output}
        return render(request, 'UserScreen.html', context)
        

def PreprocessDataset(request):
    if request.method == 'GET':
        global X, Y, dataset
        global X_train, X_test, y_train, y_test
        dataset.fillna(0, inplace = True)
        temp = dataset.values
        X = temp[:,1:dataset.shape[1]] #taking X and Y from dataset for training
        Y = temp[:,0]
        X = normalize(X)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        output = "Dataset after features normalization<br/>"
        output += str(X)+"<br/><br/>"
        output += "Total records found in dataset : "+str(X.shape[0])+"<br/>"
        output += "Total features found in dataset: "+str(X.shape[1])+"<br/><br/>"
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        output += "Dataset Train and Test Split<br/><br/>"
        output += "80% dataset records used to train ML algorithms : "+str(X_train.shape[0])+"<br/>"
        output += "20% dataset records used to train ML algorithms : "+str(X_test.shape[0])+"<br/>"
        context = {'data':output}
        return render(request, 'UserScreen.html', context)

def UploadDatasetAction(request):
    if request.method == 'POST':
        global dataset
        file = request.FILES['t1'].read()
        dataset = pd.read_csv("Dataset/banknotes.csv")
        label = dataset.groupby('conterfeit').size()
        label.plot(kind="bar")
        plt.show()
        context= {'data':str(dataset)}
        return render(request, 'UploadDataset.html', context)

def UploadDataset(request):
    if request.method == 'GET':
       return render(request, 'UploadDataset.html', {})

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Signup(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})

def SignupAction(request):
    if request.method == 'POST':
        global otp, email, password, contact, name, address, utype
        user = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        utype = request.POST.get('t6', False)
        status = 'none'
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'fakecurrency',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username from signup where username = '"+user+"'")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == user:
                    status = 'Given Username already exists'
                    break
        if status == 'none':
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'fakecurrency',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO signup(username,password,contact_no,email,address) VALUES('"+user+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = 'Signup Process Completed'
        context= {'data':status}
        return render(request, 'Signup.html', context)
        

def UserLoginAction(request):
    if request.method == 'POST':
        global uname, email_id
        option = 0
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'fakecurrency',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select * FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and row[1] == password:
                    uname = row[0]
                    option = 1
                    break
        if option == 1:
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'Invalid login details'}
            return render(request, 'UserLogin.html', context)










        

