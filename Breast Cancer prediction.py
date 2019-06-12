# -*- coding: utf-8 -*-
"""
Created on Thu May 31 09:58:46 2018

@author: Saloni Singh
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#This is for collecting and filtering requisite datasets from already existing one

dataset=pd.read_csv('dataset_cancer.csv')
#X=dataset.iloc[:, 2:32].values
#Y=dataset.iloc[:, 1].values


dataset.info()
dataset.drop('Unnamed: 32', axis=1, inplace=True)
dataset.drop("id", axis=1, inplace=True)

features_mean=list(dataset.columns[1:11])
features_se=list(dataset.columns[11:20])
features_worst=list(dataset.columns[20:31])
#print(features_mean)
#print("----------------------------------------")
#print(features_se)
#print("----------------------------------------")
#print(features_worst)



dataset['diagnosis']=dataset['diagnosis'].map({'M':1, 'B':0})
X=dataset.iloc[:, 1:11].values
Y=dataset.iloc[:, 0].values

#this is used to make the plot more interactive




import seaborn as sns
corr=dataset[features_mean].corr()
plt.figure(figsize=(14, 14))
sns.heatmap(corr, cbar=True, square=True, annot=True, fmt='.2f', annot_kws={'size':15}, 
  xticklabels=features_mean, yticklabels=features_mean, cmap='coolwarm')

#dataset.describe()
#from sklearn.preprocessing import LabelEncoder
#label_encoder=LabelEncoder()
#Y=label_encoder.fit_transform(Y) 
prediction_var=['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'area_mean']
#X=dataset.iloc[:, 2:7].values



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.4)


#feature sclaing
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#RandomForestClassification
from sklearn.ensemble import RandomForestClassifier
reg=RandomForestClassifier(n_estimators=100)
reg.fit(X_train, Y_train)

y_pred=reg.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test, y_pred)

from sklearn.metrics import f1_score
f1_score(Y_test, y_pred)



"""
plt.scatter(X_test, Y_test.astype(float), color='orange')
plt.plot(X_test, y_pred.astype(float), color='blue')
plt.show()
"""
#splitting of dataset
#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=0)

#feature Scaling
#from sklearn.preprocessing import StandardScaler
#obj=StandardScaler()
#X_train=obj.fit_transform(X_train)
#X_test=obj.transform(X_test)

#perform and check how well does linear regression does on this data

#y_pred=regressor.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test, y_pred)
plt.matshow(cm)
plt.colorbar()

precision=cm[1, 1]/(cm[0, 1]+cm[1, 1])
recall=cm[1, 1]/(cm[1, 0]+cm[1, 1])
print(precision)
print(recall)
#precision=cm[0, 0]/(cm[0, 0]+cm[1, 1])
#recall=cm[0, 0]/()
#sns.heatmap(cm, annot=True, annot_kws={"size": 16}) 

#Applying Support Vector Machine
"""
from sklearn.svm import SVC
model=SVC()
model.fit(X_train, Y_train)
y_pred=model.predict(X_test)

accuracy_score(Y_test, y_pred)

"""

#Appyling PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance=pca.explained_variance_ratio_


#Logistic Regression
from sklearn.linear_model import LogisticRegression
regressor=LogisticRegression(random_state=0)
regressor.fit(X_train, Y_train)


y_pred_2=regressor.predict(X_test)

cm_2=confusion_matrix(Y_test, y_pred_2)
plt.matshow(cm_2)
plt.colorbar()

precision_2=cm_2[1, 1]/(cm_2[0, 1]+cm_2[1, 1])
recall_2=cm_2[1, 1]/(cm_2[1, 0]+cm_2[1, 1])

"""
plt.scatter(X_train[:,0], X_train[:,1], s=40, c=Y_train, cmap=plt.cm.Spectral)
plot_decision_boundary(lambda x: regressor.predict(x))
plt.title('Logistic Regression')
"""

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, regressor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Regressor (Training set)')
#plt.xlabel('')
#plt.ylabel('')
plt.legend()
plt.show()


"""eleminating the features
this pice of code is used in some kind of linear_models"""
"""
from sklearn.feature_selection import RFE
rfe=RFE(reg, 3)
rfe=rfe.fit(X_train, Y_train)

print(rfe.support_)
print(rfe.ranking_)
"""


"""Further, this piece of code is used for eliminating features in ensemble models"""
"""print(reg.feature_importances_) 
prediction_var=['perimeter_mean', 'compactness_mean', 'texture mean']
X=dataset[prediction_var]
#from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2)
reg.fit(X_train, Y_train)
y_pred=reg.predict(X_test)
"""
#accuracy_score(Y_test, y_pred)

"""
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((569, 1)).astype(int), values=X, axis=1)
X_opt=X[:, [0, 1, 3, 5]]
regressor_OLS=sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()
"""

"""
X_opt=X[:, [0, 1, 2, 3, 5]]
regressor_OLS=sm.OLS(endog=Y, exog=X_opt).fit()
regressor_OLS.summary()
"""

#SupportVectorMachine
from sklearn.svm import SVC
sv=SVC(kernel='rbf')
sv.fit(X_train, Y_train)

y_pred_3=sv.predict(X_test)


#Confusion matrix for this model
cm_3=confusion_matrix(Y_test, y_pred_3)
plt.matshow(cm_3)
plt.colorbar()

precision_3=cm_3[1, 1]/(cm_3[0, 1]+cm_3[1, 1])
recall_3=cm_3[1, 1]/(cm_3[1, 0]+cm_3[1, 1])

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, sv.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Regressor (Training set)')
#plt.xlabel('')
#plt.ylabel('')
plt.legend()
plt.show()



#NeuralNetworking
def sigmoid(X):
    return 1/(1+np.exp(-X))

#Computing the cost function
m=len(X_train[:, 0])
h=sigmoid(X_train)
J=(-1/m)*sum(np.dot(np.matrix.transpose(np.log(h)), Y_train)+np.dot(np.matrix.transpose(np.log(1-h)), (1-Y_train)))

#Training the neural network with one hidden layer having 5 activation nodes (excluding the bias node)

alpha=[0.01, 0.1, 1, 10, 100]
X_train=np.append(np.ones([m, 1], dtype=int), X_train, 1)
Alp=np.ones([5, 60], dtype=float)
k=-1;
for a in alpha:
    k+=1
    u=0
    np.random.seed(2)
    Theta1=np.random.randn(5, 3)
    Theta2=np.random.randn(1, 6)
    print("For "+str(a)+" , being the value of chosen alpha(learning rate), computed error seems to be : ")
    for j in range(6000):
        a1=X_train
        Z1=np.dot(Theta1, np.matrix.transpose(X_train))
        a2=sigmoid(Z1)
        a2=np.append(np.ones([1, m], dtype=int), a2, 0)
        Z2=np.dot(Theta2, a2)
        a3=sigmoid(Z2)
        a3=np.matrix.transpose(a3)    
        error_3=np.subtract(a3, Y_train.reshape(m, 1))
        error_2=np.multiply(np.dot(error_3, Theta2), np.matrix.transpose(a2))
        error_2=np.multiply(error_2, np.matrix.transpose((1-a2)))
        del_1=np.dot(np.matrix.transpose(error_2), a1)
        del_2=np.dot(np.matrix.transpose(error_3), np.matrix.transpose(a2))
        del_1=del_1[1:6, :]
        Theta1=Theta1-a*del_1
        Theta2=Theta2-a*del_2
        if (j%100)==0:
             print ("Error after "+str(j)+" iterations:" + str(np.mean(np.abs(error_3))))
             Alp[k, u]=np.mean(np.abs(error_3))
             u+=1
#fig.canvas.draw()    
    
         
o=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900]     
#lets plot the graph for each alpha to study (computed error Vs no. of iterations)         
for h in range(5):
    print("For the value of alpha "+str(alpha[h])+" the graph so plotted is :")
    plt.scatter(o, Alp[h, :], color='orange') 
    #plt.yticks(np.arange(min(Alp[h, :]), max(Alp[h, :])+1, ))     
    plt.ylim(0.04, 0.2)
    plt.show()     
         
             
#By looking at the plot, we chose the value of alpha to be 0.1
#Now, drawing the error for test set
m1=len(X_test[:, 0])
X_test=np.append(np.ones([m1, 1], dtype=int), X_test, 1)    
np.random.seed(2)
Theta1=np.random.randn(5, 3)
Theta2=np.random.randn(1, 6)
final_alpha=0.1
e=0
Alp1=np.ones([10, m1], dtype=float)          
for j in range(10000):
    a1=X_test
    Z1=np.dot(Theta1, np.matrix.transpose(X_test))
    a2=sigmoid(Z1)
    a2=np.append(np.ones([1, m1], dtype=int), a2, 0)
    Z2=np.dot(Theta2, a2)
    a3=sigmoid(Z2)
    a3=np.matrix.transpose(a3)    
    error_3=np.subtract(a3, Y_test.reshape(m1, 1))
    error_2=np.multiply(np.dot(error_3, Theta2), np.matrix.transpose(a2))
    error_2=np.multiply(error_2, np.matrix.transpose((1-a2)))
    del_1=np.dot(np.matrix.transpose(error_2), a1)
    del_2=np.dot(np.matrix.transpose(error_3), np.matrix.transpose(a2))
    del_1=del_1[1:6, :]
    Theta1=Theta1-final_alpha*del_1
    Theta2=Theta2-final_alpha*del_2
    if (j%1000)==0:
        print ("Error after "+str(j)+" iterations:" + str(np.mean(np.abs(error_3))))
        Alp1[e, :]=np.matrix.transpose(a3)
        e+=1
        #Alp1[k, u]=np.mean(np.abs(error_3))
        #u+=1         
         
        
#Now plotting
from matplotlib.colors import ListedColormap
for r in range(10):
    r_y=Alp1[r, :]        
    plt.xlabel('Feature_1')
    plt.ylabel('Feature_2')
    plt.scatter(X_test[:, 1], X_test[:, 2], marker='o', c=Y_test, s=75, cmap="RdBu", alpha=0.9, edgecolor='white')
    plt.colorbar()
    plt.scatter(X_test[:, 1], X_test[:, 2], marker='+', c=r_y, s=100, cmap="Set1", alpha=0.5, edgecolor='white')
    plt.title('Comparison between the actual Y_test and predicted one after some iterations')
    plt.colorbar()
    plt.show()
#plt.contourf(X1, X2, sv.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#            alpha = 0.75, cmap = ListedColormap(('red', 'green')))
   
#plt.xlabel('')
#plt.ylabel('')
    
    
#Here the accuracy is
r_y_i=np.around(r_y)
r_y_i=r_y_i.astype(int)
cm_Neural=confusion_matrix(Y_test, r_y_i)
plt.matshow(cm_Neural)
plt.colorbar()    
#accuracy_rate=(206/228)*100    
precision_Neural=cm_Neural[1, 1]/(cm_Neural[0, 1]+cm_Neural[1, 1])
recall_Neural=cm_Neural[1, 1]/(cm_Neural[1, 0]+cm_Neural[1, 1])
