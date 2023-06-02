# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Santhosh U
RegisterNumber:  212222240092
*/

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
### 1. Result output
![Output1](https://github.com/SanthoshUthiraKumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477975/ad56ee08-df9a-4e50-a089-97fb2747afe8)

### 2. data.head()
![Output2](https://github.com/SanthoshUthiraKumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477975/2ae7ea90-b009-4208-80a7-443a885c520f)

### 3. data.info()
![Output3](https://github.com/SanthoshUthiraKumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477975/47f320d3-a20f-4f85-a167-2335fd937f61)

### 4. data.isnull().sum()
![Output4](https://github.com/SanthoshUthiraKumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477975/2ee7a51c-4a40-48ad-88fa-14e831975eb5)

### 5. Y_prediction value
![Output5](https://github.com/SanthoshUthiraKumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477975/c03cca1c-46d5-43b8-b3c3-a3d45ca67237)

### 6. Accuracy value
![Output6](https://github.com/SanthoshUthiraKumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477975/2c68fcb0-0ed3-4d2c-8e49-97932204b1a1)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
