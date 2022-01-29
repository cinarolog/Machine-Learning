# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 18:23:14 2022

@author: cinar
"""

#%% Import data and libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=Warning)

#%% information

df=pd.read_csv("Automobile.csv")

df.head(5)
df.tail(5)
df.shape
df.info()
df.describe().T
df.isnull().sum()
len(df)

len(df.describe().columns)# numerical columns
df.describe().columns 


#%% Data Preprocessing

for col in df.columns:
    print(col,df[col].unique())


#categorical
for col in df.columns:
    values = []
    
    # numerik olmayanlar -> kategorik
    if col not in df.describe().columns:
        for val in df[col].unique():
            values.append(val)
        
        print("{0} -> {1}".format(col, values))
        

df.CarName

manufacturer = df['CarName'].apply(lambda x: x.split(' '))
manufacturer


manufacturer = df['CarName'].apply(lambda x: x.split(' ')[0])
manufacturer

data=df.copy()
data.drop(columns=["CarName"],axis=1,inplace=True)
data.insert(3,"manufacturer",manufacturer)
data.groupby(by="manufacturer").count()

# üretiici isimlerini düzeltelim

data.manufacturer.unique()
data.manufacturer=data.manufacturer.str.lower()
data.manufacturer.unique()

data.replace({
    
    "maxda" : "mazda",
    "porcsche" : "porsche",
    "toyouta" : "toyota",
    "vokswagen" : "vw",
    "volkswagen" : "vw"
    
    
    }, inplace=True)


data.manufacturer.unique()
data.head(10)

# Univariate (tekli) Analysis

sns.countplot(data.symboling)
plt.show()


fig = plt.figure(figsize=(20,12))

plt.subplot(2,3,1)
plt.title('Fueltype')
sns.countplot(data.fueltype)
# benzinli (gas) arabalar çoğunlukta


plt.subplot(2,3,2)
plt.title('Fuelsystem')
sns.countplot(data.fuelsystem)
# mpfi (multi point fuel injection) en çok tercih edilen, yeni teknoloji


plt.subplot(2,3,3)
plt.title('Aspiration')
sns.countplot(data.aspiration)
# çoğunluk standart beslemeli


plt.subplot(2,3,4)
plt.title('Door Number')
sns.countplot(data.doornumber)
# çoğunluk 4 kapılı


plt.subplot(2,3,5)
plt.title('Car Body')
sns.countplot(data.carbody)
# çoğunluk sedan


plt.subplot(2,3,6)
plt.title('Drive Wheel')
sns.countplot(data.drivewheel)
# çekiş sistemi, standart çeker çoğunlukta

plt.show()


# Bivarite 2'li Analysis


plt.figure(figsize=(16,8))
plt.title('Üretici Fiyatları', fontsize=16)
sns.barplot(x=data.manufacturer, y=data.price, 
            hue=data.fueltype, palette='Set2')
plt.xticks(rotation=90)
plt.tight_layout()


#symboling:

plt.figure(figsize=(8,6))
sns.boxplot(x=data.symboling, y=data.price)
plt.show()

#fueltype:

plt.figure(figsize=(8,6))
sns.boxplot(x=data.fueltype, y=data.price)
plt.show()

#enginelocation:

plt.title('Engine Location')
sns.countplot(data.enginelocation)
sns.boxplot(x=data.enginelocation, y=data.price)
plt.show()

# çoğunlukla motoru önde olan arabalar var ve fiyatları daha düşük
# motor arkada ise fiyat çok yüksek oluyor​
#cylindernumber:

plt.title('Cylinder Number')
sns.countplot(data.cylindernumber)
sns.boxplot(x=data.cylindernumber, y=data.price)
plt.show()

# silindir sayısı arttıkça fiyatın görece arttığını söyleyebiliriz
# bakalım ne kadar dörğru çıkacak
# Fiyatın Kendi İçinde Dağılımı:
# Şimdi sadece fiyatın nasıl kümelendiğine bakalım:
    
sns.distplot(data.price)
plt.show()

# fiyatın genelde 5.000 ile 20.000 USD arasında dağıldığı görülüyor

plt.title('Fiyat Dağılımı')
sns.boxplot(y=data.price)
plt.show()

data.price.describe()

data.columns

cols = ['wheelbase', 'carlength', 'carwidth', 'carheight', 
        'curbweight', 'enginesize', 'boreratio', 'stroke',
        'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']
len(cols)

# regresyon doğruları ile ilişkiyi görelim
plt.figure(figsize=(20,25))

for i in range(len(cols)):
    plt.subplot(5,3,i+1)
    plt.title(cols[i] + ' - Fiyat')
    sns.regplot(x=eval('data' + '.' + cols[i]), y=data.price)
    
plt.tight_layout()


# Burada nerdeyse tüm değişkenleri önemli. Yani fiyat üzerinde etkisi olabilir.
# Etkisiz olanlar:
# carheight
# stroke
# compression ratio
# peak rpm
# highway mpg
# city mpg
# Bunları çıkarabiliriz.

data_new = data[[
        'car_ID', 'symboling', 'fueltype', 'manufacturer', 'aspiration',
        'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'wheelbase',
        'carlength', 'carwidth', 'curbweight', 'enginetype', 'cylindernumber',
        'enginesize', 'fuelsystem', 'boreratio', 'horsepower', 'price']]


data_new.head()


  
sns.pairplot(data_new)
plt.show()




#%% Feature Engineering

torque=(data.horsepower*5252 )/ (data.peakrpm)
torque

data.insert(10,"torque")


data.insert(10, 'torque', 
            pd.Series(data.horsepower * 5252 / data.peakrpm, 
            index=data.index))
data.torque


#torque and sales graphıcs
plt.title('Torque - Sales', fontsize=16)
plt.xlabel("Torque")
plt.ylabel("Sales")
sns.regplot(x=data.torque, y=data.price)
plt.legend(loc="upper left")
plt.show()


plt.title("Torque",fontsize=18,color="red")
sns.distplot(data.torque,color="green")
plt.legend(loc="upper right")
plt.show()


# yakıt ekonomısı

data["fueleconomy"]=(data.citympg * 0.55) + (data.highwaympg* 0.45)
data.fueleconomy

plt.title("Fuel economy-Sales", fontsize=18,color="blue")
sns.distplot(data.fueleconomy,color="red")
plt.show()


# etkısı az olan feature ları sil


data.drop(columns=['car_ID','manufacturer','doornumber','carheight',
                   'compressionratio', 'symboling','stroke','citympg',
                   'highwaympg', 'fuelsystem', 'peakrpm'], 
          axis=1, inplace=True)
data.head(5)
data.hist()

# sns.pairplot(data)
# plt.show()

cars=data.copy()

cars.info()

"""

0   fueltype        205 non-null    object 
 1   aspiration      205 non-null    object 
 2   carbody         205 non-null    object 
 3   drivewheel      205 non-null    object 
 4   enginelocation  205 non-null    object 
 5   wheelbase       205 non-null    float64
 6   torque          205 non-null    float64
 7   carlength       205 non-null    float64
 8   carwidth        205 non-null    float64
 9   curbweight      205 non-null    int64  
 10  enginetype      205 non-null    object 
 11  cylindernumber  205 non-null    object 
 12  enginesize      205 non-null    int64  
 13  boreratio       205 non-null    float64
 14  horsepower      205 non-null    int64  
 15  price           205 non-null    float64
 16  fueleconomy     205 non-null    float64
dtypes: float64(7), int64(3), object(7)

"""

# one hot encoding


dummies_list = ['fueltype', 'aspiration', 'carbody','drivewheel',
                'enginelocation', 'enginetype', 'cylindernumber']

for i in dummies_list:
    
    temp_df = pd.get_dummies(eval('cars' + '.' + i), drop_first=True)
    
    cars = pd.concat([cars, temp_df], axis=1)
    
    cars.drop([i], axis=1, inplace=True)
    
cars.head()

cars.info()

"""

#   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   wheelbase    205 non-null    float64
 1   torque       205 non-null    float64
 2   carlength    205 non-null    float64
 3   carwidth     205 non-null    float64
 4   curbweight   205 non-null    int64  
 5   enginesize   205 non-null    int64  
 6   boreratio    205 non-null    float64
 7   horsepower   205 non-null    int64  
 8   price        205 non-null    float64
 9   fueleconomy  205 non-null    float64
 10  gas          205 non-null    uint8  
 11  turbo        205 non-null    uint8  
 12  hardtop      205 non-null    uint8  
 13  hatchback    205 non-null    uint8  
 14  sedan        205 non-null    uint8  
 15  wagon        205 non-null    uint8  
 16  fwd          205 non-null    uint8  
 17  rwd          205 non-null    uint8  
 18  rear         205 non-null    uint8  
 19  dohcv        205 non-null    uint8  
 20  l            205 non-null    uint8  
 21  ohc          205 non-null    uint8  
 22  ohcf         205 non-null    uint8  
 23  ohcv         205 non-null    uint8  
 24  rotor        205 non-null    uint8  
 25  five         205 non-null    uint8  
 26  four         205 non-null    uint8  
 27  six          205 non-null    uint8  
 28  three        205 non-null    uint8  
 29  twelve       205 non-null    uint8  
 30  two          205 non-null    uint8  
dtypes: float64(7), int64(3), uint8(21)


"""



#%% train test split  and   scaler

from sklearn.model_selection import train_test_split

train_data,test_data=train_test_split(cars,test_size=0.2,random_state=0)
train_data.head()


from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
# numerik kolonları scale edelim
# price hariç -> price (y) değişkeni scale edilmez

scale_cols = ['wheelbase', 'torque','carlength','carwidth','curbweight',
              'enginesize', 'horsepower','fueleconomy','boreratio']

train_data[scale_cols] = scaler.fit_transform(train_data[scale_cols])
train_data.head()


y_train = train_data.pop('price')
y_train.head()

X_train = train_data
train_data.head()


#%%  Model kurulumu

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
# lineer regresyon nesnesi alalım


lr=LinearRegression()
lr.fit(X_train,y_train)



#%% RFE backwards elimination


# RFE'yi hazırlayalım
# RFE(estimator, n_features_to_select)
# geriye 10 adet değişken bırakacak şekilde RFE tanımlayalım

rfe = RFE(lr, 10)
# rfe'yi train edelim

rfe = rfe.fit(X_train,y_train)

rfe.support_
rfe.ranking_

list(zip(X_train.columns,rfe.support_,rfe.ranking_))
X_train.columns[rfe.support_]

# Dolayısı ile artık önemli olan sütunları biliyoruz
X_train_rfe = X_train[X_train.columns[rfe.support_]]
X_train_rfe


# OLS için kopyalayalım

X_train_rfemodel = X_train_rfe.copy()
# statsmodels için add_constant -> beta_0 için 1'lerden oluşan sütun

X_train_rfemodel = sm.add_constant(X_train_rfemodel)
# OLS çalıştıralım

lr = sm.OLS(y_train, X_train_rfemodel).fit()
# özeti görelim

print(lr.summary())



X_train_rfemodel = X_train_rfemodel.drop(['torque'],axis=1)

def train_ols(X,y):
    X = sm.add_constant(X)
    lr = sm.OLS(y,X).fit()
    print(lr.summary())

train_ols(X_train_rfemodel, y_train)

X_train_final = X_train[['curbweight', 'enginesize', 'horsepower', 'rear', 'four',
       'six', 'twelve']]

X_train_final.columns

lr_final = LinearRegression()
lr_final.fit(X_train_final, y_train)

lr_final.coef_

katsayilar = pd.DataFrame(lr_final.coef_, index = ['curbweight', 
                                                   'enginesize', 'horsepower', 'rear', 'four',
                                                   'six', 'twelve'], columns=['Katsayı'])


katsayilar.sort_values(by=['Katsayı'], ascending=False)

train_ols(X_train_final, y_train)



"""

train_ols(X_train_final, y_train)
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  price   R-squared:                       0.887
Model:                            OLS   Adj. R-squared:                  0.882
Method:                 Least Squares   F-statistic:                     174.4
Date:                Tue, 25 Jan 2022   Prob (F-statistic):           2.18e-70
Time:                        19:09:50   Log-Likelihood:                -1522.8
No. Observations:                 164   AIC:                             3062.
Df Residuals:                     156   BIC:                             3086.
Df Model:                           7                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       8203.6127   1045.890      7.844      0.000    6137.679    1.03e+04
curbweight  1.213e+04   2138.751      5.672      0.000    7906.712    1.64e+04
enginesize  1.759e+04   3303.971      5.323      0.000    1.11e+04    2.41e+04
horsepower  4735.4440   2333.645      2.029      0.044     125.825    9345.063
rear        1.308e+04   1876.571      6.969      0.000    9371.569    1.68e+04
four       -5306.9256    834.753     -6.357      0.000   -6955.802   -3658.049
six        -4221.0668   1022.975     -4.126      0.000   -6241.736   -2200.398
twelve     -6059.7592   3283.765     -1.845      0.067   -1.25e+04     426.622
==============================================================================
Omnibus:                       16.175   Durbin-Watson:                   1.852
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               43.184
Skew:                           0.299   Prob(JB):                     4.20e-10
Kurtosis:                       5.442   Cond. No.                         27.9
==============================================================================

"""











