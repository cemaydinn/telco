import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Veri setini yükleme
data = pd.read_csv('C:/Users/Casper/Desktop/miuul/Proje/odev2/TelcoChurn/Telco/Telco-Customer-Churn')

# Görev 1: Keşifçi Veri Analizi
# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
print(data[numerical_cols].describe())
print(data[categorical_cols].describe())

# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
for col in categorical_cols:
  print(data.groupby(col)['Churn'].value_counts(normalize=True))

# Adım 5: Aykırı gözlem var mı inceleyiniz.
# Basit bir aykırı değer analizi
for col in numerical_cols:
  print(f"{col} outliers: ", data[col][(data[col] - data[col].mean()).abs() > 3 * data[col].std()])

# Adım 6: Eksik gözlem var mı inceleyiniz.
print(data.isnull().sum())

# Görev 2: Feature Engineering
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
data.fillna(data.median(), inplace=True)

# Adım 2: Yeni değişkenler oluşturunuz.
data['TotalServices'] = (data[['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                             'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                             'StreamingMovies']] == 'Yes').sum(axis=1)

# Adım 3: Encoding işlemlerini gerçekleştiriniz.
label_encoders = {}
for col in categorical_cols:
  le = LabelEncoder()
  data[col] = le.fit_transform(data[col])
  label_encoders[col] = le

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Görev 3: Modelleme
# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.
X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Basit bir modelleme örneği
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile modeli tekrar kurunuz.
param_grid = {
  'n_estimators': [100, 200],
  'max_depth': [None, 10, 20],
  'min_samples_split': [2, 5],
  'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)
print("Optimized Random Forest Accuracy:", accuracy_score(y_test, y_pred_best))