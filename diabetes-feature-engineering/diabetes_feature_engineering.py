##############################################################
# DİYABET ÖZELLİK MÜHENDİSLİĞİ (DIABETES FEATURE ENGINEERING)
##############################################################

# İş Problemi
# -----------
# Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını
# tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.
# Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği
# adımları gerçekleştirilecektir.

# Veri Seti Hikayesi
# -------------------
# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde
# tutulan büyük veri setinin parçasıdır. Arizona'daki Phoenix şehrinde yaşayan
# 21 yaş ve üzerindeki Pima Indian kadınları üzerinde yapılan diyabet araştırması
# için kullanılan verilerdir. 9 Değişken, 768 Gözlem.
#
# Pregnancies                : Hamilelik sayısı
# Glucose                    : Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
# BloodPressure               : Kan basıncı (küçük tansiyon) (mm Hg)
# SkinThickness               : Cilt kalınlığı
# Insulin                     : 2 saatlik serum insülini (mu U/ml)
# DiabetesPedigreeFunction    : Soydaki kişilere göre diyabet olma ihtimalini hesaplayan fonksiyon
# BMI                         : Vücut kitle endeksi
# Age                         : Yaş (yıl)
# Outcome                     : Hastalığa sahip (1) ya da değil (0)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


######################################
# YARDIMCI FONKSİYONLAR (HELPER FUNCTIONS)
######################################

def check_df(dataframe, head=5):
    """Veri setine dair genel resmi (boyut, tip, head/tail, eksik değer, quantile) özetler."""
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik görünümlü kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenen dataframe.
    cat_th: int, optional
        Numerik fakat kategorik olan değişkenler için sınıf eşik değeri.
    car_th: int, optional
        Kategorik fakat kardinal değişkenler için sınıf eşik değeri.

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


def cat_summary(dataframe, col_name, plot=False):
    """Kategorik değişken için sınıf frekansı ve oranını özetler."""
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                         "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


def num_summary(dataframe, numerical_col, plot=False):
    """Numerik değişken için betimsel istatistikleri özetler."""
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def target_summary_with_cat(dataframe, target, categorical_col):
    """Kategorik değişkenlere göre hedef değişkenin ortalamasını verir."""
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def target_summary_with_num(dataframe, target, numerical_col):
    """Hedef değişkene göre numerik değişkenlerin ortalamasını verir."""
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """Bir değişken için IQR yöntemiyle alt ve üst aykırı değer eşiklerini hesaplar."""
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    """Bir değişkende eşik değerleri aşan aykırı gözlem olup olmadığını döner (True/False)."""
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    return False


def replace_with_thresholds(dataframe, variable):
    """Aykırı değerleri hesaplanan alt/üst eşik değerleriyle baskılar (capping)."""
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def missing_values_table(dataframe, na_name=False):
    """Eksik değer içeren değişkenleri, eksik gözlem sayısı ve oranıyla birlikte listeler."""
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


######################################
# GÖREV 1: KEŞİFÇİ VERİ ANALİZİ (EDA)
######################################

import os
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/diabetes.csv")

# Adım 1: Genel resmi inceleyiniz.
check_df(df)

# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
for col in cat_cols:
    cat_summary(df, col)

for col in num_cols:
    num_summary(df, col)

# Adım 4: Hedef değişken analizi yapınız.
# (Kategorik değişkenlere göre hedef değişkenin ortalaması,
#  hedef değişkene göre numerik değişkenlerin ortalaması)
for col in cat_cols:
    target_summary_with_cat(df, "Outcome", col)

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

# Adım 5: Aykırı gözlem analizi yapınız.
for col in num_cols:
    print(col, check_outlier(df, col))

# Adım 6: Eksik gözlem analizi yapınız.
missing_values_table(df)
# Not: Veri setinde NaN olarak işaretlenmiş eksik gözlem yok; ancak Glucose,
# BloodPressure gibi değişkenlerde 0 değeri fizyolojik olarak imkansızdır ve
# gerçekte eksik veriyi temsil eder. Bu durum Görev 2 - Adım 1'de ele alınacaktır.

# Adım 7: Korelasyon analizi yapınız.
corr = df[num_cols].corr()
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu", annot=True)
plt.title("Korelasyon Matrisi")
plt.tight_layout()
plt.savefig("outputs/correlation_matrix.png")
plt.close()


######################################
# GÖREV 2: FEATURE ENGINEERING
######################################

# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız.
# Glikoz, Insulin vb. değişkenlerde 0 değeri eksik veriyi ifade ediyor olabilir
# (bir kişinin glikoz veya insülin değeri 0 olamaz). Bu nedenle ilgili
# değişkenlerdeki 0 değerleri NaN olarak işaretlenir.

zero_not_allowed_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_not_allowed_cols:
    df[col] = df[col].replace(0, np.nan)

missing_values_table(df)

# Eksik değerler, ilgili değişkenin Outcome kırılımındaki medyanı ile doldurulur.
for col in zero_not_allowed_cols:
    df[col] = df[col].fillna(df.groupby("Outcome")[col].transform("median"))

missing_values_table(df)  # kontrol: eksik gözlem kalmamalı

# Aykırı değerler eşik değerleriyle baskılanır (capping).
for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))  # kontrol: hepsi False olmalı

# Adım 2: Yeni değişkenler oluşturunuz.

# Yaşa göre kategori
df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

# BMI'a göre kategori (WHO sınıflandırması)
df["NEW_BMI"] = pd.cut(df["BMI"], bins=[0, 18.5, 24.9, 29.9, 100],
                        labels=["Underweight", "Healthy", "Overweight", "Obese"])

# Glikoz seviyesine göre kategori
df["NEW_GLUCOSE"] = pd.cut(df["Glucose"], bins=[0, 140, 200, 300],
                            labels=["Normal", "Prediabetes", "Diabetes"])

# Yaş ve beden kitle indeksini bir arada değerlendirme
df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

# Yaş ve Glikoz değerlerini bir arada değerlendirme
df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"

# İnsülin değerine göre kategori
def set_insulin(dataframe, col_name="Insulin"):
    return "Normal" if 16 <= dataframe[col_name] <= 166 else "Abnormal"

df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

# Glikoz * İnsülin ve diğer etkileşim değişkenleri
df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]
df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]

# Değişken isimlerinin büyütülmesi
df.columns = [col.upper() for col in df.columns]

print(df.head())
print(df.shape)

# Adım 3: Encoding işlemlerini gerçekleştiriniz.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Label Encoding (iki sınıflı kategorik değişkenler)
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    df = label_encoder(df, col)

# One-Hot Encoding (ikiden fazla sınıflı kategorik değişkenler)
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

print(df.head())
print(df.shape)

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col != "OUTCOME"]

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print(df.head())
print(df.shape)

# Adım 5: Model oluşturunuz.
y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 2)}")
print(f"Recall: {round(recall_score(y_test, y_pred), 3)}")
print(f"Precision: {round(precision_score(y_test, y_pred), 2)}")
print(f"F1: {round(f1_score(y_test, y_pred), 2)}")
print(f"AUC: {round(roc_auc_score(y_test, y_pred), 2)}")


# Değişken önem düzeyleri (feature importance)
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature",
                data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Değişken Önem Düzeyleri (Feature Importances)")
    plt.tight_layout()
    if save:
        plt.savefig("outputs/feature_importance.png")
    plt.show(block=True)


plot_importance(rf_model, X, save=True)
