"""
1. Verinin genel hatlarını tanımak. Kaç gözlem var, kaç değişken var. Verinin iç ve dış yapısını tanımak.
Eksik değer var mı, varsa kaç tane var gibi gibi.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe()
df.isnull().sum()
df.isnull().values.any()


def check_df(dataframe, head=5):
    print("##################### SHAPE ########################")
    print(dataframe.shape)
    print("##################### TYPES ########################")
    print(dataframe.dtypes)
    print("##################### HEAD ########################")
    print(dataframe.head(head))
    print("##################### TAIL ########################")
    print(dataframe.tail(head))
    print("##################### NA ########################")
    print(dataframe.isnull().sum())
    print("##################### QUANTILES ########################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

#df = sns.load_dataset("tips")
#df = sns.load_dataset("exercise")
"""
ilk önce kategorik değişkenleri bulduk, sonra nümerik görünen ama aslında kategorik olan değişkenleri bulduk. 
10dan az sınıf varsa kategoriktr dedik. bu yoruma bağlı. categorik olan ama anlam ifade etmeyen, ad gibi belirli
bir kategorik veri sağlamayan için de eğer unique değerl sayısı 20den fazlaysa ve 
datatypeı category ya da object ise bu listeye koy.
sonra kategorigk columnsları eskisi + num_but_cat ile birleştirip güncelliyoruz. sonra da
cat_colsdan cat_but_car olanları çıkarıyoruz çünkü yapı olarak kategorik olsalar da
kategorik anlam içermiyorlar. cat but car: kategorşk ama kardinal değişken
"""
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]
df[cat_cols].nunique()

#bunlar kategorik olmayanlar

[col for col in df.columns if col not in cat_cols]

df["survived"].value_counts()
#ilgili kategorik değişkenin oran bilgisi
100 * df["survived"].value_counts() / len(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
                        }))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


cat_summary(df, "sex", True)

for col in cat_cols:
    cat_summary(df, col, True)

#adult male boolean type ama astype int deyince int oluyor
df["adult_male"].astype(int)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
    else:
        cat_summary(df, col, True)

df[["age", "fare"]].describe().T

num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
num_cols = [col for col in num_cols if col not in cat_cols]


def num_summary(dataframe, numerical_col, plot=False):
    print("#############################")
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


num_summary(df, "age")

for col in num_cols:
    num_summary(df, col, True)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
  Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
  Parameters
  ----------
  dataframe: Dataframe
    değişken isimleri alınmak istenen dataframedir.
  cat_th: int, float
    numerik fakat kategorik olan değişkenler için sınıf eşik değeri
  car_th: int, float
    kategorik fakat kardinal değişkenler için sınıf eşik değeri

  Returns
  -------
  cat_cols: list
    kategorik değişken listesi
  num_cols: list
    numeric değişken listesi
  cat_but_car: kategorik görünümlü kardinal değişken listesi

  Notes
  -------
  cat_cols + num_cols + cat_but_car = toplam değişken sayısı
  num_but_cat cat_cols'un içerisinde.
  return olan 3 liste toplamı toplam değişken sayısına eşittir:
  """
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].dtypes in ["int", "float"] and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in cat_cols if
                   str(dataframe[col].dtypes) in ["category", "object"] and dataframe[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


grab_col_names(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    cat_summary(df, col, True)

for col in num_cols:
    num_summary(df, col, True)

for col in cat_but_car:
    cat_summary(df, col, True)

#bonus

df = sns.load_dataset("titanic")
df.info()
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

df["survived"].value_counts()
cat_summary(df, "survived")

#kadın olmak hayatta kalmayı etkileyen bir faktör olabilir.
df.groupby("sex")["survived"].mean()


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))

#first class yolcular daha fazla hayatta kalmış
target_summary_with_cat(df, "survived", "pclass")

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "survived", col)