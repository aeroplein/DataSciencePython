import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)

df = pd.read_csv("persona.csv")
print(df.head())
print(df.tail())
df.shape
df.columns
df.dtypes
df["SOURCE"].nunique() # 2 unique source
df["SOURCE"].value_counts() # android 2974 ios 2026
df["PRICE"].nunique() #6
df["PRICE"].value_counts()
# 29    1305
# 39    1260
# 49    1031
# 19     992
# 59     212
# 9      200
df["COUNTRY"].value_counts()
# usa    2065
# bra    1496
# deu     455
# tur     451
# fra     303
# can     230
df.groupby("COUNTRY")["PRICE"].sum()
# bra    51354
# can     7730
# deu    15485
# fra    10177
# tur    15689
# usa    70225
df.groupby("COUNTRY")["PRICE"].count()
# bra    1496
# can     230
# deu     455
# fra     303
# tur     451
# usa    2065
df.groupby("SOURCE")["PRICE"].count()
# android    2974
# ios        2026
df.groupby("COUNTRY")["PRICE"].mean()
# bra    34.327540
# can    33.608696
# deu    34.032967
# fra    33.587459
# tur    34.787140
# usa    34.007264
df.groupby("SOURCE")["PRICE"].mean()
# android    34.174849
# ios        34.069102
df.pivot_table("PRICE", ["COUNTRY", "SOURCE"], aggfunc="mean")
# PRICE
# COUNTRY SOURCE
# bra     android  34.387029
# ios      34.222222
# can     android  33.330709
# ios      33.951456
# deu     android  33.869888
# ios      34.268817
# fra     android  34.312500
agg_df = df.pivot_table("PRICE",
               ["COUNTRY", "SOURCE", "SEX", "AGE"],
               aggfunc="mean").sort_values("PRICE",
                                           ascending=False).reset_index()

bins = [0, 18, 23, 30, 40, 70]
labels = ['0_18', '19_23', '24_30', '31_40', '41_70']
agg_df["AGE_CAT"] = pd.cut(df["AGE"], bins=bins, labels=labels)
cols = agg_df.select_dtypes(include=["object", "category"]).columns
agg_df["customers_level_based"] = agg_df[cols].apply(lambda x: '_'.join(x.str.upper()), axis=1)
# Her bir row yani satır x olarak alınır, yani sırasıyla satırlara uygulanır.
#lambda ile o satırdaki tüm kat değerleri alıyoruz
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], q=4, labels=["D", "C", "B", "A"])
agg_df.groupby("SEGMENT")["PRICE"].agg(["mean", "max", "sum"])
new_user = 'TUR_ANDROID_FEMALE_31_40'
agg_df[agg_df["customers_level_based"] == new_user]
agg_df[agg_df["customers_level_based"] == "FRA_IOS_FEMALE_31_40"]
agg_df["customers_level_based"].unique() #empty
"FRA_IOS_FEMALE_31_40" in agg_df["AGE_CAT"].unique()