import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)
df = pd.read_excel("miuul_gezinomi.xlsx")


# Veri seti ile ilgili genel bilgiler
def check_df(dataframe, head=5):
    """
    Gives a general picture of the given dataframe.
    :param dataframe: DataFrame
        dataset that we want to get the variables from
    :param head: int
        threshold for head and tail
    :return:
    """
    print("################################")
    print("SHAPE")
    print(dataframe.shape)
    print("COLUMNS")
    print(dataframe.columns)
    print("DATA TYPES")
    print(dataframe.dtypes)
    print("HEAD")
    print(dataframe.head(head))
    print("TAIL")
    print(dataframe.tail(head))
    print("NA")
    print(dataframe.isnull().sum())
    print("QUANTILES")
    print(dataframe.describe([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1]))


check_df(df)
df["SaleCityName"].nunique()  #6
df["SaleCityName"].unique()
df["SaleCityName"].value_counts()
df["ConceptName"].value_counts()  #3

df.groupby("ConceptName")["SaleId"].count()
df.groupby("SaleCityName")["Price"].sum()
df.groupby("ConceptName")["Price"].sum()
df.groupby("SaleCityName")["Price"].mean()
df.groupby("ConceptName")["Price"].mean()
df.groupby(["SaleCityName", "ConceptName"])["Price"].mean()

bins = [0, 7, 30, 90, df["SaleCheckInDayDiff"].max() + 1]  #+1 dedim çünkü max değeri dahil etmediği için ona
# kategori veremedi, NaN kaldı. O yüzden 1 ekledim.
labels = ["Last Minuters", "Potential Planners", "Planners", "Early Bookers"]
df["EB_Score"] = pd.cut(df["SaleCheckInDayDiff"], bins=bins, labels=labels, right=False)
df.columns

city_concept_score = pd.pivot_table(df, index=["SaleCityName", "ConceptName", "EB_Score"],
                                    values="Price", aggfunc=["mean", "count"]).reset_index()
city_concept_season = pd.pivot_table(df, index=["SaleCityName", "ConceptName", "Seasons"], values="Price",
                                     aggfunc=["mean", "count"])
season_concept_cin = pd.pivot_table(df, index=["Seasons", "ConceptName", "CInDay"], values="Price",
                                    aggfunc=["mean", "count"])
agg_df = city_concept_season.sort_values(by=("mean", "Price"), ascending=False)
city_concept_season.columns
city_concept_score.reset_index()
city_concept_season.reset_index()
season_concept_cin.reset_index()

df["sales_level_based"] = (
        df["SaleCityName"].str.upper() + "_" +
        df["ConceptName"].str.upper().str.replace(" ", "_") + "_" +  #boşlukları _ile değiştirdik
        df["Seasons"].str.upper()
)

agg_df = df.groupby("sales_level_based").agg({"Price": "mean"}).reset_index()  #sales level basede göre
#fiyat ortalamasını hesaplayıp indexlerimizi 01234 diye düzenledik

agg_df["Segment"] = pd.qcut(agg_df["Price"], 4, labels=["D", "C", "B", "A"])  #quartilelara göre
#price columnunu 4 quartileını  a b c d diye etiketlendirdik. 0.25 quartile d olur gib gibi.
agg_df.groupby("Segment")["Price"].agg(["mean", "max", "sum"])
agg_df.sort_values("Price")
#segmente göre fiyat ortalaması, max değer ve toplam bulduk.
new_user = "ANTALYA_HERŞEY_DAHIL_HIGH"
agg_df[agg_df["sales_level_based"] == new_user]
agg_df["sales_level_based"].unique()
agg_df[agg_df["sales_level_based"] == "GIRNE_YARIM_PANSIYON_LOW"]
