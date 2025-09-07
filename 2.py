from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
cancer = load_breast_cancer()
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df.head()
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)
df = df.iloc[:, 1:-1]

"""
Korelasyon: Değişkenlerin ne kadar ilişkili olduğunu belirtir. -1 1 arasında değişir.
-1 ya da 1 e yaklaştıkça ilişkinin şiddeti kuvvetlenir.  1 e yaklaşıyorsa pozitif kore
lasyon, doğru orantı, -1e yaklaşıyprsa - korelasyon, ters orantı. 0a yakınsa düşük korelasyon
Eğer korelasyon çoksa, bu ikisinin aynı anda bulunmamasını tercih ederiz kalabalık oluşturmaması adına.
Örneğin radius mean perimeter mean aralarında 0.99 kor var neredeyse aynı falanlar perimeter
2 ile çarpılmış hali. Bize bilgi anlamında fark sağlamıyor. At gitsin.
"""

num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

# Yüksek korelasyonlu  değişkenlerin silinmesi

cor_matrix = df.corr().abs()
upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
drop_columns = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]
new_cols = [col for col in df.columns if col not in drop_columns]
df[drop_columns].shape
df[new_cols].shape
df.shape

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_columns = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90)]
    if plot:
        sns.set(rc={'figure.figsize':(15,15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_columns

high_correlated_cols(df, True)
drop_list = high_correlated_cols(df)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)