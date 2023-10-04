import seaborn as sns
import matplotlib.pyplot as plt
from data_preprocessing import data_preprocess

def data_visualization():
    data=data_preprocess()
    sns.countplot(x="Persistency_Flag",data=data, dodge=True)
    sns.countplot(x="Persistency_Flag",hue='Tscore_Bucket_Prior_Ntm', data=data)
    sns.countplot(x="Persistency_Flag",hue='Adherent_Flag', data=data)
    sns.countplot(x="Persistency_Flag", hue='Age_Bucket', data=data)
    sns.countplot(x="Persistency_Flag", hue='Gender', data=data)
    plt.show()
    return data

data_visualization()
