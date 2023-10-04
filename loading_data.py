import  pandas as pd

def loading_data():
    data = pd.read_csv('./Persistent_vs_NonPersistent.csv')
    return data

loading_data()