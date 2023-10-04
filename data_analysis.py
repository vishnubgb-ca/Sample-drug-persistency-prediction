from loading_data import loading_data

def data_analysis():
    data=loading_data()
    data.head()
    data.tail()
    data.describe
    print ("Rows     : " ,data.shape[0])
    print ("Columns  : \n" ,data.shape[1])
    print ("Features : \n" ,data.columns.tolist())
    print ("Missing values\n\n",data.isnull().any(),sep='')
    print ("Unique values\n\n",data.nunique(),sep='')

data_analysis()