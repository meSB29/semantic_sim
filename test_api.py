import requests
import pandas as pd
import numpy as np
# 1905
df=pd.read_csv('Precily_Text_Similarity.csv')
row=np.random.randint(0,len(df))


url = 'http://localhost:5000/api/calculate_similarity'
data = {
    "text1": str(df.iloc[row,0]),
    "text2": str(df.iloc[row,1])
}

response = requests.post(url, json=data)
print(row)
print(response.json())
