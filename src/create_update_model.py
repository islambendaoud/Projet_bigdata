import requests
import pandas as pd
import numpy as np 

# Replace 'your_link_here' with the actual link to your CSV file
csv_link ="https://www.data.gouv.fr/fr/datasets/r/62c20524-d442-46f5-bfd8-982c59763ec8"
# If the CSV file is online, you can use requests to download it
response = requests.get(csv_link)
if response.status_code == 200:
    # Get the content of the CSV file
    csv_content = response.content.decode('utf-8')

    # Specify the local file name where you want to save the CSV file
    local_file_path = 'data/usagers_file.csv'

    # Write the CSV content to a local file
    with open(local_file_path, 'w', newline='') as local_file:
        local_file.write(csv_content)

    print(f"CSV file saved to {local_file_path}")
else:
    print(f"Failed to download usagers CSV file. Status code: {response.status_code}")



# Replace 'your_link_here' with the actual link to your CSV file
csv_link ="https://www.data.gouv.fr/fr/datasets/r/c9742921-4427-41e5-81bc-f13af8bc31a0"
# If the CSV file is online, you can use requests to download it
response = requests.get(csv_link)
if response.status_code == 200:
    # Get the content of the CSV file
    csv_content = response.content.decode('utf-8')

    # Specify the local file name where you want to save the CSV file
    local_file_path = 'data/vehicule_file.csv'

    # Write the CSV content to a local file
    with open(local_file_path, 'w', newline='') as local_file:
        local_file.write(csv_content)

    print(f"CSV file saved to {local_file_path}")
else:
    print(f"Failed to download usagers CSV file. Status code: {response.status_code}")




# Replace 'your_link_here' with the actual link to your CSV file
csv_link ="https://www.data.gouv.fr/fr/datasets/r/a6ef711a-1f03-44cb-921a-0ce8ec975995"
# If the CSV file is online, you can use requests to download it
response = requests.get(csv_link)
if response.status_code == 200:
    # Get the content of the CSV file
    csv_content = response.content.decode('utf-8')

    # Specify the local file name where you want to save the CSV file
    local_file_path = 'data/lieux_file.csv'

    # Write the CSV content to a local file
    with open(local_file_path, 'w', newline='') as local_file:
        local_file.write(csv_content)

    print(f"CSV file saved to {local_file_path}")
else:
    print(f"Failed to download usagers CSV file. Status code: {response.status_code}")




csv_link="https://www.data.gouv.fr/fr/datasets/r/5fc299c0-4598-4c29-b74c-6a67b0cc27e7"
response = requests.get(csv_link)
if response.status_code == 200:
    # Get the content of the CSV file
    csv_content = response.content.decode('utf-8')

    # Specify the local file name where you want to save the CSV file
    local_file_path = 'data/charateristics_file.csv'

    # Write the CSV content to a local file
    with open(local_file_path, 'w', newline='') as local_file:
        local_file.write(csv_content)

    print(f"CSV file saved to {local_file_path}")
else:
    print(f"Failed to download usagers CSV file. Status code: {response.status_code}")


print("reading usagers_file.csv")
df_u = pd.read_csv("data/usagers_file.csv" , sep = ';' , encoding= "utf-8")
print("reading vehicule_file.csv")
df_v = pd.read_csv("data/vehicule_file.csv" , sep = ';' , encoding= "utf-8")
print("reading lieux_file.csv")
df_l = pd.read_csv("data/lieux_file.csv", sep = ';' , encoding= "utf-8" , low_memory=False)
print("reading charateristics_file.csv")
df_c = pd.read_csv("data/charateristics_file.csv" , sep = ';' , encoding= "utf-8")



df_uv = pd.merge(df_u, df_v, on='id_vehicule' , suffixes=('','_y'))
df_uv = df_uv.drop(df_uv.filter(regex='_y$').columns.tolist(),axis=1)
df_uvc = pd.merge(df_uv,df_c,left_on='Num_Acc', right_on='Accident_Id', suffixes=('','_y'))
df = pd.merge(df_uvc, df_l, on='Num_Acc' , suffixes=('','_y'))
df = df.drop(df.filter(regex='_y$').columns.tolist(),axis=1)


df.dropna(thresh=0.5*len(df), axis=1, inplace=True)

df = df.select_dtypes(exclude=['object'])
df.drop(columns =[ 'Num_Acc'], inplace=True)
df.drop(columns =[ 'Accident_Id'], inplace=True)

# import train_test_split from sklearn
from sklearn.model_selection import train_test_split

# y is df['grav'] and X is df.drop(columns=['grav'])
y = df['grav']
y = y.replace(-1,0)
X = df.drop(columns=['grav'])


# treat the na values by replacing them with the mean of the column
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values= np.nan, strategy='mean')
imp.fit(X)
X = imp.transform(X)


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



model = XGBClassifier()
model.fit(X_train, y_train)
print("Model trained")
print("Model score calculated" , model.score(X_test, y_test))
import pickle
pickle.dump(model, open("models/model.pkl", "wb"))
print("Model saved locally as model.pkl")
