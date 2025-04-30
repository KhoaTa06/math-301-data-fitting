import pandas as pd

dataFilepath = "/Users/khoa/Desktop/Math_301/math-301-data-fitting/khoa_code/Group_1_Data_10000.txt"
data = pd.read_csv(dataFilepath)
data.columns
data.drop('y)', axis=1, inplace=True)
data.columns=["x", "y", "fxy"]

def cut_dataset(data, size=1000):
    cut_data = pd.DataFrame({"x": [], "y": [], "fxy": []})
    space = 10000 / size
    for i in range(1000):
        cut_data.loc[len(cut_data)] = data.iloc[int(i * space)]
    return cut_data

size = 200
cut_data = cut_dataset(data)
cut_data.to_csv(f'Data_{size}.csv', index=False)