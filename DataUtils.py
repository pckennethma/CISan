import numpy as np

def read_table(data_path:str):
    with open(data_path) as f:
        lines = f.readlines()
    data = []
    column_values = [[] for i in range(len(lines[0].split()))]
    for line in lines[1:]:
        values = line.strip().split()[1:]
        row = []
        for idx, val in enumerate(values):
            if val not in column_values[idx]: column_values[idx].append(val)
            row.append(column_values[idx].index(val))
        data.append(row)
    data = np.array(data)
    print("Data Shape:", data.shape)
    return data