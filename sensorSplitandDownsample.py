import numpy as np
import pandas as pd

split_list = ['data/H1_NoLeak_test2.csv','data/H1_SmallLeak_test2.csv','data/H1_BigLeak_test2.csv'] #'data/H1_Leak_test1.csv',
splits = 240

for file in split_list:
    data_in = pd.read_csv(file)
    data_len = len(data_in)
    # print(data_len)
    split_nums = np.round(np.linspace(0,data_len,num=splits),0)
    # print(splits)
    for i, index in np.ndenumerate(split_nums):
        # print(i)
        # if i != (0,):
        temp_split = data_in[int(split_nums[i[0]-1]):int(index)]
        temp_split = temp_split.rename(columns={temp_split.columns[1]:'data'})
        times = []
        data = []
        for j, row in temp_split.iterrows():
            # if j > 3:
            # print(row)
            # print(row['Timestamp'])
            # print(type(row['Timestamp']))
            if (row['Timestamp']*100000 % 10) == 0:
                times.append(row['Timestamp'])
                data.append(row['data'])
        temp_split2 = pd.DataFrame({'Timestamp':times,'data':data})
        temp_split2.to_csv(file[0:len(file)-4] + f'_split{i[0]+1}.csv',index = False)