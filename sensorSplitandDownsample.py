import numpy as np
import pandas as pd

input_folder = 'data/'
output_folder = 'data/Leak4/'
split_list = ['Combo_NoLeak_test3.csv','Combo_BigLeak_test3.csv','Combo_SmallLeak_test3.csv','Combo_BothLeak_test3.csv','Combo_NoLeak_test3_wpump.csv']
splits = 1200

for file in split_list:
    print(file)
    data_in = pd.read_csv(input_folder + file)
    data_len = len(data_in)
    split_nums = np.round(np.linspace(0,data_len,num=splits+1),0)
    for i, index in np.ndenumerate(split_nums):
        # print(i)
        if i != (0,):
            temp_split = data_in[int(split_nums[i[0]-1]):int(index)]
            # temp_split = temp_split.rename(columns={temp_split.columns[1]:'data'})
            temp_split.to_csv(output_folder + file[0:len(file) - 4] + f'_split{i[0]}.csv', index=False)
            # times = []
            # data = []
            # for j, row in temp_split.iterrows():
            #     # if j > 3:
            #     # print(row)
            #     # print(row['Timestamp'])
            #     # print(type(row['Timestamp']))
            #     if (j % 3) == 0:
            #         times.append(row['Timestamp'])
            #         data.append(row['data'])
            # temp_split2 = pd.DataFrame({'Timestamp':times,'data':data})
            # temp_split2.to_csv(file[0:len(file)-4] + f'_split{i[0]}.csv',index = False)

# files = ['NoLeak_test3.csv','BigLeak_test3.csv','SmallLeak_test3.csv','BothLeak_test3.csv','NoLeak_test3_wpump.csv']
# for file in files:
#     data1 = pd.read_csv(input_folder + 'H1_' + file)
#     data2 = pd.read_csv(input_folder + 'H2_' + file)
#     data3 = pd.read_csv(input_folder + 'A_' + file)
#     data1 = data1.rename(columns={data1.columns[1]:'H1'})
#     data1['H2'] = data2['V']
#     data1['A'] = data3['V']
#     data1.to_csv(input_folder + 'Combo_' + file, index=False)

# for file in split_list:
#     data = pd.read_csv(input_folder + file)
#     data['Timestamp'] = data['Timestamp'] - 3
#     data['Timestamp'] = data['Timestamp'].round(10)
#     data = data[(data['Timestamp'] >= 0) & (data['Timestamp'] <= 600)]
#     # print(data)
#     print(len(data))
#     data.to_csv(input_folder + file, index=False)