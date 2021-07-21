import pandas as pd


fileName = 'NSE-TATA.csv'

df = pd.read_csv(fileName)

df['CloseRateOfChange'] = ''

fileNameResult = fileName[0:len(fileName) - 4] + 'Result.csv'


# Do trong file csv, index = 0 tương ứng dòng đầu tiên của file, ta duyệt từ trên xuống, nên previous_value có index + 1
for index, row in df.iterrows():
    current_value = df.at[index, 'Close']
    previous_value = 0
    if index == len(df.index) - 1:
        previous_value = current_value
    else:
        previous_value = df.at[index + 1, 'Close']
    
    Rate_Of_Change = (current_value/previous_value - 1) * 100
    df.at[index, 'CloseRateOfChange'] = Rate_Of_Change

df.to_csv(fileNameResult,index=False, encoding='utf-8-sig')

