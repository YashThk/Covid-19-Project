# -*- coding: utf-8 -*-
"""
Spyder Editor
Author: YshThk
"""
import pandas as pd
import numpy as np
import os
import xlwt
import csv
import requests as rq
import datetime as dt
pd.set_option('display.max_columns', None)

response = rq.get('https://api.covidtracking.com/v1/states/daily.json')
data = pd.json_normalize(response.json())
frame = [data['date'], data['state'], data['deathConfirmed'], data['deathIncrease'], data['deathProbable'], data['hospitalizedCumulative'], data['hospitalizedCurrently'], data['hospitalizedIncrease'], data['positive'], data['positiveIncrease'], data['recovered'], data['totalTestResultsIncrease']]
header = ['date', 'state', 'deathConfirmed', 'New Deaths', 'deathProbable', 'hospitalizedCumulative', 'Hospitalizations', 'hospitalizedIncrease', 'positive', 'NewCases', 'recovered', 'Tested']
df_covid = pd.concat(frame, axis=1, keys=header)
df_covid[['date']] = df_covid[['date']].applymap(str).applymap(lambda s: "{}/{}/{}".format(s[4:6],s[6:], s[0:4]))
df_covid = df_covid.sort_values(by = 'date')

f = lambda x: x.to_csv(os.getcwd() + "/data_{}.csv".format(x.name.lower()), index=False)
df_covid.groupby('state').apply(f)
csv_data = [x for x in os.listdir('.') if x.startswith('data_') and x.endswith('.csv')]
states_recovery_missing = []

get = rq.get('https://api.census.gov/data/2019/pep/population?get=POP,NAME&for=state:*&key=1bc6279a675008577fe91748d018cbc3555dfc68')
col_names = ['Population', 'State', 'State_no']
df = pd.DataFrame(columns = col_names, data=get.json())
df = df.drop('State_no', 1)
df = df.drop([0])
df = df.sort_values(by = 'State')
df['STATE'] = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "PR", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
df = df.drop('State',1)
df = df[df.STATE != 'PR']
get2 = rq.get('http://api.census.gov/data/2019/pep/housing?get=NAME,HUEST&for=state:*&key=1bc6279a675008577fe91748d018cbc3555dfc68')
col_names2 = ['State','Housing', 'State_no']
df2 = pd.DataFrame(columns = col_names2, data=get2.json())
df2 = df2.drop('State_no', 1)
df2 = df2.drop([0])
df2 = df2.sort_values(by = 'State')
df2['STATE'] = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
df2 = df2.drop('State',1)
population = pd.merge(df, df2, on='STATE', how = "inner")
STATE = population.pop('STATE')
population.insert(0,'State', STATE)
population[['Population', 'Housing']] = population[['Population', 'Housing']].apply(pd.to_numeric)
population['AverageHousehold'] = round(population['Population']/population['Housing'],2)

def missing_columns(filename):
    df_state = pd.read_csv(filename)
    empty_columns = [cols for cols in df_state.columns if df_state[cols].isnull().all()]
    if len(empty_columns):  print(filename, ': ', empty_columns)
    if 'recovered' in empty_columns:  states_recovery_missing.append(filename)
    return states_recovery_missing

for i in csv_data:  missing_columns(i)    
optimized_data = [i for i in csv_data if i not in states_recovery_missing]
for i in states_recovery_missing: os.remove(i)

def anamoly_detection1(filename):
    df_state = pd.read_csv(filename)
    for i in df_state.index:
        if i > 0:
            if df_state['deathConfirmed'][i] > df_state['deathConfirmed'][i-1]:  df_state['deathConfirmed'][i-1] = df_state['deathConfirmed'][i]
            if df_state['hospitalizedCumulative'][i] > df_state['hospitalizedCumulative'][i-1]:  df_state['hospitalizedCumulative'][i-1] = df_state['hospitalizedCumulative'][i]
            if df_state['positive'][i] > df_state['positive'][i-1]:  df_state['positive'][i-1] = df_state['positive'][i]
            if df_state['recovered'][i] > df_state['recovered'][i-1]:  df_state['recovered'][i-1] = df_state['recovered'][i]    
    df_state.to_csv(filename, index = False)
    print(filename + ' anamoly detection request 1 : satisfied')
    
def daily_recovery(filename):
    df_state = pd.read_csv(filename, low_memory=False)
    df_state['date'] = pd.to_datetime(df_state['date'].astype(str), format='%m/%d/%Y')
    df_state['Recoveries'] = df_state['recovered'] - df_state['recovered'].shift(1)
    df_state.to_csv(filename, index = False)
    print(filename + ' addition of daily recovery column: satisfied')
    
def anamoly_detection2(filename):
    df_state = pd.read_csv(filename)
    df_state['New Deaths_MA7'] = df_state.iloc[:,3].rolling(window=7).mean()
    df_state['hospitalizedIncrease_MA7'] = df_state.iloc[:,7].rolling(window=7).mean()
    df_state['NewCases_MA7'] = df_state.iloc[:,9].rolling(window=7).mean()
    df_state['Tested_MA7'] = df_state.iloc[:,11].rolling(window=7).mean()
    df_state['New Deaths'] = np.where(df_state['New Deaths'] <= 0, round(df_state['New Deaths_MA7']), df_state['New Deaths'])
    df_state['hospitalizedIncrease'] = np.where(df_state['hospitalizedIncrease'] <= 0, round(df_state['hospitalizedIncrease_MA7']), df_state['hospitalizedIncrease'])
    df_state['NewCases'] = np.where(df_state['NewCases'] <= 0, round(df_state['NewCases_MA7']), df_state['NewCases'])
    df_state['Tested'] = np.where(df_state['Tested'] <= 0, round(df_state['Tested_MA7']), df_state['Tested'])
    df_state.to_csv(filename, index = False)
    print(filename + ' anamoly detection requqest 2 : satisfied')    
    
def anamoly_detection3(filename):
    df_state = pd.read_csv(filename)
    df_state['recoveredIncrease_MA7'] = df_state.iloc[:,12].rolling(window=7, min_periods=1).mean()
    df_state['Recoveries'] = np.where(df_state['Recoveries'] <= 0, round(df_state['recoveredIncrease_MA7']), df_state['Recoveries'])
    df_state['recoveredUpper'] = df_state['recoveredIncrease_MA7']*1.33
    df_state['recoveredLower'] = df_state['recoveredIncrease_MA7']*0.67
    df_state['Recoveries'] = np.where(df_state['Recoveries'] <= df_state['recoveredLower'], round(df_state['recoveredIncrease_MA7']), df_state['Recoveries'])
    df_state['Recoveries'] = np.where(df_state['Recoveries'] >= df_state['recoveredUpper'], round(df_state['recoveredIncrease_MA7']), df_state['Recoveries'])
    df_state = df_state.drop(['recoveredUpper', 'recoveredLower'],1)
    df_state['recoveredIncrease_2'] = df_state['Recoveries'].abs()
    df_state.to_csv(filename, index=False)
    print(filename + ' anamoly detection request 3 : satisfied')
    
def anamoly_detection4(filename):
    df_state = pd.read_csv(filename)
    df_state['recoveredIncrease_MA7_2'] = df_state.iloc[:,17].rolling(window=7).mean()
    df_state['recoveredIncrease_MA7'] = np.where(df_state['recoveredIncrease_MA7'] < 0, df_state['recoveredIncrease_MA7_2'], df_state['recoveredIncrease_MA7'])
    df_state['Recoveries'] = np.where(df_state['Recoveries'] < 0, round(df_state['recoveredIncrease_MA7']), df_state['Recoveries'])
    df_state = df_state.drop(['recoveredIncrease_2', 'recoveredIncrease_MA7_2', 'New Deaths_MA7', 'hospitalizedIncrease_MA7', 'NewCases_MA7', 'recoveredIncrease_MA7', 'Tested_MA7'],1)
    df_state.to_csv(filename, index=False)
    print(filename + ' anamoly detection request 4 : satisfied')

def weekly_recovery_reporting(filename):
    df_state = pd.read_csv(filename)
    lbl = (df_state.Recoveries != df_state.Recoveries.shift()).cumsum()
    df_state['flag'] =  (lbl.map(lbl.value_counts()) >= 3).astype(int)
    if df_state.flag.sum(axis = 0) < 14:
        df_state = df_state.drop('flag',1)
        df_state = df_state.reindex(columns = ['date', 'state', 'deathConfirmed', 'New Deaths', 'deathProbable', 'hospitalizedCumulative', 'Hospitalizations', 'hospitalizedIncrease', 'positive', 'NewCases', 'recovered', 'Recoveries', 'Tested'] )
    else:
        df_state = df_state.drop('flag',1)
        df_state['weekly_recovery reporting'] = 'Yes'
        df_state['rec_MA7_2'] = df_state.iloc[:,12].rolling(window=7).mean()
        df_state['recoveredIncrease'] = df_state['rec_MA7_2']
        df_state = df_state.drop('rec_MA7_2',1)
        df_state = df_state.reindex(columns = ['date', 'state', 'deathConfirmed', 'New Deaths', 'deathProbable', 'hospitalizedCumulative', 'Hospitalizations', 'hospitalizedIncrease', 'positive', 'NewCases', 'recovered', 'Recoveries', 'weekly_recovery reporting', 'Tested'] )
    df_state.to_csv(filename, index=False)
    print(filename + ' weekly_recovery_reporting check : performed')
    
def date_correction(filename):
    df_state = pd.read_csv(filename)
    start = dt.datetime(2020, 1, 1)
    end = dt.datetime.strptime(df_state['date'].iloc[-1], '%Y-%m-%d')
    dateList = [start + dt.timedelta(days=x) for x in range(0, (end-start).days + 1)]
    df_state2 = pd.DataFrame(dateList, columns=['date'])
    df_state['date'] = pd.to_datetime(df_state['date'].astype(str), format='%Y-%m-%d')
    df_state2['date'] = pd.to_datetime(df_state2['date'].astype(str), format='%Y-%m-%d')
    df_state3 = pd.merge_asof(df_state2, df_state, on = 'date', allow_exact_matches=True)
    df_state3['state'] = np.where(df_state3['state'].isnull(), df_state['state'].unique(), df_state3['state'])
    df_state3 = df_state3.fillna(0)
    df_state3.to_csv(filename, index = False)
    print(filename + 'starting date correction: performed')
    
def first_format(filename):
    df_state = pd.read_csv(filename)
    df_state = df_state.drop(['deathConfirmed', 'deathProbable', 'hospitalizedCumulative', 'hospitalizedIncrease', 'positive',  'recovered'],1)
    df_state.insert(2, 'SI', '')
    df_state.to_csv(filename, index=False)
    print(filename + ' fist_formatting : performed')
  
def population_column(filename):
    df_state = pd.read_csv(filename)
    for i in range(len(population['State'])):
        if population['State'][i] == df_state['state'].unique().astype(str):
            df_state['AverageHousehold'] = population['AverageHousehold'][i]
            df_state['Population'] = population['Population'][i]        
    df_state.to_csv(filename, index=False)
    print(filename + ' population & average household column: statisfied')
    
def final_format(filename):
    df_state = pd.read_csv(filename)
    if len(df_state.columns.tolist()) == 11: df_state = df_state.drop('weekly_recovery reporting',1)
    df_state['date'] = pd.to_datetime(df_state['date'], format='%Y-%m-%d').dt.strftime('%m/%d/%Y')
    df_state.to_csv(filename, index=False)
    print(filename + ' final formatting: performed')

for i in optimized_data:
    anamoly_detection1(i)
    daily_recovery(i)
    anamoly_detection2(i)
    anamoly_detection3(i)
    anamoly_detection4(i)
    weekly_recovery_reporting(i)
    date_correction(i)
    first_format(i)
    population_column(i)
    final_format(i)
    
'''________Make sure to change the folder location as per the device and code file storage location_________'''
csv_folder = "C:/Users/HP/Desktop/Research Assistant/Covid-19 project/"    
book = xlwt.Workbook(csv_folder)
for fil in optimized_data:
    sheet = book.add_sheet(fil[:-4])
    with open(csv_folder + fil) as filname:
        reader = csv.reader(filname)
        i = 0
        for row in reader:
            for j, each in enumerate(row): sheet.write(i, j, each)
            i += 1
book.save("Covid_19.xls")
print('All concerned files combined to a single Excel file.')

for i in optimized_data: os.remove(i)
print('All single csv files deleted.')