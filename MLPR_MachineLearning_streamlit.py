import streamlit as st
import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from dateutil.relativedelta import relativedelta

df1 = pd.read_csv('Data_Emission.csv')
df1 = df1.query("Country == 'Indonesia' | Country == 'Argentina' | Country == 'United Kingdom' | Country == 'Brazil' | Country == 'Canada' | Country == 'United States' | Country == 'Italy' | Country == 'France' | Country == 'Germany' | Country == 'South Africa' | Country == 'Japan' | Country == 'Mexico' | Country == 'Saudi Arabia' | Country == 'Turkey' | Country == 'Australia' | Country == 'China' | Country == 'India'")
df1 = df1.drop('Data source', axis=1)
df1 = df1.drop('Sector', axis=1)
df1 = df1.drop('Gas', axis=1)
df1 = df1.drop('Unit', axis=1)

ss = StandardScaler()

def GenerateMonthlyEmissionData(Dataframe):
  years = 28

  dfm = pd.DataFrame(columns=['Emission Amount'])
  df_old = Dataframe
  i = 0
  j = 0
  k = 0
  counter = 0

  monthIndex = []

  while(years!=0):
    months = 12
    j = i + 1

    start_emission = df_old['Emission Amount'].iloc[i]
    if(i < 27):
      end_emission = df_old['Emission Amount'].iloc[j]
    else:
      monthIndex.append(counter+1)
      dfn = pd.DataFrame(monthIndex, columns = ['Month'])
      dfm['Month'] = dfn['Month']
      result = dfm
      result.set_index('Month')
      return result

    diff = end_emission - start_emission
    divdiff = round(diff / 12,2) # 11 -> months - 1

    dfm.loc[k] = start_emission

    m = 0

    while(months!=0):
      m = m + 1
      k = k + 1

      monthincrement = divdiff * m
      noise = np.random.normal(0, 0.1)
      dfm.loc[k] =  round(start_emission + monthincrement + noise,2)
      months = months - 1
      counter = counter + 1
      monthIndex.append(counter)

    i = i + 1
    years = years - 1


# ---------------------- PREPARING FOR MONTHLY GENERATION -------------------------#

df1_IND = df1.query("Country == 'Indonesia'")

t1_IND = df1_IND.transpose()
t1_IND = t1_IND.reset_index()
t1_IND = t1_IND.drop([0])
t1_IND = t1_IND.rename(columns = {"index":'Year',5:"Emission Amount"})
x_IND = GenerateMonthlyEmissionData(t1_IND)

df1_ARG = df1.query("Country == 'Argentina'")

t1_ARG = df1_ARG.transpose()
t1_ARG = t1_ARG.reset_index()
t1_ARG = t1_ARG.drop([0])
t1_ARG = t1_ARG.rename(columns = {"index":'Year',30:"Emission Amount"})
x_ARG = GenerateMonthlyEmissionData(t1_ARG)

df1_ENG = df1.query("Country == 'United Kingdom'")

t1_ENG = df1_ENG.transpose()
t1_ENG = t1_ENG.reset_index()
t1_ENG = t1_ENG.drop([0])
t1_ENG = t1_ENG.rename(columns = {"index":'Year',19:"Emission Amount"})
x_ENG = GenerateMonthlyEmissionData(t1_ENG)

df1_BRZ = df1.query("Country == 'Brazil'")

t1_BRZ = df1_BRZ.transpose()
t1_BRZ = t1_BRZ.reset_index()
t1_BRZ = t1_BRZ.drop([0])
t1_BRZ = t1_BRZ.rename(columns = {"index":'Year',8:"Emission Amount"})
x_BRZ = GenerateMonthlyEmissionData(t1_BRZ)

df1_CAN = df1.query("Country == 'Canada'")

t1_CAN = df1_CAN.transpose()
t1_CAN = t1_CAN.reset_index()
t1_CAN = t1_CAN.drop([0])
t1_CAN = t1_CAN.rename(columns = {"index":'Year',11:"Emission Amount"})
x_CAN = GenerateMonthlyEmissionData(t1_CAN)

df1_USA = df1.query("Country == 'United States'")

t1_USA = df1_USA.transpose()
t1_USA = t1_USA.reset_index()
t1_USA = t1_USA.drop([0])
t1_USA = t1_USA.rename(columns = {"index":'Year',2:"Emission Amount"})
x_USA = GenerateMonthlyEmissionData(t1_USA)

df1_ITA = df1.query("Country == 'Italy'")

t1_ITA = df1_ITA.transpose()
t1_ITA = t1_ITA.reset_index()
t1_ITA = t1_ITA.drop([0])
t1_ITA = t1_ITA.rename(columns = {"index":'Year',21:"Emission Amount"})
x_ITA = GenerateMonthlyEmissionData(t1_ITA)

df1_FRC = df1.query("Country == 'France'")

t1_FRC = df1_FRC.transpose()
t1_FRC = t1_FRC.reset_index()
t1_FRC = t1_FRC.drop([0])
t1_FRC = t1_FRC.rename(columns = {"index":'Year',24:"Emission Amount"})
x_FRC = GenerateMonthlyEmissionData(t1_FRC)

df1_GRM = df1.query("Country == 'Germany'")

t1_GRM = df1_GRM.transpose()
t1_GRM = t1_GRM.reset_index()
t1_GRM = t1_GRM.drop([0])
t1_GRM = t1_GRM.rename(columns = {"index":'Year',9:"Emission Amount"})
x_GRM = GenerateMonthlyEmissionData(t1_GRM)

df1_SAF = df1.query("Country == 'South Africa'")

t1_SAF = df1_SAF.transpose()
t1_SAF = t1_SAF.reset_index()
t1_SAF = t1_SAF.drop([0])
t1_SAF = t1_SAF.rename(columns = {"index":'Year',16:"Emission Amount"})
x_SAF = GenerateMonthlyEmissionData(t1_SAF)

df1_JPN = df1.query("Country == 'Japan'")

t1_JPN = df1_JPN.transpose()
t1_JPN = t1_JPN.reset_index()
t1_JPN = t1_JPN.drop([0])
t1_JPN = t1_JPN.rename(columns = {"index":'Year',6:"Emission Amount"})
x_JPN = GenerateMonthlyEmissionData(t1_JPN)

df1_MXC = df1.query("Country == 'Mexico'")

t1_MXC = df1_MXC.transpose()
t1_MXC = t1_MXC.reset_index()
t1_MXC = t1_MXC.drop([0])
t1_MXC = t1_MXC.rename(columns = {"index":'Year',15:"Emission Amount"})
x_MXC = GenerateMonthlyEmissionData(t1_MXC)

df1_ARB = df1.query("Country == 'Saudi Arabia'")

t1_ARB = df1_ARB.transpose()
t1_ARB = t1_ARB.reset_index()
t1_ARB = t1_ARB.drop([0])
t1_ARB = t1_ARB.rename(columns = {"index":'Year',14:"Emission Amount"})
x_ARB = GenerateMonthlyEmissionData(t1_ARB)

df1_TUR = df1.query("Country == 'Turkey'")

t1_TUR = df1_TUR.transpose()
t1_TUR = t1_TUR.reset_index()
t1_TUR = t1_TUR.drop([0])
t1_TUR = t1_TUR.rename(columns = {"index":'Year',18:"Emission Amount"})
x_TUR = GenerateMonthlyEmissionData(t1_TUR)

df1_AUS = df1.query("Country == 'Australia'")

t1_AUS = df1_AUS.transpose()
t1_AUS = t1_AUS.reset_index()
t1_AUS = t1_AUS.drop([0])
t1_AUS = t1_AUS.rename(columns = {"index":'Year',17:"Emission Amount"})
x_AUS = GenerateMonthlyEmissionData(t1_AUS)

df1_CHN = df1.query("Country == 'China'")

t1_CHN = df1_CHN.transpose()
t1_CHN = t1_CHN.reset_index()
t1_CHN = t1_CHN.drop([0])
t1_CHN = t1_CHN.rename(columns = {"index":'Year',1:"Emission Amount"})
x_CHN = GenerateMonthlyEmissionData(t1_CHN)

df1_INA = df1.query("Country == 'India'")

t1_INA = df1_INA.transpose()
t1_INA = t1_INA.reset_index()
t1_INA = t1_INA.drop([0])
t1_INA = t1_INA.rename(columns = {"index":'Year',4:"Emission Amount"})
x_INA = GenerateMonthlyEmissionData(t1_INA)

# ---------------------- RETURNING TO ORIGINAL FORMAT -------------------------#

x_IND.rename(columns={'Emission Amount': 'Indonesia'}, inplace=True)
x_IND.drop(columns='Month', inplace=True)
trx_IND = x_IND.transpose()

x_ARG.rename(columns={'Emission Amount': 'Argentina'}, inplace=True)
x_ARG.drop(columns='Month', inplace=True)
trx_ARG = x_ARG.transpose()

x_ENG.rename(columns={'Emission Amount': 'United Kingdom'}, inplace=True)
x_ENG.drop(columns='Month', inplace=True)
trx_ENG = x_ENG.transpose()

x_BRZ.rename(columns={'Emission Amount': 'Brazil'}, inplace=True)
x_BRZ.drop(columns='Month', inplace=True)
trx_BRZ = x_BRZ.transpose()

x_CAN.rename(columns={'Emission Amount': 'Canada'}, inplace=True)
x_CAN.drop(columns='Month', inplace=True)
trx_CAN = x_CAN.transpose()

x_USA.rename(columns={'Emission Amount': 'United States'}, inplace=True)
x_USA.drop(columns='Month', inplace=True)
trx_USA = x_USA.transpose()

x_ITA.rename(columns={'Emission Amount': 'Italy'}, inplace=True)
x_ITA.drop(columns='Month', inplace=True)
trx_ITA = x_ITA.transpose()

x_FRC.rename(columns={'Emission Amount': 'France'}, inplace=True)
x_FRC.drop(columns='Month', inplace=True)
trx_FRC = x_FRC.transpose()

x_GRM.rename(columns={'Emission Amount': 'Germany'}, inplace=True)
x_GRM.drop(columns='Month', inplace=True)
trx_GRM = x_GRM.transpose()

x_SAF.rename(columns={'Emission Amount': 'South Africa'}, inplace=True)
x_SAF.drop(columns='Month', inplace=True)
trx_SAF = x_SAF.transpose()

x_JPN.rename(columns={'Emission Amount': 'Japan'}, inplace=True)
x_JPN.drop(columns='Month', inplace=True)
trx_JPN = x_JPN.transpose()

x_MXC.rename(columns={'Emission Amount': 'Mexico'}, inplace=True)
x_MXC.drop(columns='Month', inplace=True)
trx_MXC = x_MXC.transpose()

x_ARB.rename(columns={'Emission Amount': 'Saudi Arabia'}, inplace=True)
x_ARB.drop(columns='Month', inplace=True)
trx_ARB = x_ARB.transpose()

x_TUR.rename(columns={'Emission Amount': 'Turkey'}, inplace=True)
x_TUR.drop(columns='Month', inplace=True)
trx_TUR = x_TUR.transpose()

x_AUS.rename(columns={'Emission Amount': 'Australia'}, inplace=True)
x_AUS.drop(columns='Month', inplace=True)
trx_AUS = x_AUS.transpose()

x_CHN.rename(columns={'Emission Amount': 'China'}, inplace=True)
x_CHN.drop(columns='Month', inplace=True)
trx_CHN = x_CHN.transpose()

x_INA.rename(columns={'Emission Amount': 'India'}, inplace=True)
x_INA.drop(columns='Month', inplace=True)
trx_INA = x_INA.transpose()

# ---------------------- Data 1 -------------------------#
Indonesia = x_IND.copy()
Indonesia = ss.fit_transform(Indonesia)
Indonesia = pd.DataFrame(Indonesia)
Indonesia = Indonesia.rename(columns={Indonesia.columns[0]: 'Indonesia'})
Indonesia = Indonesia.reset_index()
Indonesia = Indonesia.rename(columns={Indonesia.columns[0]: 'Date'})

base_date = pd.Timestamp('1990-12-01')
Indonesia['Date'] = base_date + Indonesia.index.to_series().apply(lambda x: relativedelta(months=x))
Indonesia.set_index('Date',inplace = True)

max_shift = 10
Country_Name = 'Indonesia'

for shift_value in range(1, max_shift + 1):
    new_column_name = f'Lag_{shift_value}'
    Indonesia[new_column_name] = Indonesia[Country_Name].shift(shift_value)
Indonesia.dropna(inplace=True)
# ---------------------- Data 2 -------------------------#
Argentina = x_ARG.copy()
Argentina = ss.fit_transform(Argentina)
Argentina = pd.DataFrame(Argentina)
Argentina = Argentina.rename(columns={Argentina.columns[0]: 'Argentina'})
Argentina = Argentina.reset_index()
Argentina = Argentina.rename(columns={Argentina.columns[0]: 'Date'})

base_date = pd.Timestamp('1990-12-01')
Argentina['Date'] = base_date + Argentina.index.to_series().apply(lambda x: relativedelta(months=x))
Argentina.set_index('Date', inplace=True)

max_shift = 10
Country_Name = 'Argentina'

for shift_value in range(1, max_shift + 1):
    new_column_name = f'Lag_{shift_value}'
    Argentina[new_column_name] = Argentina[Country_Name].shift(shift_value)
Argentina.dropna(inplace=True)
# ---------------------- Data 3 -------------------------#
United_Kingdom = x_ENG.copy()
United_Kingdom = ss.fit_transform(United_Kingdom)
United_Kingdom = pd.DataFrame(United_Kingdom)
United_Kingdom = United_Kingdom.rename(columns={United_Kingdom.columns[0]: 'United Kingdom'})
United_Kingdom = United_Kingdom.reset_index()
United_Kingdom = United_Kingdom.rename(columns={United_Kingdom.columns[0]: 'Date'})

base_date = pd.Timestamp('1990-12-01')
United_Kingdom['Date'] = base_date + United_Kingdom.index.to_series().apply(lambda x: relativedelta(months=x))
United_Kingdom.set_index('Date', inplace=True)

max_shift = 10
Country_Name = 'United Kingdom'

for shift_value in range(1, max_shift + 1):
    new_column_name = f'Lag_{shift_value}'
    United_Kingdom[new_column_name] = United_Kingdom[Country_Name].shift(shift_value)
United_Kingdom.dropna(inplace=True)
# ---------------------- Data 4 -------------------------#
Brazil = x_BRZ.copy()
Brazil = ss.fit_transform(Brazil)
Brazil = pd.DataFrame(Brazil)
Brazil = Brazil.rename(columns={Brazil.columns[0]: 'United Kingdom'})
Brazil = Brazil.reset_index()
Brazil = Brazil.rename(columns={Brazil.columns[0]: 'Date'})

base_date = pd.Timestamp('1990-12-01')
Brazil['Date'] = base_date + Brazil.index.to_series().apply(lambda x: relativedelta(months=x))
Brazil.set_index('Date', inplace=True)

max_shift = 10
Country_Name = 'United Kingdom'

for shift_value in range(1, max_shift + 1):
    new_column_name = f'Lag_{shift_value}'
    Brazil[new_column_name] = Brazil[Country_Name].shift(shift_value)
Brazil.dropna(inplace=True)
# ---------------------- Data 5 -------------------------#
Canada = x_CAN.copy()
Canada = ss.fit_transform(Canada)
Canada = pd.DataFrame(Canada)
Canada = Canada.rename(columns={Canada.columns[0]: 'Canada'})
Canada = Canada.reset_index()
Canada = Canada.rename(columns={Canada.columns[0]: 'Date'})

base_date = pd.Timestamp('1990-12-01')
Canada['Date'] = base_date + Canada.index.to_series().apply(lambda x: relativedelta(months=x))
Canada.set_index('Date', inplace=True)

max_shift = 10
Country_Name = 'Canada'

for shift_value in range(1, max_shift + 1):
    new_column_name = f'Lag_{shift_value}'
    Canada[new_column_name] = Canada[Country_Name].shift(shift_value)
Canada.dropna(inplace=True)
# ---------------------- Data 6 -------------------------#
Italy = x_ITA.copy()
Italy = ss.fit_transform(Italy)
Italy = pd.DataFrame(Italy)
Italy = Italy.rename(columns={Italy.columns[0]: 'Italy'})
Italy = Italy.reset_index()
Italy = Italy.rename(columns={Italy.columns[0]: 'Date'})

base_date = pd.Timestamp('1990-12-01')
Italy['Date'] = base_date + Italy.index.to_series().apply(lambda x: relativedelta(months=x))
Italy.set_index('Date', inplace=True)

max_shift = 10
Country_Name = 'Italy'

for shift_value in range(1, max_shift + 1):
    new_column_name = f'Lag_{shift_value}'
    Italy[new_column_name] = Italy[Country_Name].shift(shift_value)
Italy.dropna(inplace=True)
# ---------------------- Data 7 -------------------------#
United_States = x_USA.copy()
United_States = ss.fit_transform(United_States)
United_States = pd.DataFrame(United_States)
United_States = United_States.rename(columns={United_States.columns[0]: 'United States'})
United_States = United_States.reset_index()
United_States = United_States.rename(columns={United_States.columns[0]: 'Date'})

base_date = pd.Timestamp('1990-12-01')
United_States['Date'] = base_date + United_States.index.to_series().apply(lambda x: relativedelta(months=x))
United_States.set_index('Date', inplace=True)

max_shift = 10
Country_Name = 'United States'

for shift_value in range(1, max_shift + 1):
    new_column_name = f'Lag_{shift_value}'
    United_States[new_column_name] = United_States[Country_Name].shift(shift_value)
United_States.dropna(inplace=True)
# ---------------------- Data 8 -------------------------#
France = x_FRC.copy()
France = ss.fit_transform(France)
France = pd.DataFrame(France)
France = France.rename(columns={France.columns[0]: 'France'})
France = France.reset_index()
France = France.rename(columns={France.columns[0]: 'Date'})

base_date = pd.Timestamp('1990-12-01')
France['Date'] = base_date + France.index.to_series().apply(lambda x: relativedelta(months=x))
France.set_index('Date', inplace=True)

max_shift = 10
Country_Name = 'France'

for shift_value in range(1, max_shift + 1):
    new_column_name = f'Lag_{shift_value}'
    France[new_column_name] = France[Country_Name].shift(shift_value)
France.dropna(inplace=True)
# ---------------------- Data 9 -------------------------#
Germany = x_GRM.copy()
Germany = ss.fit_transform(Germany)
Germany = pd.DataFrame(Germany)
Germany = Germany.rename(columns={Germany.columns[0]: 'Germany'})
Germany = Germany.reset_index()
Germany = Germany.rename(columns={Germany.columns[0]: 'Date'})

base_date = pd.Timestamp('1990-12-01')
Germany['Date'] = base_date + Germany.index.to_series().apply(lambda x: relativedelta(months=x))
Germany.set_index('Date', inplace=True)

max_shift = 10
Country_Name = 'Germany'

for shift_value in range(1, max_shift + 1):
    new_column_name = f'Lag_{shift_value}'
    Germany[new_column_name] = Germany[Country_Name].shift(shift_value)
Germany.dropna(inplace=True)
# ---------------------- Data 10 -------------------------#
South_Africa = x_SAF.copy()
South_Africa = ss.fit_transform(South_Africa)
South_Africa = pd.DataFrame(South_Africa)
South_Africa = South_Africa.rename(columns={South_Africa.columns[0]: 'South Africa'})
South_Africa = South_Africa.reset_index()
South_Africa = South_Africa.rename(columns={South_Africa.columns[0]: 'Date'})

base_date = pd.Timestamp('1990-12-01')
South_Africa['Date'] = base_date + South_Africa.index.to_series().apply(lambda x: relativedelta(months=x))
South_Africa.set_index('Date', inplace=True)

max_shift = 10
Country_Name = 'South Africa'

for shift_value in range(1, max_shift + 1):
    new_column_name = f'Lag_{shift_value}'
    South_Africa[new_column_name] = South_Africa[Country_Name].shift(shift_value)
South_Africa.dropna(inplace=True)
# ---------------------- Data 11 -------------------------#
Japan = x_JPN.copy()
Japan = ss.fit_transform(Japan)
Japan = pd.DataFrame(Japan)
Japan = Japan.rename(columns={Japan.columns[0]: 'Japan'})
Japan = Japan.reset_index()
Japan = Japan.rename(columns={Japan.columns[0]: 'Date'})

base_date = pd.Timestamp('1990-12-01')
Japan['Date'] = base_date + Japan.index.to_series().apply(lambda x: relativedelta(months=x))
Japan.set_index('Date', inplace=True)

max_shift = 10
Country_Name = 'Japan'

for shift_value in range(1, max_shift + 1):
    new_column_name = f'Lag_{shift_value}'
    Japan[new_column_name] = Japan[Country_Name].shift(shift_value)
Japan.dropna(inplace=True)
# ---------------------- Data 12 -------------------------#
Mexico = x_MXC.copy()
Mexico = ss.fit_transform(Mexico)
Mexico = pd.DataFrame(Mexico)
Mexico = Mexico.rename(columns={Mexico.columns[0]: 'Mexico'})
Mexico = Mexico.reset_index()
Mexico = Mexico.rename(columns={Mexico.columns[0]: 'Date'})

base_date = pd.Timestamp('1990-12-01')
Mexico['Date'] = base_date + Mexico.index.to_series().apply(lambda x: relativedelta(months=x))
Mexico.set_index('Date', inplace=True)

max_shift = 10
Country_Name = 'Mexico'

for shift_value in range(1, max_shift + 1):
    new_column_name = f'Lag_{shift_value}'
    Mexico[new_column_name] = Mexico[Country_Name].shift(shift_value)
Mexico.dropna(inplace=True)
# ---------------------- Data 13 -------------------------#
Saudi_Arabia = x_ARB.copy()
Saudi_Arabia = ss.fit_transform(Saudi_Arabia)
Saudi_Arabia = pd.DataFrame(Saudi_Arabia)
Saudi_Arabia = Saudi_Arabia.rename(columns={Saudi_Arabia.columns[0]: 'Saudi Arabia'})
Saudi_Arabia = Saudi_Arabia.reset_index()
Saudi_Arabia = Saudi_Arabia.rename(columns={Saudi_Arabia.columns[0]: 'Date'})

base_date = pd.Timestamp('1990-12-01')
Saudi_Arabia['Date'] = base_date + Saudi_Arabia.index.to_series().apply(lambda x: relativedelta(months=x))
Saudi_Arabia.set_index('Date', inplace=True)

max_shift = 10
Country_Name = 'Saudi Arabia'

for shift_value in range(1, max_shift + 1):
    new_column_name = f'Lag_{shift_value}'
    Saudi_Arabia[new_column_name] = Saudi_Arabia[Country_Name].shift(shift_value)
Saudi_Arabia.dropna(inplace=True)
# ---------------------- Data 14 -------------------------#
Turkey = x_TUR.copy()
Turkey = ss.fit_transform(Turkey)
Turkey = pd.DataFrame(Turkey)
Turkey = Turkey.rename(columns={Turkey.columns[0]: 'Turkey'})
Turkey = Turkey.reset_index()
Turkey = Turkey.rename(columns={Turkey.columns[0]: 'Date'})

base_date = pd.Timestamp('1990-12-01')
Turkey['Date'] = base_date + Turkey.index.to_series().apply(lambda x: relativedelta(months=x))
Turkey.set_index('Date', inplace=True)

max_shift = 10
Country_Name = 'Turkey'

for shift_value in range(1, max_shift + 1):
    new_column_name = f'Lag_{shift_value}'
    Turkey[new_column_name] = Turkey[Country_Name].shift(shift_value)
Turkey.dropna(inplace=True)
# ---------------------- Data 15 -------------------------#
Australia = x_AUS.copy()
Australia = ss.fit_transform(Australia)
Australia = pd.DataFrame(Australia)
Australia = Australia.rename(columns={Australia.columns[0]: 'Australia'})
Australia = Australia.reset_index()
Australia = Australia.rename(columns={Australia.columns[0]: 'Date'})

base_date = pd.Timestamp('1990-12-01')
Australia['Date'] = base_date + Australia.index.to_series().apply(lambda x: relativedelta(months=x))
Australia.set_index('Date', inplace=True)

max_shift = 10
Country_Name = 'Australia'

for shift_value in range(1, max_shift + 1):
    new_column_name = f'Lag_{shift_value}'
    Australia[new_column_name] = Australia[Country_Name].shift(shift_value)
Australia.dropna(inplace=True)
# ---------------------- Data 16 -------------------------#
China = x_CHN.copy()
China = ss.fit_transform(China)
China = pd.DataFrame(China)
China = China.rename(columns={China.columns[0]: 'China'})
China = China.reset_index()
China = China.rename(columns={China.columns[0]: 'Date'})

base_date = pd.Timestamp('1990-12-01')
China['Date'] = base_date + China.index.to_series().apply(lambda x: relativedelta(months=x))
China.set_index('Date', inplace=True)

max_shift = 10
Country_Name = 'China'

for shift_value in range(1, max_shift + 1):
    new_column_name = f'Lag_{shift_value}'
    China[new_column_name] = China[Country_Name].shift(shift_value)
China.dropna(inplace=True)
# ---------------------- Data 17 -------------------------#
India = x_INA.copy()
India = ss.fit_transform(India)
India = pd.DataFrame(India)
India = India.rename(columns={India.columns[0]: 'India'})
India = India.reset_index()
India = India.rename(columns={India.columns[0]: 'Date'})

base_date = pd.Timestamp('1990-12-01')
India['Date'] = base_date + India.index.to_series().apply(lambda x: relativedelta(months=x))
India.set_index('Date', inplace=True)

max_shift = 10
Country_Name = 'India'

for shift_value in range(1, max_shift + 1):
    new_column_name = f'Lag_{shift_value}'
    India[new_column_name] = India[Country_Name].shift(shift_value)
India.dropna(inplace=True)
# ------------------------------------------------------------#


# --------------------Loading Models--------------------------#

model1 = joblib.load('model_Indonesia.pkl')
model2 = joblib.load('model_Argentina.pkl')
model3 = joblib.load('model_United_Kingdom.pkl')
model4 = joblib.load('model_Brazil.pkl')
model5 = joblib.load('model_Canada.pkl')
model6 = joblib.load('model_United_States.pkl')
model7 = joblib.load('model_Italy.pkl')
model8 = joblib.load('model_France.pkl')
model9 = joblib.load('model_Germany.pkl')
model10 = joblib.load('model_South_Africa.pkl')
model11 = joblib.load('model_Japan.pkl')
model12 = joblib.load('model_Mexico.pkl')
model13 = joblib.load('model_Saudi_Arabia.pkl')
model14 = joblib.load('model_Turkey.pkl')
model15 = joblib.load('model_Australia.pkl')
model16 = joblib.load('model_China.pkl')
model17 = joblib.load('model_India.pkl')
# ------------------------------------------------------------#
def make_predictions_and_plot(n_future_steps,model_selection):
    
    n_steps = n_future_steps
    
    if model_selection == 'Indonesia':
        model = model1
        input_data = Indonesia.iloc[-1, 1:].values.tolist()
        
        future_predictions = []
        for _ in range(n_future_steps):
            input_array = np.array(input_data).reshape(1, -1)

            next_value = model.predict(input_array)[0]

            future_predictions.append(next_value)

            input_data.append(next_value)
            input_data.pop(0)
        
        future_predictions = pd.DataFrame(future_predictions)
        future_predictions = future_predictions.reset_index()
        future_predictions = future_predictions.rename(columns={future_predictions.columns[0]: 'Date'})
        future_predictions = future_predictions.rename(columns={future_predictions.columns[1]: 'future_predictions'})

        base_date = pd.Timestamp('2018-1-01')
        future_predictions['Date'] = base_date + future_predictions.index.to_series().apply(lambda x: relativedelta(months=x))
        future_predictions.set_index('Date', inplace=True)

        future_predictions['future_predictions'] = ss.inverse_transform(future_predictions[['future_predictions']])
        Indonesia['Indonesia'] = ss.inverse_transform(Indonesia[['Indonesia']])
        
        fig = plt.figure(figsize=(10, 5))
        plt.plot(Indonesia.index, Indonesia['Indonesia'], label='Actual')
        plt.plot(future_predictions.index, future_predictions, label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig)
        
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(future_predictions.index, future_predictions['future_predictions'], label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression (Predictions Only)")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig1)
        
        plt.show()
        
        return fig, fig1

        
    elif model_selection == 'Argentina':
        model = model2
        input_data = Argentina.iloc[-1, 1:].values.tolist()
        
        future_predictions = []
        for _ in range(n_future_steps):
            input_array = np.array(input_data).reshape(1, -1)

            next_value = model.predict(input_array)[0]

            future_predictions.append(next_value)

            input_data.append(next_value)
            input_data.pop(0)

        future_predictions = pd.DataFrame(future_predictions)
        future_predictions = future_predictions.reset_index()
        future_predictions = future_predictions.rename(columns={future_predictions.columns[0]: 'Date'})
        future_predictions = future_predictions.rename(columns={future_predictions.columns[1]: 'future_predictions'})

        base_date = pd.Timestamp('2018-1-01')
        future_predictions['Date'] = base_date + future_predictions.index.to_series().apply(lambda x: relativedelta(months=x))
        future_predictions.set_index('Date', inplace=True)

        future_predictions['future_predictions'] = ss.inverse_transform(future_predictions[['future_predictions']])
        Argentina['Argentina'] = ss.inverse_transform(Argentina[['Argentina']])
        
        fig = plt.figure(figsize=(10, 5))
        plt.plot(Argentina.index, Argentina['Argentina'], label='Actual')
        plt.plot(future_predictions.index, future_predictions, label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig)
        
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(future_predictions.index, future_predictions['future_predictions'], label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression (Predictions Only)")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig1)
        
        return fig, fig1


    elif model_selection == 'United Kingdom':
        model = model3
        input_data = United_Kingdom.iloc[-1, 1:].values.tolist()
        
        future_predictions = []
        for _ in range(n_future_steps):
            input_array = np.array(input_data).reshape(1, -1)

            next_value = model.predict(input_array)[0]

            future_predictions.append(next_value)

            input_data.append(next_value)
            input_data.pop(0)
            
        future_predictions = pd.DataFrame(future_predictions)
        future_predictions = future_predictions.reset_index()
        future_predictions = future_predictions.rename(columns={future_predictions.columns[0]: 'Date'})
        future_predictions = future_predictions.rename(columns={future_predictions.columns[1]: 'future_predictions'})

        base_date = pd.Timestamp('2018-1-01')
        future_predictions['Date'] = base_date + future_predictions.index.to_series().apply(lambda x: relativedelta(months=x))
        future_predictions.set_index('Date', inplace=True)

        future_predictions['future_predictions'] = ss.inverse_transform(future_predictions[['future_predictions']])
        United_Kingdom['United Kingdom'] = ss.inverse_transform(United_Kingdom[['United Kingdom']])
        
        fig = plt.figure(figsize=(10, 5))
        plt.plot(United_Kingdom.index, United_Kingdom['United Kingdom'], label='Actual')
        plt.plot(future_predictions.index, future_predictions, label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig)
        
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(future_predictions.index, future_predictions['future_predictions'], label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression (Predictions Only)")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig1)
        
        return fig, fig1

        
    elif model_selection == 'Brazil':
        model = model4
        input_data = Brazil.iloc[-1, 1:].values.tolist()
        
        future_predictions = []
        for _ in range(n_future_steps):
            input_array = np.array(input_data).reshape(1, -1)

            next_value = model.predict(input_array)[0]

            future_predictions.append(next_value)

            input_data.append(next_value)
            input_data.pop(0)
        
        future_predictions = pd.DataFrame(future_predictions)
        future_predictions = future_predictions.reset_index()
        future_predictions = future_predictions.rename(columns={future_predictions.columns[0]: 'Date'})
        future_predictions = future_predictions.rename(columns={future_predictions.columns[1]: 'future_predictions'})

        base_date = pd.Timestamp('2018-1-01')
        future_predictions['Date'] = base_date + future_predictions.index.to_series().apply(lambda x: relativedelta(months=x))
        future_predictions.set_index('Date', inplace=True)

        future_predictions['future_predictions'] = ss.inverse_transform(future_predictions[['future_predictions']])
        Brazil['Brazil'] = ss.inverse_transform(Brazil[['Brazil']])
        
        fig = plt.figure(figsize=(10, 5))
        plt.plot(Brazil.index, Brazil['Brazil'], label='Actual')
        plt.plot(future_predictions.index, future_predictions, label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig)
        
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(future_predictions.index, future_predictions['future_predictions'], label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression (Predictions Only)")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig1)
        
        return fig, fig1

        
    elif model_selection == 'Canada':
        model = model5
        input_data = Canada.iloc[-1, 1:].values.tolist()
        
        future_predictions = []
        for _ in range(n_future_steps):
            input_array = np.array(input_data).reshape(1, -1)

            next_value = model.predict(input_array)[0]

            future_predictions.append(next_value)

            input_data.append(next_value)
            input_data.pop(0)
            
        future_predictions = pd.DataFrame(future_predictions)
        future_predictions = future_predictions.reset_index()
        future_predictions = future_predictions.rename(columns={future_predictions.columns[0]: 'Date'})
        future_predictions = future_predictions.rename(columns={future_predictions.columns[1]: 'future_predictions'})

        base_date = pd.Timestamp('2018-1-01')
        future_predictions['Date'] = base_date + future_predictions.index.to_series().apply(lambda x: relativedelta(months=x))
        future_predictions.set_index('Date', inplace=True)

        future_predictions['future_predictions'] = ss.inverse_transform(future_predictions[['future_predictions']])
        Canada['Canada'] = ss.inverse_transform(Canada[['Canada']])
        
        fig = plt.figure(figsize=(10, 5))
        plt.plot(Canada.index, Canada['Canada'], label='Actual')
        plt.plot(future_predictions.index, future_predictions, label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig)
        
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(future_predictions.index, future_predictions['future_predictions'], label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression (Predictions Only)")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig1)
        
        return fig, fig1

    elif model_selection == 'United States':
        model = model6
        input_data = United_States.iloc[-1, 1:].values.tolist()
        
        future_predictions = []
        for _ in range(n_future_steps):
            input_array = np.array(input_data).reshape(1, -1)

            next_value = model.predict(input_array)[0]

            future_predictions.append(next_value)

            input_data.append(next_value)
            input_data.pop(0)
            
        future_predictions = pd.DataFrame(future_predictions)
        future_predictions = future_predictions.reset_index()
        future_predictions = future_predictions.rename(columns={future_predictions.columns[0]: 'Date'})
        future_predictions = future_predictions.rename(columns={future_predictions.columns[1]: 'future_predictions'})

        base_date = pd.Timestamp('2018-1-01')
        future_predictions['Date'] = base_date + future_predictions.index.to_series().apply(lambda x: relativedelta(months=x))
        future_predictions.set_index('Date', inplace=True)

        future_predictions['future_predictions'] = ss.inverse_transform(future_predictions[['future_predictions']])
        United_States['United States'] = ss.inverse_transform(United_States[['United States']])
        
        fig = plt.figure(figsize=(10, 5))
        plt.plot(United_States.index, United_States['United States'], label='Actual')
        plt.plot(future_predictions.index, future_predictions, label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig)
        
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(future_predictions.index, future_predictions['future_predictions'], label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression (Predictions Only)")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig1)
        
        return fig, fig1

        
    elif model_selection == 'Italy':
        model = model7
        input_data = Italy.iloc[-1, 1:].values.tolist()
        
        future_predictions = []
        for _ in range(n_future_steps):
            input_array = np.array(input_data).reshape(1, -1)

            next_value = model.predict(input_array)[0]

            future_predictions.append(next_value)

            input_data.append(next_value)
            input_data.pop(0)
            
        future_predictions = pd.DataFrame(future_predictions)
        future_predictions = future_predictions.reset_index()
        future_predictions = future_predictions.rename(columns={future_predictions.columns[0]: 'Date'})
        future_predictions = future_predictions.rename(columns={future_predictions.columns[1]: 'future_predictions'})

        base_date = pd.Timestamp('2018-1-01')
        future_predictions['Date'] = base_date + future_predictions.index.to_series().apply(lambda x: relativedelta(months=x))
        future_predictions.set_index('Date', inplace=True)

        future_predictions['future_predictions'] = ss.inverse_transform(future_predictions[['future_predictions']])
        Italy['Italy'] = ss.inverse_transform(Italy[['Italy']])
        
        fig = plt.figure(figsize=(10, 5))
        plt.plot(Italy.index, Italy['Italy'], label='Actual')
        plt.plot(future_predictions.index, future_predictions, label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig)
        
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(future_predictions.index, future_predictions['future_predictions'], label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression (Predictions Only)")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig1)
        
        return fig, fig1

        
    elif model_selection == 'France':
        model = model8
        input_data = France.iloc[-1, 1:].values.tolist()
        
        future_predictions = []
        for _ in range(n_future_steps):
            input_array = np.array(input_data).reshape(1, -1)

            next_value = model.predict(input_array)[0]

            future_predictions.append(next_value)

            input_data.append(next_value)
            input_data.pop(0)
            
        future_predictions = pd.DataFrame(future_predictions)
        future_predictions = future_predictions.reset_index()
        future_predictions = future_predictions.rename(columns={future_predictions.columns[0]: 'Date'})
        future_predictions = future_predictions.rename(columns={future_predictions.columns[1]: 'future_predictions'})

        base_date = pd.Timestamp('2018-1-01')
        future_predictions['Date'] = base_date + future_predictions.index.to_series().apply(lambda x: relativedelta(months=x))
        future_predictions.set_index('Date', inplace=True)

        future_predictions['future_predictions'] = ss.inverse_transform(future_predictions[['future_predictions']])
        France['France'] = ss.inverse_transform(France[['France']])
        
        fig = plt.figure(figsize=(10, 5))
        plt.plot(France.index, France['France'], label='Actual')
        plt.plot(future_predictions.index, future_predictions, label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig)
        
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(future_predictions.index, future_predictions['future_predictions'], label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression (Predictions Only)")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig1)
        
        return fig, fig1

        
    elif model_selection == 'Germany':
        model = model9
        input_data = Germany.iloc[-1, 1:].values.tolist()
        
        future_predictions = []
        for _ in range(n_future_steps):
            input_array = np.array(input_data).reshape(1, -1)

            next_value = model.predict(input_array)[0]

            future_predictions.append(next_value)

            input_data.append(next_value)
            input_data.pop(0)
        
        future_predictions = pd.DataFrame(future_predictions)
        future_predictions = future_predictions.reset_index()
        future_predictions = future_predictions.rename(columns={future_predictions.columns[0]: 'Date'})
        future_predictions = future_predictions.rename(columns={future_predictions.columns[1]: 'future_predictions'})

        base_date = pd.Timestamp('2018-1-01')
        future_predictions['Date'] = base_date + future_predictions.index.to_series().apply(lambda x: relativedelta(months=x))
        future_predictions.set_index('Date', inplace=True)

        future_predictions['future_predictions'] = ss.inverse_transform(future_predictions[['future_predictions']])
        Germany['Germany'] = ss.inverse_transform(Germany[['Germany']])
        
        fig = plt.figure(figsize=(10, 5))
        plt.plot(Germany.index, Germany['Germany'], label='Actual')
        plt.plot(future_predictions.index, future_predictions, label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig)
        
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(future_predictions.index, future_predictions['future_predictions'], label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression (Predictions Only)")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig1)
        
        return fig, fig1
        
    elif model_selection == 'South Africa':
        model = model10
        input_data = South_Africa.iloc[-1, 1:].values.tolist()
        
        future_predictions = []
        for _ in range(n_future_steps):
            input_array = np.array(input_data).reshape(1, -1)

            next_value = model.predict(input_array)[0]

            future_predictions.append(next_value)

            input_data.append(next_value)
            input_data.pop(0)
        
        future_predictions = pd.DataFrame(future_predictions)
        future_predictions = future_predictions.reset_index()
        future_predictions = future_predictions.rename(columns={future_predictions.columns[0]: 'Date'})
        future_predictions = future_predictions.rename(columns={future_predictions.columns[1]: 'future_predictions'})

        base_date = pd.Timestamp('2018-1-01')
        future_predictions['Date'] = base_date + future_predictions.index.to_series().apply(lambda x: relativedelta(months=x))
        future_predictions.set_index('Date', inplace=True)

        future_predictions['future_predictions'] = ss.inverse_transform(future_predictions[['future_predictions']])
        South_Africa['South Africa'] = ss.inverse_transform(South_Africa[['South Africa']])
        
        fig = plt.figure(figsize=(10, 5))
        plt.plot(South_Africa.index, South_Africa['South Africa'], label='Actual')
        plt.plot(future_predictions.index, future_predictions, label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig)
        
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(future_predictions.index, future_predictions['future_predictions'], label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression (Predictions Only)")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig1)
        
        return fig, fig1


        
    elif model_selection == 'Japan':
        model = model11
        input_data = Japan.iloc[-1, 1:].values.tolist()
        
        future_predictions = []
        for _ in range(n_future_steps):
            input_array = np.array(input_data).reshape(1, -1)

            next_value = model.predict(input_array)[0]

            future_predictions.append(next_value)

            input_data.append(next_value)
            input_data.pop(0)
        
        future_predictions = pd.DataFrame(future_predictions)
        future_predictions = future_predictions.reset_index()
        future_predictions = future_predictions.rename(columns={future_predictions.columns[0]: 'Date'})
        future_predictions = future_predictions.rename(columns={future_predictions.columns[1]: 'future_predictions'})

        base_date = pd.Timestamp('2018-1-01')
        future_predictions['Date'] = base_date + future_predictions.index.to_series().apply(lambda x: relativedelta(months=x))
        future_predictions.set_index('Date', inplace=True)

        future_predictions['future_predictions'] = ss.inverse_transform(future_predictions[['future_predictions']])
        Japan['Japan'] = ss.inverse_transform(Japan[['Japan']])
        
        fig = plt.figure(figsize=(10, 5))
        plt.plot(Japan.index, Japan['Japan'], label='Actual')
        plt.plot(future_predictions.index, future_predictions, label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig)
        
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(future_predictions.index, future_predictions['future_predictions'], label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression (Predictions Only)")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig1)
        
        return fig, fig1



    elif model_selection == 'Mexico':
        model = model12
        input_data = Mexico.iloc[-1, 1:].values.tolist()
        
        future_predictions = []
        for _ in range(n_future_steps):
            input_array = np.array(input_data).reshape(1, -1)

            next_value = model.predict(input_array)[0]

            future_predictions.append(next_value)

            input_data.append(next_value)
            input_data.pop(0)
            
        future_predictions = pd.DataFrame(future_predictions)
        future_predictions = future_predictions.reset_index()
        future_predictions = future_predictions.rename(columns={future_predictions.columns[0]: 'Date'})
        future_predictions = future_predictions.rename(columns={future_predictions.columns[1]: 'future_predictions'})

        base_date = pd.Timestamp('2018-1-01')
        future_predictions['Date'] = base_date + future_predictions.index.to_series().apply(lambda x: relativedelta(months=x))
        future_predictions.set_index('Date', inplace=True)

        future_predictions['future_predictions'] = ss.inverse_transform(future_predictions[['future_predictions']])
        Mexico['Mexico'] = ss.inverse_transform(Mexico[['Mexico']])
        
        fig = plt.figure(figsize=(10, 5))
        plt.plot(Mexico.index, Mexico['Mexico'], label='Actual')
        plt.plot(future_predictions.index, future_predictions, label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig)
        
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(future_predictions.index, future_predictions['future_predictions'], label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression (Predictions Only)")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig1)
        
        return fig, fig1


        
    elif model_selection == 'Saudi Arabia':
        model = model13
        input_data = Saudi_Arabia.iloc[-1, 1:].values.tolist()
        
        future_predictions = []
        for _ in range(n_future_steps):
            input_array = np.array(input_data).reshape(1, -1)

            next_value = model.predict(input_array)[0]

            future_predictions.append(next_value)

            input_data.append(next_value)
            input_data.pop(0)
            
        future_predictions = pd.DataFrame(future_predictions)
        future_predictions = future_predictions.reset_index()
        future_predictions = future_predictions.rename(columns={future_predictions.columns[0]: 'Date'})
        future_predictions = future_predictions.rename(columns={future_predictions.columns[1]: 'future_predictions'})

        base_date = pd.Timestamp('2018-1-01')
        future_predictions['Date'] = base_date + future_predictions.index.to_series().apply(lambda x: relativedelta(months=x))
        future_predictions.set_index('Date', inplace=True)

        future_predictions['future_predictions'] = ss.inverse_transform(future_predictions[['future_predictions']])
        Saudi_Arabia['Saudi Arabia'] = ss.inverse_transform(Saudi_Arabia[['Saudi Arabia']])
        
        fig = plt.figure(figsize=(10, 5))
        plt.plot(Saudi_Arabia.index, Saudi_Arabia['Saudi Arabia'], label='Actual')
        plt.plot(future_predictions.index, future_predictions, label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig)
        
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(future_predictions.index, future_predictions['future_predictions'], label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression (Predictions Only)")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig1)
        
        return fig, fig1


        
    elif model_selection == 'Turkey':
        model = model14
        input_data = Turkey.iloc[-1, 1:].values.tolist()
        
        future_predictions = []
        for _ in range(n_future_steps):
            input_array = np.array(input_data).reshape(1, -1)

            next_value = model.predict(input_array)[0]

            future_predictions.append(next_value)

            input_data.append(next_value)
            input_data.pop(0)
        
        future_predictions = pd.DataFrame(future_predictions)
        future_predictions = future_predictions.reset_index()
        future_predictions = future_predictions.rename(columns={future_predictions.columns[0]: 'Date'})
        future_predictions = future_predictions.rename(columns={future_predictions.columns[1]: 'future_predictions'})

        base_date = pd.Timestamp('2018-1-01')
        future_predictions['Date'] = base_date + future_predictions.index.to_series().apply(lambda x: relativedelta(months=x))
        future_predictions.set_index('Date', inplace=True)

        future_predictions['future_predictions'] = ss.inverse_transform(future_predictions[['future_predictions']])
        Turkey['Turkey'] = ss.inverse_transform(Turkey[['Turkey']])
        
        fig = plt.figure(figsize=(10, 5))
        plt.plot(Turkey.index, Turkey['Turkey'], label='Actual')
        plt.plot(future_predictions.index, future_predictions, label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig)
        
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(future_predictions.index, future_predictions['future_predictions'], label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression (Predictions Only)")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig1)
        
        return fig, fig1


        
    elif model_selection == 'Australia':
        model = model15
        input_data = Australia.iloc[-1, 1:].values.tolist()
        
        future_predictions = []
        for _ in range(n_future_steps):
            input_array = np.array(input_data).reshape(1, -1)

            next_value = model.predict(input_array)[0]

            future_predictions.append(next_value)

            input_data.append(next_value)
            input_data.pop(0)
            
        future_predictions = pd.DataFrame(future_predictions)
        future_predictions = future_predictions.reset_index()
        future_predictions = future_predictions.rename(columns={future_predictions.columns[0]: 'Date'})
        future_predictions = future_predictions.rename(columns={future_predictions.columns[1]: 'future_predictions'})

        base_date = pd.Timestamp('2018-1-01')
        future_predictions['Date'] = base_date + future_predictions.index.to_series().apply(lambda x: relativedelta(months=x))
        future_predictions.set_index('Date', inplace=True)

        future_predictions['future_predictions'] = ss.inverse_transform(future_predictions[['future_predictions']])
        Australia['Australia'] = ss.inverse_transform(Australia[['Australia']])
        
        fig = plt.figure(figsize=(10, 5))
        plt.plot(Australia.index, Australia['Australia'], label='Actual')
        plt.plot(future_predictions.index, future_predictions, label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig)
        
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(future_predictions.index, future_predictions['future_predictions'], label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression (Predictions Only)")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig1)
        
        return fig, fig1

        
    elif model_selection == 'China':
        model = model16
        input_data = China.iloc[-1, 1:].values.tolist()
        
        future_predictions = []
        for _ in range(n_future_steps):
            input_array = np.array(input_data).reshape(1, -1)

            next_value = model.predict(input_array)[0]

            future_predictions.append(next_value)

            input_data.append(next_value)
            input_data.pop(0)
            
        future_predictions = pd.DataFrame(future_predictions)
        future_predictions = future_predictions.reset_index()
        future_predictions = future_predictions.rename(columns={future_predictions.columns[0]: 'Date'})
        future_predictions = future_predictions.rename(columns={future_predictions.columns[1]: 'future_predictions'})

        base_date = pd.Timestamp('2018-1-01')
        future_predictions['Date'] = base_date + future_predictions.index.to_series().apply(lambda x: relativedelta(months=x))
        future_predictions.set_index('Date', inplace=True)

        future_predictions['future_predictions'] = ss.inverse_transform(future_predictions[['future_predictions']])
        China['China'] = ss.inverse_transform(China[['China']])
        
        fig = plt.figure(figsize=(10, 5))
        plt.plot(China.index, China['China'], label='Actual')
        plt.plot(future_predictions.index, future_predictions, label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig)
        
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(future_predictions.index, future_predictions['future_predictions'], label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression (Predictions Only)")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig1)
        
        return fig, fig1
        
        
        
    elif model_selection == 'India':
        model = model17
        input_data = India.iloc[-1, 1:].values.tolist()
        
        future_predictions = []
        for _ in range(n_future_steps):
            input_array = np.array(input_data).reshape(1, -1)

            next_value = model.predict(input_array)[0]

            future_predictions.append(next_value)

            input_data.append(next_value)
            input_data.pop(0)
            
        future_predictions = pd.DataFrame(future_predictions)
        future_predictions = future_predictions.reset_index()
        future_predictions = future_predictions.rename(columns={future_predictions.columns[0]: 'Date'})
        future_predictions = future_predictions.rename(columns={future_predictions.columns[1]: 'future_predictions'})

        base_date = pd.Timestamp('2018-1-01')
        future_predictions['Date'] = base_date + future_predictions.index.to_series().apply(lambda x: relativedelta(months=x))
        future_predictions.set_index('Date', inplace=True)

        future_predictions['future_predictions'] = ss.inverse_transform(future_predictions[['future_predictions']])
        India['India'] = ss.inverse_transform(India[['India']])
        
        fig = plt.figure(figsize=(10, 5))
        plt.plot(India.index, India['India'], label='Actual')
        plt.plot(future_predictions.index, future_predictions, label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig)
        
        fig1 = plt.figure(figsize=(10, 5))
        plt.plot(future_predictions.index, future_predictions['future_predictions'], label='Future Predictions', linestyle='--')
        plt.title("Future Forecasting with MLP Regression (Predictions Only)")
        plt.legend()
        plt.ylabel('Carbon Emission in MtCO₂e')
        plt.xlabel('Year')
        plt.close(fig1)
        
        return fig, fig1
    else:
        return
    
    return fig

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    local_css("style.css")
    
    st.title('Time Series Regression Model Deployment')

    # st.header('Select Number of Years:')
    # n_future_steps = st.number_input('Enter the number of future steps:', min_value=1, max_value=100, value=10, step=1)

    st.header('Enter the number of future steps to forecast:')
    n_future_steps = st.text_input('The most optimal range is between 1 to 36 months, with 24  as the furthest point for prediction, in which the condition of the prediction is still optimal.', value='10')
    try:
        n_future_steps = int(n_future_steps)
    except ValueError:
        st.error('Please enter a valid number.')

    st.header('Select country to forecast:')
    model_selection = st.selectbox('17 Countries from G20 with the exception of : Russia, South Korea, and the European Union.', ['Indonesia', 'Argentina', 'United Kingdom', 'Brazil', 'Canada', 'United States', 'Italy', 'France', 'Germany', 'South Africa', 'Japan', 'Mexico', 'Saudi Arabia', 'Turkey', 'Australia', 'China', 'India'])
    
    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False

    def make_prediction():
        st.session_state.button_clicked = True

    st.button('Make Prediction', on_click=make_prediction)

    # if st.button('Make Prediction'):
    # result = make_predictions_and_plot(n_future_steps,model_selection)
    # st.success(f'The prediction is: ')
    # st.pyplot(result)

    if st.session_state.button_clicked:
        fig, fig1 = make_predictions_and_plot(n_future_steps, model_selection)
        st.success('The prediction is:')
        st.pyplot(fig)   
        
        for ax in fig1.axes:
            for line in ax.lines:
                line.set_color('orange')
        
        st.pyplot(fig1)
        

if __name__ == '__main__':
    main()