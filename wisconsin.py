# Code by Emi Aoki

import requests
from datetime import datetime
import pandas as pd
from io import BytesIO

import df_timeSeries


def get_WI_POPL_DATA_COUSUB(location_list, dataYear=2020, fileYear=2022):
  """
  Get Wisconsin municipal-level population data from State of Wisconsin Department of Administration

  Parameters
  ----------
  location_list : list or array holds names of geographic locations of interest
                  - i.e., cities, towns
  dataYear : int

  Returns
  -------
  tmp_list : list of population data; listed as location_list
  """
  if dataYear > datetime.now().year:
    print('Population Data for selected year DOES NOT EXISTS')
    return None

  link = f'https://doa.wi.gov/DIR/Time_Series_MCD_{fileYear}.xlsx'
  res = requests.get(link)
  if res.status_code != 200:
    res = requests.get(f'https://doa.wi.gov/DIR/Time_Series_MCD_{fileYear-1}.xlsx')
    if res.status_code == 200:
      print(f'FILE EXISTS ({datetime.now().year} ver.')
    else:
      print('CANNOT ACCESS DATA FILE FROM THE WEBSITE.')
      return None
  # res = requests.get(link)
  if res.ok:
    data = res.content
    df = pd.io.excel.read_excel(BytesIO(data))
  dfcp=df.copy()
  dfcp.columns = dfcp.iloc[3]
  dfcp.columns.name = None
  dfcp.drop(index=[0,1,2,3], axis=0, inplace=True)
  dfcp = dfcp.loc[dfcp['MCD Type'] == 'C'][['Municipality',f'Final 1/1/{dataYear} Estimate']]
  dfcp['Municipality'] = dfcp['Municipality'].str.replace(' \*','')
  dfcp['Municipality'] = dfcp['Municipality'] + ' city'
  dfcp = dfcp.groupby('Municipality').sum()
  
  tmp_list = ['']*len(location_list)
  for i, name in enumerate(location_list):
    tmp_list[i] = dfcp.loc[name].values[0]
  return tmp_list


def get_WI_POPL_DATA_COU(location_list, dataYear=2020, fileYear=2022):
  """
  Get Wisconsin county-level population data from State of Wisconsin Department of Administration

  Parameters
  ----------
  location_list : list or array holds names of geographic locations of interest
                  - i.e., counties 
  dataYear : int

  Returns
  -------
  tmp_list : list of population data; listed as location_list
  """
  if dataYear > datetime.now().year:
    print('Population Data for selected year DOES NOT EXISTS')
    return None

  link = f'https://doa.wi.gov/DIR/Time_Series_Co_{fileYear}.xlsx'
  res = requests.get(link)
  if res.status_code != 200:
    res = requests.get(f'https://doa.wi.gov/DIR/Time_Series_Co_{fileYear-1}.xlsx')
    if res.status_code == 200:
      print(f'FILE EXISTS ({datetime.now().year} ver.')
    else:
      print('CANNOT ACCESS DATA FILE FROM THE WEBSITE.')
      return None
  # res = requests.get(link)
  if res.ok:
    data = res.content
    df = pd.io.excel.read_excel(BytesIO(data))
  dfcp=df.copy()
  dfcp.columns = dfcp.iloc[2]
  dfcp.columns.name = None
  dfcp.drop(index=[0,1,2], axis=0, inplace=True)
  dfcp = dfcp[['County Name',f'Final 1/1/{dataYear} Estimate']]
  dfcp['County Name'] = dfcp['County Name'].str.replace(' \*','')
  dfcp.drop(dfcp.loc[dfcp['County Name']=='STATE Total'].index.values[0], axis=0, inplace=True)
  dfcp = dfcp.groupby('County Name').sum()

  tmp_list = ['']*len(location_list)
  for i, name in enumerate(location_list):
    tmp_list[i] = dfcp.loc[name].values[0]
  return tmp_list


def get_COVIDdata_TS_interval(df_TS, location_list, date_Ymd, freq='1W', agg_type='mean'):
  """
  Extract time series data based on the specifications: date_Ymd, freq, and agg_type

  Parameters
  ----------
  df_TS : pandas dataframe contains time series data
   - row: times (assuming daily)
   - column: locations
  location_list : list or array contains name of geographic locations of interest
  date_Ymd : date in YYYY-mm-dd format
  freq : string indicates time interval of interest
  agg_type : string indicates types of data aggregation -i.e.,'sum','mean','var','min','max',etc

  Returns
  -------
  tmp_df : pandas dataframe contains specified data extracted from the input
    row : observations = location_list
    columns : COVID cases aggregated based on speficied time interval, date, data aggregation type
  """
  cp = df_TS.copy()
  cp = cp.reset_index().rename(columns={'index':'DATE'})
  cp['DATE'] = pd.to_datetime(cp['DATE'])
  cp.set_index('DATE',inplace=True)
  cp = cp.resample(freq, label='left', closed='left').agg([agg_type]).reset_index()
  tm = pd.DataFrame(cp.loc[cp['DATE'] <= date_Ymd].iloc[-1,1:])
  tm.reset_index(level=1, drop=True, inplace=True)
  tm = tm.reset_index().rename(columns={'index':'NAME'})
  tm = tm.rename(columns={tm.columns.to_list()[-1]:'COVID'})
  return tm



def get_COVIDdata_TS_interval_cum(df_TS, location_list, date_Ymd, var_list, freq='1W', agg_type='mean'):
  """
  Extract time series data based on the specifications: date_Ymd, freq, and agg_type

  Parameters
  ----------
  df_TS : pandas dataframe contains cumulative numbers of data
   - row: observations by geographic locations over time
   - column: variables (i.e., 'POS_FEM', 'POS_MALE', etc) - 'DATE' and 'NAME' have to be included
  location_list : list or array contains name of geographic locations of interest
  date_Ymd : date in YYYY-mm-dd format
  var_list : list array of variable names of interest
  freq : string indicates time interval of interest
  agg_type : string indicates types of data aggregation -i.e.,'sum','mean','var','min','max',etc

  Returns
  -------
  tmp_df : pandas dataframe contains specified data extracted from the input
    row : observations = location_list
    columns : variable(s) aggregated based on speficied time interval, date, data aggregation type
  """
  cp = df_TS.copy()
  for i, name in enumerate(var_list):
    tmp = df_timeSeries.generate_TSDataFrame(df_TS=cp, location_list=location_list, var_name=name)
    tmp = tmp.diff()[1:] # values are cumulative; thus, diff() is computed
    tmp = df_timeSeries.generate_df_moving_average_7DAY(tmp) # 7-day moving averagae data
    tmp = get_COVIDdata_TS_interval(df_TS=tmp, location_list=location_list, date_Ymd=date_Ymd, freq=freq, agg_type=agg_type)
    tmp.rename(columns={tmp.columns.to_list()[-1]: name.replace('_CP','') if '_CP' in name else name}, inplace=True)
    if i==0:
      tmp_all = tmp.copy()
    else:
      tmp_all = tmp_all.merge(tmp, on='NAME')
  return tmp_all



# This functions won't be needed if input dataframe is already formatted, 
def get_COVIDdata_interval(df_TS, location_list, date_Ymd, var_list=['POS_NEW'], freq='1W', agg_type='sum'):
  """
  NOTE1: This function won't be needed if input dataframe is already formatted, such as an output of functions in df_timeSeries.py
  NOTE2: This function assumes that the format of input data frame is the same as the format used by Wisconsin COVID raw data
  REF1: https://towardsdatascience.com/pandas-resample-tricks-you-should-know-for-manipulating-time-series-data-7e9643a7e7f3
  REF2: https://www.earthdatascience.org/courses/use-data-open-source-python/use-time-series-data-in-python/date-time-types-in-pandas-python/resample-time-series-data-pandas-python/
  
  Parameters
  ----------
  df_TS : pandas dataframe contains time series data
   - row: observations for multiple geographic area over time
   - column: variables (i.e., 'POS_NEW', 'NEG_NEW' etc)
  location_list : list or array contains name of geographic locations of interest
  date_Ymd : date in YYYY-mm-dd format
  var_list : list or array contains variable name to be extracated from the data set
  freq : string indicates time interval of interest
  agg_type : string indicates types of data aggregation -i.e.,'sum','mean','var','min','max',etc

  Returns
  -------
  tmp_df : pandas dataframe contains specified data extracted from the input
    row : observations
    columns : selected variable(s)
  """
  cp = df_TS.copy()
  cp.set_index('DATE', inplace=True)
  cp = cp.groupby('NAME').resample('D').sum()
  tmp_df = pd.DataFrame(index=location_list, columns=var_list)
  for i, name in enumerate(location_list):
    tmp = cp.loc[name].reset_index().set_index('DATE').iloc[1:,:].resample(freq, label='left', closed='left').agg([agg_type]).reset_index()
    tmp_df.iloc[i] = tmp.loc[tmp['DATE'] <= date_Ymd].iloc[-1][var_list].values
  print('CHECKING >>> ')
  print('DATE:', tmp.loc[tmp['DATE'] <=date_Ymd].iloc[-1]['DATE'])
  tmp_df.reset_index(inplace=True)
  tmp_df.rename(columns={'index':'NAME'}, inplace=True)
  if tmp_df.columns.str.contains('_CP').any():
    tmp_df.columns = tmp_df.columns.str.replace('_CP', '')
  print(f'Extracted data: {var_list}\ninterval: {freq}\ndata aggregation type: {agg_type}')
  return tmp_df