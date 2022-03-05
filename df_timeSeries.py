import numpy as np
import pandas as pd

def generate_TSDataFrame(df_TS, location_list, var_name='POS_NEW'):
  """
  Parameters
  ----------
  df_TS : pandas dataframe 
    - row: observations for each location over time (assuming datetime daily)
    - column: variables such as 'POS_NEW', 'NEG_POS', etc
  location_list : list or array 
    - contains names of geographic locations of interest (cities, towns, states, etc)
  var_name : string holds a variable name of interest
    - by default, 'POS_NEW' - WI COVID-19 daily positive cases

  Returns
  -------
  tmpDF : pandas dataframe
    - row: observations over time (assuming datetime daily)
    - column: locations of interest (cities, towns, states, etc)
  """
  cp = df_TS.copy()
  cp.set_index('DATE', inplace=True)
  cp = cp.groupby('NAME').resample('D').sum()
  tmpDF = pd.DataFrame(columns=np.concatenate((['DATE'], location_list)))
  tmpDF['DATE'] = df_TS.DATE.unique().date[1:]
  for i, name in enumerate(location_list):
    tmpDF[name] = cp.loc[name].iloc[1:,:][var_name].values
  tmpDF.set_index('DATE', inplace=True)
  tmpDF.index.name = None
  return tmpDF


def modify_negativevalues(df_TS, location_list):
  """
  For negative values at time t, the average of before and after time t is computed

  Parameters
  ----------
  df_TS : pandas dataframe
    - row: observations over time (assuming datetime daily)
    - column: locations of interest (cities, towns, states, etc)
  location_list : list or array
    - contains names of geographic locations of interest
  
  Returns 
  -------
  cp : pandas dataframe
   - row: observations over time (assuming datetime daily)
   - column: locations of interest
  """
  cp = df_TS.copy()
  cp.reset_index(inplace=True)
  for name in location_list:
    if cp[name].lt(0).any():
      indLst = cp[name].loc[cp[name] < 0].index.to_list()
      for i in indLst:
        if i > 0 and i < cp.shape[0]-1:
          #cp[name].iloc[i] = (cp[name].iloc[i+1] + cp[name].iloc[i-1]) / 2
          cp.loc[i,name] = (cp[name].iloc[i+1] + cp[name].iloc[i-1]) / 2
  cp.set_index('index', inplace=True)
  cp.index.name = None
  return cp


def generate_df_moving_average_7DAY(df_TS):
  """
  Compute 7-day average - based on Professor Thompson's code

  Parameters
  ----------
  df_TS : pandas dataframe
    - row: observations over time (assuming datetime daily)
    - column: locations of interest

  Returns
  -------
  pd.DataFrame(y) :  pandas dataframe
    - row: ovservations over time - 7-day averaged
    - column: locations of interest
  """
  window_size=7
  ioff=3
  np_TS = df_TS.to_numpy()
  N = df_TS.shape[0]
  y = np.zeros(np_TS.shape)
  for i in range(ioff, N-ioff): # N-1-ioff
    y[i,:] = (np_TS[i,:] + (np_TS[i-1,:]+np_TS[i+1,:]) + (np_TS[i-2,:]+np_TS[i+2,:]) + (np_TS[i-3,:]+np_TS[i+3,:]) ) / window_size

  # 01/08/2022 fix indexing
  y[0,:] = (np_TS[0,:] + np_TS[1,:] + np_TS[2,:] + np_TS[3,:]) / 4
  y[1,:] = (np_TS[0,:] + np_TS[1,:] + np_TS[2,:] + np_TS[3,:] + np_TS[4,:]) / 5
  y[2,:] = (np_TS[0,:] + np_TS[1,:] + np_TS[2,:] + np_TS[3,:] + np_TS[4,:] + np_TS[5,:]) / 6

  y[N-1,:] = (np_TS[N-1,:] + np_TS[N-2,:] + np_TS[N-3,:] + np_TS[N-4,:]) / 4
  y[N-2,:] = (np_TS[N-1,:] + np_TS[N-2,:] + np_TS[N-3,:] + np_TS[N-4,:] + np_TS[N-5,:]) / 5
  y[N-3,:] = (np_TS[N-1,:] + np_TS[N-2,:] + np_TS[N-3,:] + np_TS[N-4,:] + np_TS[N-5,:] + np_TS[N-6,:]) / 6

  return pd.DataFrame(data=y, columns=df_TS.columns, index=df_TS.index)
