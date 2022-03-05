# this py file contains functions to generate data frame(s) 
#   for age groups, race/ethnicity groups, and occupations groups

import df_census
import pandas as pd

def generate_dataframe_census(dataType, state, location_list, grp_lst, grp_dict, yr=2019, lvl='cousubs', joint_race_lst=None, joint_race_dict=None, i_interest=21):
  """
  This function will return complete data frame(s) after cleaning up, instead of returning raw census data.
  If raw data is required, use getDataUsingCensusAPI(Grp, tableID, state, yr, lvl) in df_census.py

  Parameters
  ----------
  dataType: strig holds data name to generate a data frame using the U.S. Census Bureau data tables
    - 'age': 'B01001' data table - contains population data by age group and geneder
    - 'work': 'C24010' data table - contains occupation data by gender
    - 'race': 'B03002' data table - contains population data by race and ethnicity group
    - 'age_race' or 'race_age': 'B01001A~I' data tables - population data by age group, race/ethnicity, and gender
  state : string holds state name
  location_list : list or array cotains name of munipal-level locations
  grp_lst : list or array cotains names of new group for age variable
  grp_dict: dictionary to convert the default variable names into new age group
  yr : int
  lvl : string - either 'cousubs' or 'counties'
  joint_race_lst : list or array contains alphabet letters; more infomation in https://censusreporter.org/topics/race-hispanic/
  joint_race_dict: dictionary to convert the default race/ethnicity group into new groups
  i_interest : int for number of variables 
  """
  if dataType.lower() == 'age':
    return generate_dataframe_censusData_dataWithGender('B01001', state, location_list, grp_lst, grp_dict, yr=yr, lvl=lvl)
    
  elif dataType.lower() == 'work':
    return generate_dataframe_censusData_dataWithGender('C24010', state, location_list, grp_lst, grp_dict, yr=yr, lvl=lvl)
    
  elif dataType.lower() == 'race':
    return generate_RaceGrp_dataframe_censusData(state, location_list, grp_lst, grp_dict, yr=yr, lvl=lvl)

  elif dataType.lower() == 'poverty':
    return generate_povertyStatus_dataframe_censusSubjectTable(state, location_list, yr=yr, lvl=lvl, i_interest=21)
    
  elif (dataType.lower() == 'age_race') or (dataType.lower() == 'race_age'):
    return generate_RaceAgeGrp_dataframe_censusData(state, location_list, grp_lst, grp_dict, yr, lvl, joint_race_lst, joint_race_dict)
    
  else:
    print('Error: Invalid keywork is passed')




def get_certain_locationsData_only(tmp_df, location_list):
  """
  Parameters
  ----------
  tmp_df : pandas dataframe
    - row: variables
    - column: location names

  Returns
  -------
  tmp_df_int : pandas dataframe
    - row: observations for selected locations only
    - column: variables
  """
  tmp_lst = tmp_df.columns.to_list()
  outp = [i for i, item in enumerate(tmp_lst) if any(place in item for place in location_list)]
  tmp_df_int = tmp_df[tmp_df.columns[outp]]
  tmp_df_int = tmp_df_int.rename(columns={col: col.split(', ')[0] for col in tmp_df_int.columns})
  tmp_df_int = tmp_df_int.T.reset_index()
  tmp_df_int = tmp_df_int.groupby('').sum().T
  return tmp_df_int



def generate_dataframe_censusData_dataWithGender(tableID, state, location_list, grp_lst, grp_dict, yr=2019, lvl='cousubs'):
  """
  Census Data Contains gender categories ('Male', 'Female') 
  i.e., 'B01001' - age group, 'C24010' - occupation
  list of available data (2019ver., acs5) is in https://api.census.gov/data/2019/acs/acs5/groups.html

  Parameters
  ----------
  state : string holds state name
  location_list : list or array cotains name of munipal-level locations
  grp_lst : list or array cotains names of new group for variables
  grp_dict: dictionary to convert the default variable names into new group
  yr : int

  Returns
  -------
  df_Grp_A : pandas dataframe combining data of male and female
   - row: observations; size = len(location_list)
   - column: variables - ['NAME'] + grp_lst
  df_Grp_M : pandas dataframe - data for male
  df_Grp_F : pandas dataframe - data for female
  """
  Grp = 'Grp'
  df_int, _ = df_census.getDataUsingCensusAPI(Grp, tableID, state, yr=yr, lvl=lvl) #Grp, tableID, state, yr=2019, lvl='cousubs'
  df_int.reset_index(inplace=True)
  df_int = df_int.replace({Grp: grp_dict})
  
  m_ind = df_int.loc[df_int[Grp]=='Male:'].index[0]
  f_ind = df_int.loc[df_int[Grp]=='Female:'].index[0]
  if f_ind > m_ind:
    df_M = df_int.iloc[m_ind:f_ind-1]
    df_F = df_int.iloc[f_ind:]
  else:
    df_M = df_int.iloc[m_ind:]
    df_F = df_int.iloc[f_ind:m_ind-1]

  df_all = df_int.loc[df_int[Grp].str.contains('|'.join(grp_lst+['Total:']))].groupby(Grp).sum()
  df_all = get_certain_locationsData_only(df_all, location_list)
  df_Grp_A = pd.DataFrame(columns=['NAME']+grp_lst)
  df_Grp_A['NAME'] = location_list
  for name in grp_lst: df_Grp_A[name] = df_all.loc[name].values

  df_M = df_M.loc[df_M[Grp].str.contains('|'.join(grp_lst+['Male:']))].groupby(Grp).sum()
  df_M.rename(index={'Male:':'Total:'},inplace=True)
  df_M = get_certain_locationsData_only(df_M, location_list)
  df_Grp_M = pd.DataFrame(columns=['NAME']+grp_lst)
  df_Grp_M['NAME'] = location_list
  for name in grp_lst: df_Grp_M[name] = df_M.loc[name].values

  df_F = df_F.loc[df_F[Grp].str.contains('|'.join(grp_lst+['Female:']))].groupby(Grp).sum()
  df_F.rename(index={'Female:':'Total:'},inplace=True)
  df_F = get_certain_locationsData_only(df_F, location_list)
  df_Grp_F = pd.DataFrame(columns=['NAME']+grp_lst)
  df_Grp_F['NAME'] = location_list
  for name in grp_lst: df_Grp_F[name] = df_F.loc[name].values
  
  return df_Grp_A, df_Grp_M, df_Grp_F



def generate_RaceGrp_dataframe_censusData(state, location_list, grp_lst, grp_dict, yr=2019, lvl='cousubs'):
  """
  Race and Ethnicity Group data from Census Data Table: B03002

  Parameters
  ----------
  state : string holds state name
  location_list : list or array cotains name of munipal-level locations
  grp_lst : list or array cotains names of new group for variables
  grp_dict: dictionary to convert the default variable names into new group
  yr : int

  Returns
  -------
  df_raceGrp : pandas dataframe
   - row: observations; size = len(location_list)
   - column: variables - ['NAME'] + grp_lst
  """
  Grp = 'RaceGrp'
  tableID = 'B03002'
  df_race_int, _ = df_census.getDataUsingCensusAPI(Grp, tableID, state, yr=yr, lvl=lvl)
  df_race_int = df_race_int.iloc[2:12]
  df_race_int.drop(index=df_race_int.index[6:9], inplace=True) # 11/23/2021 .index[6:9] => [7:9] #12/4/21 .index[7:9]=>.index[6:9]
  df_race_int.index = df_race_int.index.str.replace(' alone', '')
  df_race_int.reset_index(inplace=True)
  df_race_int = df_race_int.replace({Grp: grp_dict})
  df_race_int = df_race_int.loc[df_race_int[Grp].str.contains('|'.join(grp_lst+['Total:']))].groupby(Grp).sum()
  racePopl_int = get_certain_locationsData_only(df_race_int, location_list)
  
  # the following code will make sure that municipal-level race/ethnicty data will have the same order as location_list
  df_raceGrp = pd.DataFrame(columns=['NAME']+grp_lst)
  df_raceGrp['NAME'] = location_list
  for name in grp_lst:
    df_raceGrp[name] = racePopl_int.loc[name].values
  return df_raceGrp 


# 11/27/21
def generate_povertyStatus_dataframe_censusSubjectTable(state, location_list, yr=2019, lvl='cousubs', i_interest=21):
  """
  Poverty Status Data from Census Subject Table: S1701

  Parameters
  ----------
  state : string holds state name
  location_list : list or array cotains name of munipal-level locations
  yr : int
  lvl : cousubs or counties

  Returns
  -------
  df_pov : pandas dataframe; _c1 for total, _c2 for below poverty level, _c3 for percent below poverty level
   - row: observations; size = len(location_list)
   - column: variables - ['NAME'] +  list(variables names in data frame)
  """
  Grp='Grp'
  tableID='S1701'
  est_c1, est_c2, est_c3, mgn_c1, mgn_c2, mgn_c3 = df_census.getDataUsingCensusAPI_subjectTables(Grp=Grp,tableID=tableID, state=state, yr=yr, lvl=lvl, i_interest=i_interest)
  est_c1_int = get_certain_locationsData_only(est_c1, location_list)
  est_c2_int = get_certain_locationsData_only(est_c2, location_list)
  est_c3_int = get_certain_locationsData_only(est_c3, location_list)

  grp_lst = list(est_c1_int.index)
  df_pov_c1 = pd.DataFrame(columns=['NAME']+grp_lst); df_pov_c1['NAME'] = location_list
  df_pov_c2 = pd.DataFrame(columns=['NAME']+grp_lst); df_pov_c2['NAME'] = location_list
  df_pov_c3 = pd.DataFrame(columns=['NAME']+grp_lst); df_pov_c3['NAME'] = location_list
  for name in grp_lst:
    df_pov_c1[name] = est_c1_int.loc[name].values
    df_pov_c2[name] = est_c2_int.loc[name].values
    df_pov_c3[name] = est_c3_int.loc[name].values
  return (df_pov_c1, df_pov_c2, df_pov_c3)


def generate_RaceAgeGrp_dataframe_censusData(state, location_list, grp_lst, grp_dict, yr=2019, lvl='cousubs', joint_race_lst=None, joint_race_dict=None):
  """
  Joint variables Data - Age-Group and Race/Ethnicity Group data from Census Data Table: B01001A~I
  This contains gender data as well

  Parameters
  ----------
  state : string holds state name
  location_list : list or array cotains name of munipal-level locations
  grp_lst : list or array cotains names of new group for age variable
  grp_dict: dictionary to convert the default variable names into new age group
  yr : int
  joint_race_lst : list or array contains alphabet letters; more infomation in https://censusreporter.org/topics/race-hispanic/
  joint_race_dict: dictionary to convert the default race/ethnicity group into new groups

  Returns
  -------
  df_Grp_A : pandas dataframe combining data of male and female
   - row: observations; size = len(location_list)
   - column: variables - ['NAME'] + grp_lst + joint_race_lst
  df_Grp_M : pandas dataframe - data for male
  df_Grp_F : pandas dataframe - data for female
  """
  Grp = 'Grp'
  tableID_Base = 'B01001'
  if not joint_race_lst: joint_race_lst = ['B', 'C', 'D', 'E', 'F', 'H', 'I']
  if not joint_race_dict: joint_race_dict = {'A':'W', 'B':'BAA', 'C':'OR', 'D':'A', 'E':'OR', 'F':'OR', 'G':'OR', 'H':'NHW', 'I':'HL'} #  OR-other race
  tableIDs = [tableID_Base+i for i in joint_race_lst]
  for i, tableID in enumerate(tableIDs):
    df_A0, df_M0, df_F0 = generate_dataframe_censusData_dataWithGender(tableID, state, location_list, grp_lst, grp_dict, yr=yr, lvl=lvl)
    df_A0 = df_A0.rename(columns={col: '_'.join((joint_race_dict[tableID[-1]], col)) for col in df_A0.iloc[:,1:].columns})
    df_M0 = df_M0.rename(columns={col: '_'.join((joint_race_dict[tableID[-1]], col)) for col in df_M0.iloc[:,1:].columns})
    df_F0 = df_F0.rename(columns={col: '_'.join((joint_race_dict[tableID[-1]], col)) for col in df_F0.iloc[:,1:].columns})
    if i==0:
      df_A = df_A0.copy()
      df_M = df_M0.copy()
      df_F = df_F0.copy()
    else:
      df_A = df_A.merge(df_A0, on='NAME')
      df_M = df_M.merge(df_M0, on='NAME')
      df_F = df_F.merge(df_F0, on='NAME')

  for i in ['_x', '_y']:
    if df_A.columns.str.contains(i).any():
      df_A.columns = [col.replace(i,'') for col in df_A.columns]
      df_M.columns = [col.replace(i,'') for col in df_M.columns]
      df_F.columns = [col.replace(i,'') for col in df_F.columns]

  df_A = df_A.groupby(by=df_A.columns, axis=1).sum().set_index('NAME').reset_index()
  df_M = df_M.groupby(by=df_M.columns, axis=1).sum().set_index('NAME').reset_index()
  df_F = df_F.groupby(by=df_F.columns, axis=1).sum().set_index('NAME').reset_index()
  return df_A, df_M, df_F
