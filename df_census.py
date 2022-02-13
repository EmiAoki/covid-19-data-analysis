# Code by Emi Aoki

import pandas as pd
import requests


# STATE FIPS CODE from census
link = 'https://www2.census.gov/geo/docs/reference/state.txt'
# FIPS State Code | Official Unites States Postal Service (USPD) Code | NAME | Geographic Names Information System Identifier (GNISID)
FIPS_state = pd.read_csv(link, '|')

sqmiTosqkm = 2.59 # sq mi * sqmiTosqkm = sq km

# return FIPS code for selected state
def get_STATE_FIPS_CODE(state):
  """
  Parameters
  ----------
  state : string holds state name of interest
  """
  if FIPS_state.loc[FIPS_state.STATE_NAME == state].shape[0] == 1:
    return FIPS_state.loc[FIPS_state.STATE_NAME == state].STATE.values[0] # return FIPS code as int64
  else: return 0 # meaning FIPS code wasn't be obtained correctly

def get_USPD(state):
  """
  Parameters
  ----------
  state : string holds state name of interest
  """
  if FIPS_state.loc[FIPS_state.STATE_NAME == state].shape[0] == 1:
    return FIPS_state.loc[FIPS_state.STATE_NAME == state].STUSAB.values[0] # return FIPS code as int64
  else: return None

def create_Census_dataTable_labels_dict(tableID, yr=2019, tableCode=None):
  """
  Parameters
  ----------
  tableID : string holds a Census table name on interest 
  yr : ACS year of interest
  tableCode: letter listed in https://censusreporter.org/topics/table-codes/
            - S: Subject tables
            - None or B/C: common ACS data tables

  Returns
  -------
  est_dict : dictionary holds lables and names for estimate variables 
  mgn_dict : dictionary holds labels and names for margin variables
  """
  link = f"https://api.census.gov/data/{str(yr)}/acs/acs5/groups/{tableID}.html"
  if tableCode != None:
      if tableCode.upper() == 'S':
          link = f"https://api.census.gov/data/{str(yr)}/acs/acs5/subject/groups/{tableID}.html"
  tmp = requests.get(link)
  if not tmp.ok:
    print('Error: selected table ID and/or year data do not exist.')
    return None, None
  tbl = pd.read_html(tmp.text, )
  tbl = tbl[0]
  newtbl = tbl.iloc[:-1:2,:-1][['Name','Label']]
  newtbl['Label'] = newtbl['Label'].str.replace('!!', ': ')
  est_tbl = newtbl[::2]; 
  mgn_tbl = newtbl[1::2];
  for i in range(est_tbl.shape[0]):
    est_tbl['Label'].iloc[i] = est_tbl['Label'].iloc[i].split(': ')[-1]
    mgn_tbl['Label'].iloc[i] = mgn_tbl['Label'].iloc[i].split(': ')[-1]
  est_dict = dict(est_tbl.values)
  mgn_dict = dict(mgn_tbl.values)
  return est_dict, mgn_dict




# Access census data using api =================================================
def getDataUsingCensusAPI(Grp, tableID, state, yr=2019, lvl='cousubs'):
  """
  Access American Community Survey 5-Year Data from the Census Bureau
  Estimate and margin data frames are generated and returned
  NOTE: list of available data (2019ver., acs5) is in https://api.census.gov/data/2019/acs/acs5/groups.html

  Parameters
  ----------
  Grp : string holds a name of group - can be any name - used when clearning up data
  tableID : string holds a Census table name on interest 
  state : string holds state name of interest
  yr : ACS year of interest; yr=2019 means ACS 5-year 2015-2019 data
  lvl : string - either 'cousubs' or 'counties'

  Returns
  -------
  est_df : pandas dataframe
  mgn_df : pandas dataframe
  """
  est_dict, mgn_dict = create_Census_dataTable_labels_dict(tableID, yr)
  if est_dict == None:
    return None, None

  host = f"https://api.census.gov/data/{str(yr)}/acs/acs5?get=group({tableID})&for=county%20subdivision:*&in=state:{str(get_STATE_FIPS_CODE(state))}"
  if lvl=='counties':
      host = f"https://api.census.gov/data/{str(yr)}/acs/acs5?get=group({tableID})&for=county:*&in=state:{str(get_STATE_FIPS_CODE(state))}"
  tmp = requests.get(host)
  dftmp = pd.DataFrame(data=tmp.json())
  dftmp.columns = dftmp.iloc[0]
  dftmp = dftmp.drop(0)
  dftmp1= dftmp.T
  dftmp1= dftmp1[:-5:2]
  est_df = dftmp1[::2];                          mgn_df = dftmp1[1::2]
  est_df = est_df.astype(int).copy();            mgn_df = mgn_df.astype(int).copy()
  dftmp2 = dftmp.T
  dftmp2 = dftmp2.loc[['NAME']]
  est_df = pd.concat([dftmp2, est_df], axis=0);  mgn_df = pd.concat([dftmp2, mgn_df], axis=0)
  est_df.rename(index={'NAME':''}, inplace=True);mgn_df.rename(index={'NAME':''}, inplace=True)
  est_df.columns = est_df.iloc[0];               mgn_df.columns = mgn_df.iloc[0]
  est_df = est_df[1:];                           mgn_df = mgn_df[1:]
  est_df.index.name = Grp;                       mgn_df.index.name = Grp
  est_df.rename(index=est_dict, inplace=True);   mgn_df.rename(index=mgn_dict, inplace=True)
  return est_df, mgn_df


def getDataUsingCensusAPI_subjectTables(Grp, tableID, state, yr=2019, lvl='cousubs', i_interest=21):
  """
  NOTE: this function works well with S1701 only: others might cause an error
  Access American Community Survey 5-Year Data Subject Tables from the Census Bureau
  NOTE: description of subject tables available at https://censusreporter.org/topics/table-codes/
  Estimate and margin data frames are generated and returned
  NOTE: list of available data (2019ver., acs5) is in https://api.census.gov/data/2019/acs/acs5/subject/groups.html

  Parameters
  ----------
  Grp : string holds a name of group - can be any name - used when clearning up data
  tableID : string holds a Census subject table name (table name starting from 'S') on interest 
  state : string holds state name of interest
  yr : ACS year of interest; yr=2019 means ACS 5-year 2015-2019 data
  lvl : string - either 'cousubs' or 'counties'
  i_interest: int - number of variables to be returned

  Returns
  -------
  est_df : pandas dataframe; _c1 for total, _c2 for below poverty level, _c3 for percent below poverty level
  mgn_df : pandas dataframe; _c1 for total, _c2 for below poverty level, _c3 for percent below poverty level
  """
  est_dict, mgn_dict = create_Census_dataTable_labels_dict(tableID, yr, tableCode='S')
  if est_dict == None:
    return None, None
  host = f"https://api.census.gov/data/{str(yr)}/acs/acs5/subject?get=group({tableID})&for=county%20subdivision:*&in=state:{str(get_STATE_FIPS_CODE(state))}"
  if lvl=='counties':
      host = f"https://api.census.gov/data/{str(yr)}/acs/acs5/subject?get=group({tableID})&for=county:*&in=state:{str(get_STATE_FIPS_CODE(state))}"
  tmp = requests.get(host)
  dftmp = pd.DataFrame(data=tmp.json())
  dftmp.columns = dftmp.iloc[0]
  dftmp = dftmp.drop(0)
  dftmp1= dftmp.T
  dftmp1 = dftmp1[2:-3:2].copy()
  est_df = dftmp1[::2];                       mgn_df = dftmp1[1::2]
  est_df = est_df.reset_index().copy();       mgn_df = mgn_df.reset_index().copy()
  est_df_c1 = est_df.loc[est_df[0].str.contains('C01')].copy(); est_df_c1 = est_df_c1[:i_interest].copy(); est_df_c1.set_index(0, inplace=True); est_df_c1 = est_df_c1.astype(int).copy()
  est_df_c2 = est_df.loc[est_df[0].str.contains('C02')].copy(); est_df_c2 = est_df_c2[:i_interest].copy(); est_df_c2.set_index(0, inplace=True); est_df_c2 = est_df_c2.astype(int).copy()
  est_df_c3 = est_df.loc[est_df[0].str.contains('C03')].copy(); est_df_c3 = est_df_c3[:i_interest].copy(); est_df_c3.set_index(0, inplace=True); est_df_c3 = est_df_c3.astype(float).copy()
  mgn_df_c1 = mgn_df.loc[mgn_df[0].str.contains('C01')].copy(); mgn_df_c1 = mgn_df_c1[:i_interest].copy(); mgn_df_c1.set_index(0, inplace=True); mgn_df_c1 = mgn_df_c1.astype(int).copy()
  mgn_df_c2 = mgn_df.loc[mgn_df[0].str.contains('C02')].copy(); mgn_df_c2 = mgn_df_c2[:i_interest].copy(); mgn_df_c2.set_index(0, inplace=True); mgn_df_c2 = mgn_df_c2.astype(int).copy()
  mgn_df_c3 = mgn_df.loc[mgn_df[0].str.contains('C03')].copy(); mgn_df_c3 = mgn_df_c3[:i_interest].copy(); mgn_df_c3.set_index(0, inplace=True); mgn_df_c3 = mgn_df_c3.astype(float).copy()

  dftmp2 = dftmp.T.copy()
  dftmp2 = dftmp2.loc[['NAME']].copy()

  est_df_c1 = pd.concat([dftmp2, est_df_c1], axis=0); est_df_c1.rename(index={'NAME':''}, inplace=True); est_df_c1.columns = est_df_c1.iloc[0];
  est_df_c1 = est_df_c1[1:]; est_df_c1.index.name = Grp; est_df_c1.rename(index=est_dict, inplace=True)
  est_df_c2 = pd.concat([dftmp2, est_df_c2], axis=0); est_df_c2.rename(index={'NAME':''}, inplace=True); est_df_c2.columns = est_df_c2.iloc[0];
  est_df_c2 = est_df_c2[1:]; est_df_c2.index.name = Grp; est_df_c2.rename(index=est_dict, inplace=True)
  est_df_c3 = pd.concat([dftmp2, est_df_c3], axis=0); est_df_c3.rename(index={'NAME':''}, inplace=True); est_df_c3.columns = est_df_c3.iloc[0];
  est_df_c3 = est_df_c3[1:]; est_df_c3.index.name = Grp; est_df_c3.rename(index=est_dict, inplace=True)

  mgn_df_c1 = pd.concat([dftmp2, mgn_df_c1], axis=0); mgn_df_c1.rename(index={'NAME':''}, inplace=True); mgn_df_c1.columns = mgn_df_c1.iloc[0];
  mgn_df_c1 = mgn_df_c1[1:]; mgn_df_c1.index.name = Grp; mgn_df_c1.rename(index=est_dict, inplace=True)
  mgn_df_c2 = pd.concat([dftmp2, mgn_df_c2], axis=0); mgn_df_c2.rename(index={'NAME':''}, inplace=True); mgn_df_c2.columns = mgn_df_c2.iloc[0];
  mgn_df_c2 = mgn_df_c2[1:]; mgn_df_c2.index.name = Grp; mgn_df_c2.rename(index=est_dict, inplace=True)
  mgn_df_c3 = pd.concat([dftmp2, mgn_df_c3], axis=0); mgn_df_c3.rename(index={'NAME':''}, inplace=True); mgn_df_c3.columns = mgn_df_c3.iloc[0];
  mgn_df_c3 = mgn_df_c3[1:]; mgn_df_c3.index.name = Grp; mgn_df_c3.rename(index=mgn_dict, inplace=True)

  return est_df_c1, est_df_c2, est_df_c3, mgn_df_c1, mgn_df_c2, mgn_df_c3


def get_LAND_AREA(location_list, state, yr=2020, lvl='cousubs'):
  """
  Parameters
  ----------
  location_list : list or array
  state : string
  yr : int
  lvl : string - either 'cousubs' or 'counties'

  Returns
  -------
  tmp_list : list contains land area of selected locations as listed in location_list
  """
  link = f'https://www2.census.gov/geo/docs/maps-data/data/gazetteer/{yr}_Gazetteer/{yr}_gaz_{lvl}_{get_STATE_FIPS_CODE(state)}.txt'
  tmp = requests.get(link)
  if not tmp.ok:
    print('Error: selected table ID and/or year data do not exist.')
    return None
  dfarea = pd.read_csv(link, '\t')
  tmp_list = ['']*len(location_list)
  for i, name in enumerate(location_list):
    tmp_list[i] = dfarea.loc[dfarea.NAME == name].ALAND_SQMI.sum() * sqmiTosqkm
  return tmp_list


# for grouping city-level data by county =======================================
def makeLookUpDict_ANSItoCOUNTYFP(state, yr=2020):
  """
  Generate a dictionary to convert ANSICODE to COUNTY

  Parameters
  ----------
  state : string holds state name of interest
  yr : int 

  Returns
  -------
  dictionary : ANSICODE -> COUNTYFP
  """
  url = f'https://www2.census.gov/geo/docs/maps-data/data/gazetteer/{yr}_Gazetteer/{yr}_gaz_cousubs_{get_STATE_FIPS_CODE(state)}.txt'
  tmp = requests.get(url)
  if not tmp.ok:
    print('Error: selected state and/or year data do not exist.')
    return 
  df_gaz = pd.read_csv(url,'\t')
  df_gaz = df_gaz.iloc[:,:4]
  df_gaz['ANSICODE'] = df_gaz['ANSICODE'].astype(str) 
  df_gaz['ANSICODE'] = df_gaz['ANSICODE'].str.zfill(8)
  df_gaz['GEOID'] = df_gaz['GEOID'].astype(str)
  # from GEOID, get county number and add county name
  df_gaz = df_gaz.drop_duplicates(subset=['ANSICODE']) # 1851

  lst = [''] * df_gaz.shape[0]
  for i in range(0, len(lst)):
    lst[i] = df_gaz.GEOID.iloc[i][2:5]

  df_gaz['COUNTYFP'] = lst
  return  dict(zip(df_gaz.ANSICODE, df_gaz.COUNTYFP))

def makeLookUpDict_COUNTYFPtoCOUNTYName(state):
  """
  Generate a dictionary to convert COUNTYFP to COUNTY NAME
  REF: https://www.census.gov/library/reference/code-lists/ansi.html#par_statelist
       - County Subdivision - 2010 ANSI Codes for County Subdivisions

  Parameters
  ----------
  state : string holds state name of interest

  Returns
  -------
  dictionary : COUNTYFP -> COUNTY NAME
  """
  url = f'https://www2.census.gov/geo/docs/reference/codes/files/st{get_STATE_FIPS_CODE(state)}_{str.lower(get_USPD(state))}_cousub.txt'
  tmp = requests.get(url)
  if not tmp.ok:
    print('Error: selected state and/or year data do not exist.')
    return 
  df_county = pd.read_csv(url, ',', names=['STATE','STATEFP','COUNTYFP','COUNTYNAME','COUSUBFP','COUSUBNAME', 'FUNCSTAT'])
  df_county['COUNTYFP'] = df_county['COUNTYFP'].astype(str)
  df_county['COUNTYFP'] = df_county['COUNTYFP'].str.zfill(3)
  df_county = df_county.drop_duplicates(subset=['COUNTYNAME'])
  return dict(zip(df_county.COUNTYFP, df_county.COUNTYNAME))