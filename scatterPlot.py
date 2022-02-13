# Code by Emi Aoki

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns

# 2D plots =====================================================================
def get_scatterPlot2d_singlePlot(df, var_lst, x_var='DENS'):
  """
  Generate scatter plots

  Parameters
  ----------
  df : pandas dataframe
   - row: observations
   - column: variables
  var_lst : list or array holds variable name; plotted on y-axis
  x_var : string holds a variable name; plotted on x-axis
  """
  for var in var_lst:
    plt.scatter(df[x_var], df[var])
    plt.xlabel(x_var, fontsize=13)
    plt.ylabel(var, fontsize=13)
    plt.show()

def get_scatterPlot2d_singlePlot_Anno(df, var_lst, x_var='DENS'):
  """
  Generate scatter plots with annotation

  Parameters
  ----------
  df : pandas dataframe
   - row: observations
   - column: variables
  var_lst : list or array holds variable name; plotted on y-axis
  x_var : string holds a variable name; plotted on x-axis
  """
  for var in var_lst:
    for idx, row in df.iterrows():
      plt.text(row[x_var], row[var], idx)
    plt.scatter(df[x_var], df[var])
    plt.xlabel(x_var, fontsize=13)
    plt.ylabel(var, fontsize=13)
    plt.show()


def get_scatterPlot2d_together(df, var_lst, x_var='+1W_mean', yax_ttl='Population Proportion', xax_ttl='COVID cases', ttl='Scatter Plot'):
  """
  Generate a scatter plot - multiple variables will be plotted together

  Parameters
  ----------
  df : pandas dataframe
   - row: observations
   - column: variables
  var_lst : list or array holds variable name; plotted on y-axis
  x_var : string holds a variable name; plotted on x-axis
  yax_ttl : string holds y-axis title
  xax_ttl : string holds x-axis title
  ttl : string holds plot title
  """
  fig = plt.figure()
  ax = fig.add_subplot(111)
  colors_lst = [v for k,v in mcolors.TABLEAU_COLORS.items()]
  marker_lst = [i for i in Line2D.markers][1:-4]
  for i, var in enumerate(var_lst):
    ax.scatter(x=df[x_var], y=df[var], color=colors_lst[int(i%len(colors_lst))], marker=marker_lst[int(i%len(marker_lst))], label=var)
  plt.legend(loc='upper right')
  plt.xlabel(xax_ttl, fontsize=13)
  plt.ylabel(yax_ttl, fontsize=13)
  plt.title(ttl)
  plt.show()


def get_scatterPlot2d_together_Anno(df, var_lst, x_var='+1W_mean', yax_ttl='Population Proportion', xax_ttl='COVID cases', ttl='Scatter Plot'):
  """
  Generate a scatter plot with annotation - multiple variables will be plotted together

  Parameters
  ----------
  df : pandas dataframe
   - row: observations
   - column: variables
  var_lst : list or array holds variable name; plotted on y-axis
  x_var : string holds a variable name; plotted on x-axis
  yax_ttl : string holds y-axis title
  xax_ttl : string holds x-axis title
  ttl : string holds plot title
  """
  fig = plt.figure()
  ax = fig.add_subplot(111)
  colors_lst = [v for k,v in mcolors.TABLEAU_COLORS.items()]
  marker_lst = [i for i in Line2D.markers][1:-4]
  for i, var in enumerate(var_lst):
    for idx, row in df.iterrows():
      plt.text(row[x_var], row[var], idx)
    ax.scatter(x=df[x_var], y=df[var], color=colors_lst[int(i%len(colors_lst))], marker=marker_lst[int(i%len(marker_lst))], label=var)
  plt.legend(loc='upper right')
  plt.xlabel(xax_ttl, fontsize=13)
  plt.ylabel(yax_ttl, fontsize=13)
  plt.title(ttl)
  plt.show()


# 3D plots =====================================================================
def get_scatterPlot3d_together(df, var_lst, z_var='+1W_mean', y_var='FW', yax_ttl='Frontline Workers', zax_ttl='COVID cases', xax_ttl ='Population Proportion', ttl='Scatter Plot'):
  """
  Generate a scatter plot 3D

  Parameters
  ----------
  df : pandas dataframe
   - row: observations
   - column: variables
  var_lst : list or array holds variable name; plotted on x-axis
  z_var : string holds a variable name; plotted on z-axis
  y_var : string holds a variable name; plotted on y-axis
  yax_ttl : string holds y-axis title
  zax_ttl : string holds z-axis title
  xax_ttl : string holds x-axis title
  ttl : string holds plot title
  """
  fig=plt.figure(figsize=(8,7))
  ax = fig.add_subplot(111, projection='3d')
  colors_lst = [v for k,v in mcolors.TABLEAU_COLORS.items()]
  marker_lst = [i for i in Line2D.markers][1:-4]
  for i, var in enumerate(var_lst):
    ax.scatter(xs=df[var], ys=df[y_var], zs=df[z_var], color=colors_lst[int(i%len(colors_lst))], marker=marker_lst[int(i%len(marker_lst))], label=var)
  ax.set_xlabel(xax_ttl, fontsize=12)
  ax.set_ylabel(yax_ttl, fontsize=12)
  ax.set_zlabel(zax_ttl, fontsize=12)
  ax.legend(var_lst, loc='upper left', fontsize=10)
  ax.yaxis.labelpad=10
  ax.xaxis.labelpad=10
  ax.zaxis.labelpad=5
  plt.title(ttl, loc='left', fontsize=14)
  #plt.show()
  return ax # this allows for modifications on plots such as ax.set_xlim() etc

def get_scatterPlot3d_together_Anno(df, var_lst, z_var='+1W_mean', y_var='FW', yax_ttl='Frontline Workers', zax_ttl='COVID cases', xax_ttl ='Population Proportion', ttl='Scatter Plot'):
  """
  Generate a scatter plot 3D with annotation

  Parameters
  ----------
  df : pandas dataframe
   - row: observations
   - column: variables
  var_lst : list or array holds variable name; plotted on x-axis
  z_var : string holds a variable name; plotted on z-axis
  y_var : string holds a variable name; plotted on y-axis
  yax_ttl : string holds y-axis title
  zax_ttl : string holds z-axis title
  xax_ttl : string holds x-axis title
  ttl : string holds plot title
  """
  fig=plt.figure(figsize=(8,7))
  ax = fig.add_subplot(111, projection='3d')
  colors_lst = [v for k,v in mcolors.TABLEAU_COLORS.items()]
  marker_lst = [i for i in Line2D.markers][1:-4]
  for i, var in enumerate(var_lst):
    for idx, row in df.iterrows():
      ax.text(row[var], row[y_var], row[z_var], idx)
    ax.scatter(xs=df[var], ys=df[y_var], zs=df[z_var], color=colors_lst[int(i%len(colors_lst))], marker=marker_lst[int(i%len(marker_lst))], label=var)
  ax.set_xlabel(xax_ttl, fontsize=12)
  ax.set_ylabel(yax_ttl, fontsize=12)
  ax.set_zlabel(zax_ttl, fontsize=12)
  ax.legend(var_lst, loc='upper left', fontsize=10)
  ax.yaxis.labelpad=10
  ax.xaxis.labelpad=10
  ax.zaxis.labelpad=5
  plt.title(ttl, loc='left', fontsize=14)
  #plt.show()
  return ax # this allows for modifications on plots such as ax.set_xlim() etc


# pairwise 2D scatter plot =====================================================

# REF: https://stackoverflow.com/questions/34087126/plot-lower-triangle-in-a-seaborn-pairgrid
# REF: https://stackoverflow.com/questions/59212378/how-do-i-get-the-diagonal-of-sns-pairplot
def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)
def sns_pairPlot(df, onlyLower=True):
  """
  generate seaborn.pairplot() - plots of pairwise relationships in a dataset

  Parameters
  ----------
  df : pandas dataframe
   - row: observations
   - column: variables
  """
  g=sns.pairplot(df, diag_kind='hist')
  if onlyLower: g.map_upper(hide_current_axis)
  return g 