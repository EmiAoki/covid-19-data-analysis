# Code by Emi Aoki

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


def PCA(data_X):
  """
  Parameters
  ----------
  data_X : pandas dataframe of dimension N x K
    - row: observations
    - column: variables

  Returns
  -------
  k : int - the number of variables
  C : 2D numpy array - correlation matrix of dimension k x k
  w_sort : 1D numpy array - sorted eigenvalues of size (k, )
  v_sort : 2D numpy array - sorted eigenvectors of size (k,k)
  df_Znew : pandas dataframe if dimiension N x k - principal components
  """
  # 10/21/2021 error fixed:
  # https://stackoverflow.com/questions/40809503/python-numpy-typeerror-ufunc-isfinite-not-supported-for-the-input-types
  data_X_np = data_X.to_numpy()
  N = data_X_np.shape[0]
  k = data_X_np.shape[1]
  print('number of samples: N =', N, ', number of features k =', k)
  var_mean = data_X_np.mean(axis=0)
  var_var = data_X_np.var(axis=0)
  var_mean = np.array(var_mean, dtype=float)
  var_var = np.array(var_var, dtype=float)
  Z = (data_X_np - var_mean) / np.sqrt(var_var)
  #Z = (data_X_np - var_mean) # 11/03/2021 no np.sqrt(var_var)
  ZT = Z.T
  covZ = np.matmul(ZT,Z)
  C = covZ/(N); C = np.array(C, dtype=float)
  w,v = np.linalg.eig(C)
  idx = (-w).argsort()
  w_sort=np.zeros(w.shape) # (N,)
  w_sort=w[idx]
  v_sort=np.zeros(v.shape)
  v_sort=v[:,idx]
  print('eigenvalues: ', w_sort.shape)
  print('eigenvectors:', v_sort.shape)
  
  # in case
  w_sort = np.array(w_sort, dtype=float)
  v_sort = np.array(v_sort, dtype=float)
  # 
  Znew = np.zeros(Z.shape)
  Znew = np.matmul(Z,v_sort)
  df_Znew = pd.DataFrame(Znew, index=data_X.index, columns=['Comp'+str(i+1) for i in range(0,k)])
  return k, C, w_sort, v_sort, df_Znew


def PCA_noPreprocessing(data_X):
  """
  No preprocesssing is performed prior to PAC
  (data_X is NOT centered, also NOT scaled by the standard deviations)
  
  Parameters
  ----------
  data_X : pandas dataframe of dimension N x K
    - row: observations
    - column: variables

  Returns
  -------
  k : int - the number of variables
  C : 2D numpy array - correlation matrix of dimension k x k
  w_sort : 1D numpy array - sorted eigenvalues of size (k, )
  v_sort : 2D numpy array - sorted eigenvectors of size (k,k)
  df_Znew : pandas dataframe if dimiension N x k - principal components
  """
  # 10/21/2021 error fixed:
  # https://stackoverflow.com/questions/40809503/python-numpy-typeerror-ufunc-isfinite-not-supported-for-the-input-types
  data_X_np = data_X.to_numpy()
  N = data_X_np.shape[0]
  k = data_X_np.shape[1]
  print('number of samples: N =', N, ', number of features k =', k)
  var_mean = data_X_np.mean(axis=0)
  var_var = data_X_np.var(axis=0)
  var_mean = np.array(var_mean, dtype=float)
  var_var = np.array(var_var, dtype=float)
  #Z = (data_X_np - var_mean) / np.sqrt(var_var) # 12/02/2021 no centering
  Z = data_X_np
  #Z = (data_X_np - var_mean) # 11/03/2021 no np.sqrt(var_var)
  ZT = Z.T
  covZ = np.matmul(ZT,Z)
  C = covZ/(N); C = np.array(C, dtype=float)
  w,v = np.linalg.eig(C)
  idx = (-w).argsort()
  w_sort=np.zeros(w.shape) # (N,)
  w_sort=w[idx]
  v_sort=np.zeros(v.shape)
  v_sort=v[:,idx]
  print('eigenvalues: ', w_sort.shape)
  print('eigenvectors:', v_sort.shape)
  
  # in case
  w_sort = np.array(w_sort, dtype=float)
  v_sort = np.array(v_sort, dtype=float)
  # 
  Znew = np.zeros(Z.shape)
  Znew = np.matmul(Z,v_sort)
  df_Znew = pd.DataFrame(Znew, index=data_X.index, columns=['Comp'+str(i+1) for i in range(0,k)])
  return k, C, w_sort, v_sort, df_Znew


def PCA_notScaledByStd(data_X):
  """
  Prior to PCA, data_X is centered (subtract by means) but not scaled by the standard deviations
  
  Parameters
  ----------
  data_X : pandas dataframe of dimension N x K
    - row: observations
    - column: variables

  Returns
  -------
  k : int - the number of variables
  C : 2D numpy array - covariance matrix of dimension k x k
  w_sort : 1D numpy array - sorted eigenvalues of size (k, )
  v_sort : 2D numpy array - sorted eigenvectors of size (k,k)
  df_Znew : pandas dataframe if dimiension N x k - principal components
  """
  # 10/21/2021 error fixed:
  # https://stackoverflow.com/questions/40809503/python-numpy-typeerror-ufunc-isfinite-not-supported-for-the-input-types
  data_X_np = data_X.to_numpy()
  N = data_X_np.shape[0]
  k = data_X_np.shape[1]
  print('number of samples: N =', N, ', number of features k =', k)
  var_mean = data_X_np.mean(axis=0)
  var_mean = np.array(var_mean, dtype=float)
  Z = (data_X_np - var_mean) # 11/03/2021 no np.sqrt(var_var)
  ZT = Z.T
  covZ = np.matmul(ZT,Z)
  C = covZ/(N); C = np.array(C, dtype=float)
  w,v = np.linalg.eig(C)
  idx = (-w).argsort()
  w_sort=np.zeros(w.shape) # (N,)
  w_sort=w[idx]
  v_sort=np.zeros(v.shape)
  v_sort=v[:,idx]
  print('eigenvalues: ', w_sort.shape)
  print('eigenvectors:', v_sort.shape)
  
  # in case
  w_sort = np.array(w_sort, dtype=float)
  v_sort = np.array(v_sort, dtype=float)
  # 
  Znew = np.zeros(Z.shape)
  Znew = np.matmul(Z,v_sort)
  df_Znew = pd.DataFrame(Znew, index=data_X.index, columns=['Comp'+str(i+1) for i in range(0,k)])
  return k, C, w_sort, v_sort, df_Znew
  
 
  
  
  

def plot_CorrelationMatrix(Cnp, feat_lst):
  """
  Generate color map of correlation matrix
  Print correlation matrix as a table

  Parameters
  ----------
  Cnp : 2D numpy array (matrix)
  feat_lst : list or array contains variable names of data_X
  """
  plt.figure(figsize=(6,6))
  t1 = plt.matshow(Cnp, cmap=plt.cm.Blues, fignum=1) 
  # https://stackoverflow.com/questions/43021762/matplotlib-how-to-change-figsize-for-matshow
  plt.colorbar(t1, shrink=.8)
  plt.title('Correlation Matrix')
  plt.show()

  tmp = np.array(feat_lst)
  Ctmp = Cnp.copy()
  Ctmp = np.concatenate([tmp.reshape((tmp.shape[0],1)),Ctmp], axis=1)
  tbl = tabulate(Ctmp, headers=feat_lst)
  print('Correlation Matrix:')
  print(tbl)


def plot_eigenValues(w_sort, k):
  """
  Generate a scree plot - a line plot of eigenvalues again the correponding PC number
                          - shows how much variation each PC captures
  Print a table of eigenvalues, proportions and cumulative proportions obtained by PC

  Parameters
  ----------
  w_sort : 1-D numpy array holds sorted eigenvalues
  k : the number of eigenvalues
  
  Returns
  -------
  df_eigenVal : pandas dataframe - eigenvalues
    - row: ['component', 'eigenvalues', 'proportion', 'cumulative']
    - column: principal component
  """
  df_eigenVal = pd.DataFrame(columns=['component','eigenvalues','proportion','cumulative'])
  df_eigenVal['component'] = [i+1 for i in range(0,k)]
  df_eigenVal['eigenvalues']=w_sort.copy()

  ttlEig = w_sort.sum()
  cum = 0
  for i, val in enumerate(w_sort):
    df_eigenVal['proportion'].iloc[i] = val/ttlEig
    cum = cum + (val/ttlEig)
    df_eigenVal['cumulative'].iloc[i] = cum.copy()

  fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
  ax1.plot(df_eigenVal['component'], df_eigenVal['proportion'],'--o')
  ax1.plot(df_eigenVal['component'],df_eigenVal['cumulative'],'-o')
  ax1.set_title('Scree Plot', fontsize=14)
  ax1.set(xlabel='Component Number', ylabel='Proportion of Variance')
  ax1.legend(['Variance','Cumulative Variance'])
  ax1.set_xticks([i for i in range(df_eigenVal.shape[0]+1)])

  ax2.plot(df_eigenVal['component'],df_eigenVal['eigenvalues'],'-o')
  ax2.set_title('Scree Plot', fontsize=14)
  ax2.set(xlabel='Component Number', ylabel='Eigenvalue')
  ax2.set_xticks([i for i in range(1,df_eigenVal.shape[0]+1)])

  #plt.xticks(range(df_eigenVal.shape[0]))
  fig.show()
  tmp_np = df_eigenVal.to_numpy()
  tbl = tabulate(tmp_np, headers=df_eigenVal.columns)
  print('EigenValues:')
  print(tbl)
  return df_eigenVal


def print_eigenVec(v_sort, k, feat_lst):
  """
  Parameters
  ----------
  v_sort : 2D numpy array holds sorted eigenvalues
  k : the number of eigenvalues
  feat_lst : list or array holds feature names

  Returns
  -------
  df_eigenVec : pandas dataframe - eigenvectors
   - row: original variables
   - column: principal component
  """
  df_eigenVec = pd.DataFrame(v_sort, columns=['Comp'+str(i+1) for i in range(0,k)], index=feat_lst)
  tmpDF = df_eigenVec.reset_index()
  tmpDF.rename(columns={'index':''}, inplace=True)
  print('Eigenvectors:')
  #df_eigenVec
  tmp_np = tmpDF.to_numpy()
  tbl = tabulate(tmp_np, headers=tmpDF.columns)
  print(tbl)
  return df_eigenVec


def PCA_loadingPlot(df_eigenVal, df_eigenVec, data1='Comp1', data2='Comp2', data3='Comp3', singlePlot=False):
  """
  Generate a loading plot
    - a plot showing the relationship between principal components and the original variables
      - shows how variables correlate with one another
        - about 90 degree: no correlation
        - about 0 degree: positive correlation
        - about 180 degree: negative correlation

  Parameters
  ----------
  df_eigenVal : pandas dataframe holds sorted eigenvalues
  df_eigenVec : pandas dataframe holds sorted eigenvectors
  data1, data2, data3: loading plots of these principal component will be plotted (added on 12282021) - first 3 PCs by default

  Returns
  -------
  ax1 : matplotlib.axes._subplots.AxesSubplot - PC1 vs PC2
  ax2 : matplotlib.axes._subplots.AxesSubplot - PC1 vs PC3
  ax3 : matplotlib.axes._subplots.AxesSubplot - PC2 vs PC3
  """
  if not singlePlot:
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,4))
  # PC1 vs PC2 or 1st combination
  #data1 = 'Comp1'; data2 = 'Comp2'
  if singlePlot: 
    fig1, ax1 = plt.subplots(1,1)
  for idx, row in df_eigenVec.iterrows(): 
    ax1.arrow(0,0,row[data1],row[data2], ec='grey',)
    ax1.text(row[data1]+0.02, row[data2]-0.01, idx)
  ax1.scatter(df_eigenVec[data1], df_eigenVec[data2])
  ax1.set_xlim([-1.0, 1.0]); ax1.set_ylim([-1.0,1.0])
  ax1.set(xlabel=data1+' ({:.1f}'.format(df_eigenVal['proportion'].iloc[int(data1.split('Comp')[-1])-1] *100) +'%)',
          ylabel=data2+' ({:.1f}'.format(df_eigenVal['proportion'].iloc[int(data2.split('Comp')[-1])-1] *100) +'%)')
  ax1.axhline(y=0.0, color='gray', linestyle='--'); ax1.axvline(x=0.0, color='gray', linestyle='--')
  ax1.set_title('loading plot: ' + data1 + ' vs ' + data2, fontsize=14)
  ax1.set_xlabel(ax1.get_xlabel(), fontsize=12)
  ax1.set_ylabel(ax1.get_ylabel(), fontsize=12)

  # 2nd combination
  #data1 = 'Comp1'; data2 = 'Comp3'
  if singlePlot: 
    fig2, ax2 = plt.subplots(1,1)
  for idx, row in df_eigenVec.iterrows(): 
    ax2.arrow(0,0,row[data1],row[data3], ec='grey',)
    ax2.text(row[data1]+0.02, row[data3]-0.01, idx)
  ax2.scatter(df_eigenVec[data1], df_eigenVec[data3])
  ax2.set_xlim([-1.0, 1.0]); ax2.set_ylim([-1.0,1.0])
  ax2.set(xlabel=data1+' ({:.1f}'.format(df_eigenVal['proportion'].iloc[int(data1.split('Comp')[-1])-1] *100) +'%)',
          ylabel=data3+' ({:.1f}'.format(df_eigenVal['proportion'].iloc[int(data3.split('Comp')[-1])-1] *100) +'%)')
  ax2.axhline(y=0.0, color='gray', linestyle='--'); ax2.axvline(x=0.0, color='gray', linestyle='--')
  ax2.set_title('loading plot: ' + data1 + ' vs ' + data3, fontsize=14)
  ax2.set_xlabel(ax2.get_xlabel(), fontsize=12)
  ax2.set_ylabel(ax2.get_ylabel(), fontsize=12)

  # 3rd combination    
  #data1 = 'Comp2'; data2 ='Comp3'
  if singlePlot: 
    fig3, ax3 =plt.subplots(1,1)
  for idx, row in df_eigenVec.iterrows(): 
    ax3.arrow(0,0,row[data2],row[data3], ec='grey',)
    ax3.text(row[data2]+0.02, row[data3]-0.01, idx)
  ax3.scatter(df_eigenVec[data2], df_eigenVec[data3])
  ax3.set_xlim([-1.0, 1.0]); ax3.set_ylim([-1.0,1.0])
  ax3.set(xlabel=data2+' ({:.1f}'.format(df_eigenVal['proportion'].iloc[int(data2.split('Comp')[-1])-1] *100) +'%)',
          ylabel=data3+' ({:.1f}'.format(df_eigenVal['proportion'].iloc[int(data3.split('Comp')[-1])-1] *100) +'%)')
  ax3.axhline(y=0.0, color='gray', linestyle='--'); ax3.axvline(x=0.0, color='gray', linestyle='--')
  ax3.set_title('loading plot: ' + data2 + ' vs ' + data3, fontsize=14)
  ax3.set_xlabel(ax3.get_xlabel(), fontsize=12)
  ax3.set_ylabel(ax3.get_ylabel(), fontsize=12)
  if singlePlot:
    return (fig1, ax1), (fig2, ax2), (fig3,ax3)
  return fig, (ax1, ax2, ax3)





def PCA_scorePlot2D(df_Z, xax_var='Comp1', yax_var='Comp2', anno=True):
  """
  Generate a score plot - a plot projecting of the observations onto the PCs in 2D

  Parameters
  ----------
  df_Z : pandas dataframe
  xax_var : principal component on x-axis; 'Comp1' by default
  yax_var : principal component on y-axis; 'Comp2' by default

  Returns
  -------
  (fig, ax) : tuple (figure, matplotlib.axes._subplots.AxesSubplot)
  """
  fig = plt.figure()
  ax = fig.add_subplot(111)
  if anno:
      for idx, row in df_Z.iterrows():
        ax.text(row[xax_var], row[yax_var], idx)
  ax.scatter(df_Z[xax_var], df_Z[yax_var])
  ax.axhline(y=0.0, color='gray', linestyle='--'); ax.axvline(x=0.0, color='gray', linestyle='--')
  plt.title('Score Plot: ' + xax_var + ' vs ' + yax_var, fontsize=14)
  ax.set_xlabel(xax_var, fontsize=12); ax.set_ylabel(yax_var, fontsize=12)
  return (fig, ax)


def PCA_scorePlot3D(df_Z, xax_var='Comp1', yax_var='Comp2', zax_var='Comp3'):
  """
  Generate a score plot - a plot projecting of the observations onto the PCs in 3D

  Parameters
  ----------
  df_Z : pandas dataframe
  xax_var : principal component on x-axis; 'Comp1' by default
  yax_var : principal component on y-axis; 'Comp2' by default
  zax_var : principal component on z-axis; 'Comp3' by default

  Returns
  -------
  (fig, ax) : tuple (figure, matplotlib.axes._subplots.Axes3DSubplot)
  """
  df_Z = df_Z.reset_index()
  fig = plt.figure(figsize=(8,7))
  ax = fig.add_subplot(111, projection='3d')
  for idx, row in df_Z.iterrows():
    ax.text(row[xax_var], row[yax_var], row[zax_var], idx)
  ax.scatter(xs=df_Z[xax_var], ys=df_Z[yax_var], zs=df_Z[zax_var])
  #ax.axhline(y=0.0, color='gray', linestyle='--'); ax.axvline(x=0.0, color='gray', linestyle='--')
  plt.title('Score Plot: ' + xax_var + ' vs ' + yax_var + ' vs ' + zax_var, fontsize=14)
  ax.set_xlabel(xax_var, fontsize=12); ax.set_ylabel(yax_var, fontsize=12); ax.set_zlabel(zax_var, fontsize=12)
  return (fig, ax)


def PCA_biplot_scoreNormalized(score, eigenvec, xax='Comp1', yax='Comp2'):
  """
  Generate a biplot - combination of loadings and scores
  Scores are normalized to have the range of -1 and 1
  ref: https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot

  Parameters
  ----------
  score : result of PCA - pandas dataframe
  eigenvec: eigenvectors - pandas dataframe (columns: PCs, rows: original features)
  xax: principal component on x-axis: 'Comp1' by default
  yax: principal component on y-axis: 'Comp2' by default

  Returns
  -------
  None
  """
  xs = score[xax]
  ys = score[yax]
  n_eigenvec = eigenvec.shape[0]

  xs_min = xs.min(); xs_max = xs.max()
  #xs = (xs - xs_min)/(xs_max - xs_min)
  xs = xs/(xs_max - xs_min)
  ys_min = ys.min(); ys_max = ys.max()
  #ys = (ys - ys_min)/(ys_max - ys_min)
  ys = ys/(ys_max - ys_min)
  plt.scatter(xs, ys, color='skyblue')

  for i in range(n_eigenvec):
    plt.arrow(0,0,eigenvec[xax][i], eigenvec[yax][i], color='grey')
    plt.text(eigenvec[xax][i]+0.01, eigenvec[yax][i], eigenvec.index[i], fontsize=10)
  plt.scatter(eigenvec[xax],eigenvec[yax],marker='x', s=30, color='red')
  plt.axvline(0, linestyle='--', color='grey')
  plt.axhline(0, linestyle='--', color='grey')
  plt.xlabel(xax)
  plt.ylabel(yax)
  plt.title(f'Biplot: {xax} vs {yax}')
  plt.show()