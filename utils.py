import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import PredefinedSplit
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
import seaborn as sns
from scipy.signal import butter,filtfilt
from scipy.fft import fft, fftfreq

def plot_freq(df, channel='channel3'):
    # Number of samples in normalized_tone
    N = len(df[channel]) #SAMPLE_RATE * DURATION

    sample_rate = N/(df['time'].max()/1000)
    yf = fft(np.array(df[channel]))
    xf = fftfreq(N, 1 / sample_rate)

    i_max = np.argmax(yf)
    x_max = xf[i_max]
    print(x_max)
    plt.figure(figsize=(16,9))
    plt.plot(xf, np.abs(yf))
    plt.ylabel('Amplitud')
    plt.xlabel('Frecuencia (Hz)')
    plt.show()

def plot_channel(dataframe, channel='channel1'):
  fig, axis = plt.subplots(2, figsize = (12,12))
  plt.title(channel)
  axis[0].scatter(dataframe['time'], dataframe[channel], s=20, c=dataframe['class'], cmap='rainbow')
  axis[1].plot(dataframe['time'], dataframe[channel], color='black')
  plt.xlabel('t[ms]')
  plt.show()

def read_data(path):
    '''
    path: Carpeta en donde se encuentran todas las carpetas con la data de los sujetos.
    '''
    d_read = lambda d : pd.read_csv(d, sep='\t', header = 0)
    path_train = path +'/subj{}'
    common_names = ['1.txt', '2.txt']

    # Juntar todos los datos
    #df_data = pd.DataFrame()
    data_list = []
    for i in range(1, 31):
        sub_n = str(i).zfill(2) # Número del sujeto
        folder = path_train.format(sub_n) # Ruta de la carpeta
        for file in common_names:
            df_file = d_read(f'{folder}/{file}')
            df_file['subject'] = i
            df_file['capture'] = int(file[0])

            data_list.append(df_file)
            #df_data = pd.concat([df_data, df_file], ignore_index=True)
    return data_list

def read_test_data(path_test):

    df_test = pd.read_csv(path_test, sep=",", header=0)

    # Almacenar ventanas 800x8 en un arreglo de numpy
    test = np.array(df_test.drop('Id', axis=1)).reshape(672, 8, 800).transpose((0, 2, 1))
    
    return test

def butter_filter(data, cutoff, fs, order, btype = 'low'):
    nyq = 0.5 * fs
    normal_cutoff = np.array(cutoff) / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    y = filtfilt(b, a, data)
    return y

def substract_mean(dataframe, columns):
  df_copy = dataframe.copy(True)
  for x in columns:
    df_copy[x] = dataframe[x] - dataframe[x].mean()
  return df_copy


def filtrar_df(df, fs=970, cutoffs = 350, btype='low', channels = [f'channel{i}' for i in range(1,9)]):
  df_copy = df.copy(True)
  for ch in channels:
        y = butter_filter(df_copy[ch], cutoffs, fs, order = 2, btype=btype)
        df_copy[ch] = y
  return df_copy


def create_windows(dataframe, window_size = 800, step = 250):
  '''
  Crea ventanas de tamaño (window_size, n_columns) a partir de un dataframe.
  El número de ventanas depende del largo del dataframe y el step utilizado.
  Las ventanas son dataframes, y se retorna una lista con estos.

  Inputs:
  - dataframe: Dataframe de pandas
  '''
  windows = []
  start = 0
  while start + window_size < len(dataframe):
      window = dataframe[start:start+800]
      windows.append(window)
      start += step
  return windows

def emg_windows(df_list, window_size = 800, step = 250):
  classes = np.arange(1, 7)
  windows = []
  labels = []
  for df in df_list:
    for cl in classes:
      temp_window = create_windows(df[df['class']==cl].drop('class', axis=1), window_size, step)
      windows+=temp_window
      labels += [cl]*len(temp_window)
  return windows, labels

def train_val_split(df_list, train_sub, val_sub):
  train_list = []
  val_list = []
  for df in df_list:
    if df['subject'].unique() in train_sub:
      train_list.append(df)
    elif df['subject'].unique() in val_sub:
      val_list.append(df)
  return train_list, val_list

def window_to_features(windows_list, features):
  data = []
  for window in windows_list:
    temp = window.agg(features, axis=0).values.flatten('F') # arreglo 1d de largo n_channels*n_features
    data.append(temp)
  return np.array(data)  

def multitrain(grid_dict, x_cv, y_cv):
    '''
    grid_list : Diccionario con grillas listas para el entrenamiento
    x_cv: data de entrenamiento y validación concatenada
    y_cv: labels de entrenamiento y validación concatenados
    '''
    trained_dict = {}
    for grid in grid_dict:
        grid_dict[grid].fit(x_cv, y_cv)
        grid_obj = grid_dict[grid]
        trained_dict[grid] = [grid_obj.best_estimator_, grid_obj.best_params_] 

    return trained_dict


def show_matrixes(y_true, y_predicted, model=''):

  acc = metrics.accuracy_score(y_true, y_predicted)
  print(f'Accuracy = {round(acc, 3)}')
  fig, axs = plt.subplots(1, 1)
  fig.set_size_inches(8, 6)
  axs.set_title('Matriz de confusión (Normalizada)')
  metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_predicted, normalize='true', ax=axs)

  if model!='':
    fig.suptitle(model)
  fig.show()
  return acc


class TrainWrapper():
    '''
    Clase que reune métodos para el procesamiento de datos para su uso en el entrenamiento.
    '''
    def __init__(self, df_list):
        '''
        Constructor: Recibe una lista de dataframes procesados para ser utilizados en el proyecto.
        '''
        self.dfs = df_list
        self.df_train = None
        self.df_val = None

        self.train_windows = None
        self.val_windows = None

        self.y_train = None
        self.y_val = None

        self.x_train = None
        self.x_val = None
        self.scaler = None

        self.x_tv = None
        self.y_tv = None
        self.cv = None

    def split(self, train_sub, val_sub):
        '''
        Recibe listas con los sujetos de entrenamiento y validación.
        Crea los splits de entrenamiento y validación en la lista inicial de dataframes
        '''
        self.df_train, self.df_val = train_val_split(self.dfs, train_sub, val_sub) 
    
    def make_windows(self, window_size = 800, step = 250):
        '''
        Se crean las ventanas de entrenamiento y validación a partir de las listas respectivas.
        '''
        if self.df_train!=None and self.df_val!=None:
            self.train_windows, self.y_train = emg_windows(self.df_train, window_size, step)
            self.val_windows, self.y_val = emg_windows(self.df_val, window_size, step)

        else: 
            print('Primero debes crear un split de entrenamiento/validación (método split)')
    
    def compute_features(self, feature_list, scaler = MinMaxScaler()):
        '''
        Convierte las ventanas en data apta para entrenamiento a partir de
        la lista de caracteristicas a computar.

        feature_list: Lista de caracteristicas (debe ser apta para un pandas.agg)
        scaler: El scaler a usar en los datos
        '''
        if self.train_windows!=None and self.val_windows!=None:
            to_drop = ['subject', 'capture', 'time']
            self.x_train = [x.drop(to_drop, axis=1) for x in self.train_windows]
            self.x_val = [x.drop(to_drop, axis=1) for x in self.val_windows]

            self.x_train = window_to_features(self.x_train, feature_list)
            self.x_val = window_to_features(self.x_val, feature_list)

            #scal = scaler()
            scaler.fit(self.x_train)
            self.scaler = scaler
            self.x_train = scaler.transform(self.x_train)
            self.x_val = scaler.transform(self.x_val)
        else:
            print('Primero debes crear ventanas de entrenamiento/validación (método make_windows)')
        
    def make_test_folds(self):
        self.x_tv = np.concatenate([self.x_train, self.x_val])
        self.y_tv = np.concatenate([self.y_train, self.y_val])
        test_fold = np.concatenate([
                                    np.full(self.x_train.shape[0], -1), 
                                    np.zeros(self.x_val.shape[0])])
        self.cv = PredefinedSplit(test_fold)

    


    
