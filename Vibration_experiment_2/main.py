#import
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
# from sklearn.svm import SVC
# from keras import models
# from keras import layers
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# from sklearn import preprocessing
# from sklearn.feature_selection import SelectKBest, f_regression, f_classif, SelectPercentile
# from sklearn.pipeline import make_pipeline
# from sklearn.decomposition import PCA
# import pywt
from statsmodels import robust
# import tensorflow as tf
# from scipy.fft import fft, fftfreq
import joblib
import json

# def load_data(filepath, sample_rate, duration):
#     hop_length = sample_rate * duration
#     data = np.loadtxt(fname=filepath, delimiter=',')
#     data = np.delete(data,(3), axis = 1)
#     data = np.array([
#         data[idx:idx + hop_length] for idx in range(0, len(data), hop_length)
#     ])
#     return data
water_data = []
nowater_data = []
if __name__ == '__main__':

    water_filepath = open('./wastewaterump1_vibration/vibration8-2021-09/sensor_vibration8_2021-09-11-deal.json')
    water_json = json.load(water_filepath)
    for t in range(len(water_json)):
        water_data.append(water_json[t]['spectrum'])
    water_dataF = np.array(water_data)

    nowater_filepath = open('./test_ab_vibration.json')
    nowater_json = json.load(nowater_filepath)
    for t in range(len(nowater_json)):
        nowater_data.append(nowater_json[t]['spectrum'])
    nowater_dataF = np.array(nowater_data)
    print("Normal_data:"+str(water_dataF.shape))
    print("Abnormal_data:"+str(nowater_dataF.shape))

    sample_rate = 50
    duration = 30  
    test_size = 0.2
    random_seed = 11
    fft_size = 128
    wavelet = "db4"

    np.random.seed(seed=random_seed)
    random.seed(a=random_seed)

    # water_data = load_data(filepath=water_filepath,
    #                        sample_rate=sample_rate,
    #                        duration=duration)
    # nowater_data = load_data(filepath=nowater_filepath,
    #                          sample_rate=sample_rate,
    #                          duration=duration)

#分析svm--------------------------------------------------------------------------------------   


    #split data
    data = np.vstack([water_dataF, nowater_dataF])
    sr=1000
    cD_std = []
    cD_mad = []
    num = len(data)
    print("Total_Data:"+str(data.shape))
    cD_std = np.std(data, axis=1)
    cD_mad = robust.mad(data, axis=1)
    cD_data = np.stack((cD_std,cD_mad), axis=1).reshape(int(num), 2)
    print(cD_data.shape)
    label = np.array([0] * len(water_dataF) + [1] * len(nowater_dataF))
    x_train, x_test, y_train, y_test = train_test_split(cD_data,
                                                        label,
                                                        test_size=test_size)

    # svm = SVC(random_state=random_seed)
    # svm.fit(X=x_train, y=y_train)
    # train_score = svm.score(X=x_train, y=y_train)
    # joblib.dump(svm,'./svm.pkl')


    model = joblib.load('./svm.pkl')
    test_score = model.score(X=x_test, y=y_test)
    y_pred = model.predict(x_test)
    print(f"test_accuracy: {test_score}")

