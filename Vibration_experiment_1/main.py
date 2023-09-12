#import
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pywt
from statsmodels import robust
import tensorflow as tf
import joblib

#def
def load_data(filepath, sample_rate, duration):
    hop_length = sample_rate * duration
    data = np.loadtxt(fname=filepath, delimiter=',')
    data = np.delete(data,(3), axis = 1)
    data = np.array([
        data[idx:idx + hop_length] for idx in range(0, len(data), hop_length)
    ])
    return data

def plot_confusion_matrix(y_true, y_pred, title_name):
    cm = confusion_matrix(y_true, y_pred)#混淆矩阵
    #annot = True 格上显示数字 ，fmt：显示数字的格式控制
    ax = sns.heatmap(cm,annot=True,fmt='g',xticklabels=['0', '1'],yticklabels=['0', '1'])
    #xticklabels、yticklabels指定横纵轴标签
    ax.set_title(title_name) #标题
    ax.set_xlabel('predict') #x轴
    ax.set_ylabel('true') #y轴
    # plt.savefig(f"D:/Manufacturer_data/vibration/拆分/dataset_20220127/keras.png")
    plt.savefig(f"./svm_wavelet.png")
    # plt.show()

def plot_time_data(water_data, nowater_data, sample_rate):
    plt.figure(figsize=(16, 9))
    for idx, (subplot,
              title) in enumerate(zip([311, 312, 313], ['x', 'y', 'z'])):
        plt.subplot(subplot)
        title += f'/nwater statistical: mean: {water_data[:,:,idx].mean()}, std: {water_data[:,:,idx].std()}, min: {water_data[:,:,idx].min()}, max: {water_data[:,:,idx].max()}/nnowater statistical: mean: {nowater_data[:,:,idx].mean()}, std: {nowater_data[:,:,idx].std()}, min: {nowater_data[:,:,idx].min()}, max: {nowater_data[:,:,idx].max()}'
        plt.title(title)
        plt.plot([], [], label='red line is water data', color='red')
        plt.plot([], [], label='blue line is nowater data', color='blue')
        for v in water_data:
            time = np.linspace(0, len(v[:, 2]), len(v[:, 2])) / sample_rate
            plt.plot(time, v[:, idx], color='red')
        for v in nowater_data:
            time = np.linspace(0, len(v[:, 2]), len(v[:, 2])) / sample_rate
            plt.plot(time, v[:, idx], color='blue')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('g')
    plt.tight_layout()
    plt.show()


def plot_frequency_data(water_data, nowater_data, sample_rate, fft_size):
    plt.figure(figsize=(16, 9))
    for idx, (subplot,
              title) in enumerate(zip([311, 312, 313], ['x', 'y', 'z'])):
        plt.subplot(subplot) 

        plt.plot([],[], label='red line is water data', color='red')
        plt.plot([],[], label='blue line is nowater data', color='blue')

        # freqs = np.linspace(0, int(50/2), int(fft_size/2))
        freqs = np.linspace(0, sample_rate/2, 751)

        water_adjust = water_data[0,:, idx]-np.mean(water_data[0,:, idx], axis=0)
        print(water_adjust.shape)
        water_audio_fft = np.fft.rfft(water_adjust) #/fft_size
        plt.plot(freqs, np.abs(water_audio_fft), color='red')

        nowater_adjust = nowater_data[0,:, idx]-np.mean(nowater_data[0,:, idx], axis=0)
        nowater_audio_fft = np.fft.rfft(nowater_adjust) #/fft_size
        plt.plot(freqs, np.abs(nowater_audio_fft), color='blue')

        plt.legend()
        plt.title(title)
        plt.ylabel('Amplitude')
        plt.xlabel('Frequency(Hz)')
    plt.tight_layout()
    plt.show()


def plot_cD_data(water_data, nowater_data, sample_rate, wavelet):

    plt.figure(figsize=(16, 9))
    for idx, (subplot,
              title) in enumerate(zip([311, 312, 313], [wavelet+'_x', wavelet+'_y', wavelet+'_z'])):
        plt.subplot(subplot) 
    
        water_data_cA, water_data_cD = pywt.dwt(water_data[0, :, idx], wavelet)
        water_cD_mean = np.mean(water_data_cD)
        water_cD_std = np.std(water_data_cD)
        water_cD_mad = robust.mad(water_data_cD)
        water_y = pywt.idwt(None, water_data_cD, wavelet)
        # water_y = pywt.idwt(water_data_cA, None, wavelet)

        nowater_data_cA, nowater_data_cD = pywt.dwt(nowater_data[0, :, idx], wavelet)
        nowater_cD_mean = np.mean(nowater_data_cD)
        nowater_cD_std = np.std(nowater_data_cD)
        nowater_cD_mad = robust.mad(nowater_data_cD)
        nowater_y = pywt.idwt(None, nowater_data_cD, wavelet)
        # nowater_y = pywt.idwt(nowater_data_cA, None, wavelet)

        plt.plot([],[], label='red line is water data', color='red')
        plt.plot([],[], label='blue line is nowater data', color='blue')

        # freqs = np.linspace(0, int(50/2), int(fft_size/2))
        # freqs = np.linspace(0, sample_rate/2, 751)
        title += f'/nwater statistical: mean: {water_cD_mean}, std: {water_cD_std}, mad: {water_cD_mad}/nnowater statistical: mean: {nowater_cD_mean}, std: {nowater_cD_std}, mad: {nowater_cD_mad}'
        time = np.linspace(0, 753, 753)

        # water_adjust = water_y[0,:]-np.mean(water_y[0,:], axis=0)
        # water_audio_fft = np.fft.rfft(water_adjust) #/fft_size
        plt.plot(time, water_data_cD, color='red')

        # nowater_adjust = nowater_y[0,:]-np.mean(nowater_y[0,:], axis=0)
        # nowater_audio_fft = np.fft.rfft(nowater_adjust) #/fft_size
        plt.plot(time, nowater_data_cD, color='blue')

        plt.legend()
        plt.title(title)
        # plt.ylabel('Amplitude')
        plt.xlabel('Sample')
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    #parameters
    water_filepath = './water.txt'
    nowater_filepath = './nowater.txt'
    sample_rate = 50
    duration = 30  #seconds
    test_size = 0.2
    random_seed = 11
    fft_size = 128
    wavelet = "db4"

    #set random seed
    np.random.seed(seed=random_seed)
    random.seed(a=random_seed)

    #load data  (x, y, z, time)
    water_data = load_data(filepath=water_filepath,
                           sample_rate=sample_rate,
                           duration=duration)
    # print(water_data.shape)
    nowater_data = load_data(filepath=nowater_filepath,
                             sample_rate=sample_rate,
                             duration=duration)

    #display x, y and z
    # plot_time_data(water_data=water_data,
    #           nowater_data=nowater_data,
    #           sample_rate=sample_rate)

    # plot_frequency_data(water_data=water_data,
    #           nowater_data=nowater_data,
    #           sample_rate=sample_rate, 
    #           fft_size=fft_size)

    # plot_cD_data(water_data=water_data,
    #         nowater_data=nowater_data,
    #         sample_rate=sample_rate, 
    #         wavelet = wavelet)

#離散小波轉換分析svm--------------------------------------------------------------------------------------   

    #資料處理
    #split data
    data = np.vstack([water_data, nowater_data]) #垂直推疊
    t=30
    sr=50
    for key, idx in zip(['x', 'y', 'z'], range(3)):
        if key == 'x':
            cA, cD = pywt.dwt(data[..., idx], wavelet)
            #    print(cA.shape)#低頻
            #    print(cD.shape)#高頻
            cD_mean = np.mean(cD, axis=1)
            cD_std = np.std(cD, axis=1)
            cD_mad = robust.mad(cD, axis=1)

            # print("===========================")
            # test1 = cD_mad[0]
            # print(test1)

            cD_stack = np.stack((cD_mean, cD_std, cD_mad), axis=1).reshape((100, 3))
            cD_stack_x = tf.expand_dims(cD_stack, 1)

        if key == 'y':
            cA, cD = pywt.dwt(data[..., idx], wavelet)
            #    print(cA.shape)#低頻
            #    print(cD.shape)#高頻
            cD_mean = np.mean(cD, axis=1)
            cD_std = np.std(cD, axis=1)
            cD_mad = robust.mad(cD, axis=1)

            # test2 = cD_mad[0]
            # print(test2)

            cD_stack = np.stack((cD_mean, cD_std, cD_mad), axis=1).reshape((100, 3))
            cD_stack_y = tf.expand_dims(cD_stack, 1)

        if key == 'z':
            cA, cD = pywt.dwt(data[..., idx], wavelet)
            #    print(cA.shape)#低頻
            #    print(cD.shape)#高頻
            cD_mean = np.mean(cD, axis=1)
            cD_std = np.std(cD, axis=1)
            cD_mad = robust.mad(cD, axis=1)

            # test3 = cD_mad[0]
            # print(test3)

            cD_stack = np.stack((cD_mean, cD_std, cD_mad), axis=1).reshape((100, 3))
            cD_stack_z = tf.expand_dims(cD_stack, 1)

    cD_data = np.concatenate((cD_stack_x, cD_stack_y, cD_stack_z), axis=1)
    cD_data = cD_data.reshape((100, 3*3))

    label = np.array([0] * len(water_data) + [1] * len(nowater_data))
    x_train, x_test, y_train, y_test = train_test_split(cD_data,
                                                        label,
                                                        test_size=test_size)
    #訓練
    svm = SVC(random_state=random_seed)
    svm.fit(X=x_train, y=y_train)
    train_score = svm.score(X=x_train, y=y_train)
    joblib.dump(svm,'./svm.pkl')

    #讀模型權重，做測試
    model = joblib.load('./svm.pkl')

    test_score = model.score(X=x_test, y=y_test)

    y_pred = model.predict(x_test)
    plot= plot_confusion_matrix(y_test, y_pred,'confusion matrix')

    #display result
    print(f"train accuracy: {train_score}, test_accuracy: {test_score}")

#deep learning--------------------------------------------------------------------------------------

    # x_train_mean = np.mean(x_train)
    # x_test_mean = np.mean(x_test)
    # x_train_std = np.std(x_train)
    # x_test_std = np.std(x_test)
    # x_train = (x_train-x_train_mean)/x_train_std
    # x_test = (x_test-x_test_mean)/x_test_std

    # model = models.Sequential()
    # model.add(layers.Dense(32, activation='relu', input_shape=(1500, 3)))
    # model.add(layers.Dense(16, activation='relu'))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(1, activation='sigmoid'))

    # model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

    # history = model.fit(x_train,
    #             y_train,
    #             epochs=30,
    #             batch_size=16)

    # y_pred = model.predict(x_test)
    # # plot= plot_confusion_matrix(y_test, np.round(abs(y_pred)),'confusion matrix')

    # history_dict = history.history
    # train_score = history_dict['accuracy'][29]
    # test_score = model.evaluate(x_test, y_test)

    # print(f"/ntrain accuracy: {train_score}/ntest_accuracy: {test_score}")

