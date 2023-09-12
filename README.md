# Vibration-Anomaly-Detection

船舶上的引擎及大型馬達在運轉的時候會產生震動訊號，我們可藉由檢測此訊號來判斷船舶是否出現損壞故障，以免造成危險。
本專案使用SVM模型方法進行實驗，並且將震動訊號做一階的離散小波轉換，
提取出標準差與絕對中位差兩種重要的資訊當作模型訓練的特徵，最終準確率在實際的船舶上錄製與我們使用小型馬達模擬的兩個震動訊號資料集皆達到100%。

## 資料集介紹
### Vibration-Anomaly-Detection_1  
正常震動訊號有100筆，異常震動訊號有10筆，該資料為室內模擬的馬達震動訊號  
<img src="https://github.com/karta13373580/Vibration-Anomaly-Detection/blob/main/github_photo/3.png">  

### Vibration-Anomaly-Detection_2  
正常震動訊號有227筆，異常震動訊號有58筆，該資料為實際於船舶上收集之引擎震動訊號
<img src="https://github.com/karta13373580/Vibration-Anomaly-Detection/blob/main/github_photo/4.png">
<img src="https://github.com/karta13373580/Vibration-Anomaly-Detection/blob/main/github_photo/5.png">

## 方法
### Vibration-Anomaly-Detection_1  
<img src="https://github.com/karta13373580/Vibration-Anomaly-Detection/blob/main/github_photo/1.PNG">  
我們使用小型馬達所模擬之震動訊號其Shape是(100, 1500, 4)，並且藉由觀察正常資料(Water)與異常資料(NoWater)兩種差異，分別將x，y，z軸分開去做一階的離散小波轉換，
分解出較為突出的高頻訊號特徵，進而降低雜訊干擾，最後將此高頻訊號特徵結合統計方式，各自計算標準差與絕對中位差，最後輸入的單筆震動訊號特徵為六維


### Vibration-Anomaly-Detection_2  
<img src="https://github.com/karta13373580/Vibration-Anomaly-Detection/blob/main/github_photo/2.PNG">  
我們從實際的船舶上錄製的震動訊號可以發現正常資料(WasteWaterPump)與異常資料(NoWaterPump)在Frequency Domain上的
Amplitude具有一定程度的差異，正常Amplitude大部分都落在個位數區間，但異常則超過，因此採用直接統計的方式來做為模型訓練的特徵

## 實驗結果



<img src="https://github.com/karta13373580/Vibration-Anomaly-Detection/blob/main/github_photo/%E6%93%B7%E5%8F%96.PNG">

| Machine ID | Fan | Pump | Slider | Valve | Average |
| :----: | :----: | :----: | :----: | :----: | :----: |
| 00 | 73.18% | 97.05% | 96.82% | 66.43% | 84.94% |
| 02 | 90.03% | 76.47% | 82.01% | 98.33% | 86.79% |
| 04 | 80.31% | 86.85% | 59.60% | 55.83% | 75.24% |
| 06 | 96.25% | 86.79% | 62.55% | 73.92% | 73.62% |

## 與其他論文實驗比較
| Model | Fan | Pump | Slider | Valve |
| :----: | :----: | :----: | :----: | :----: |
| Autoencoder | 65.83% | 72.89% | 84.76% | 66.28% |
| LSTM- Autoencoder | 67.32% | 73.94% | 84.99% | 67.82% |
| Dictionary Learning-Autoencoder | 79.60% | 84.91% | 82.00% | 72.33% |
| Contrastive Learning | 80.11% | 70.12% | 77.43% | 84.17% |
| Baseline-GANomaly | 80.34% | 83.90% | 72.70% | 68.51% |
| Proposed | 84.94% | 86.79% | 75.24% | 73.62% |

## 實驗環境
* CUDA: 11.3
* Python: 3.8.0
* Pytorch: 1.12.0
* pytorch-lightning==1.6.1
```
pip install -r requirements.txt
```

## 建立資料集
請先建立一個data資料夾，資料擺放方式如下: 
```
-dataset/
  -train/
    -normal/
      -00000000.wav
      -00000001.wav
      -00000002.wav
  -val/
    -normal/
      -00000003.wav
  -thr/
    -normal/
      -00000004.wav
    -abnormal/
      -00000000.wav
      -00000001.wav
      -00000002.wav
  -test/
    -normal/
      -00000005.wav
    -abnormal/
      -00000003.wav
```
## 模型使用
### yml檔參數配置


| 模型參數 | 描述 | 聲音參數 | 描述 |
| :---- | :---- | :---- | :---- |
| root | 資料集路徑 | sample_rate | 聲音每秒採樣率 |
| checkpoint_path | 模型權重路徑 | max_waveform_length | 模型所要使用的聲音最大採樣率 |
| threshold | 測試階段需定義閥值 | n_mels | 梅爾頻譜轉換頻帶 |
| early_stopping | 是否啟用模型提早結束 | n_fft | 快速傅立葉轉換的Window大小 |
| max_epochs | 定義模型最大訓練次數 | hop_length | Window跳躍長度 |

### 訓練
```
python main.py --config MIMII_p6_dB_pump_id_00_yaml/config_MIMII_p6_dB_pump_id_00_normal_abnormal_train.yml --str_kwargs mode=train
```
### 查看Threshold
```
python main.py --config MIMII_p6_dB_pump_id_00_yaml/config_MIMII_p6_dB_pump_id_00_normal_abnormal_threshold.yml --str_kwargs mode=threshold
```
### 測試
```
python main.py --config MIMII_p6_dB_pump_id_00_yaml/config_MIMII_p6_dB_pump_id_00_normal_abnormal_test.yml --str_kwargs mode=test
```
## UI介面
<img src="https://github.com/karta13373580/Audio-Anomaly-Detection/blob/main/result_photo/github_photo/UI%E5%BD%B1%E7%89%87%20(online-video-cutter.com).gif">

### 啟用UI
```
python start.py --config MIMII_p6_dB_pump_id_00_yaml/config_MIMII_p6_dB_pump_id_00_normal_abnormal_gui.yml --str_kwargs mode=predict_gui
```
## 參考資料
* <https://github.com/fastyangmh/AudioGANomaly>
* <https://github.com/lucidrains/halonet-pytorch>
* <https://blog.csdn.net/weixin_38241876/article/details/109853433>
* <https://blog.csdn.net/pipisorry/article/details/53635895>
