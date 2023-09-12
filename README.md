# Vibration-Anomaly-Detection

船舶上的引擎及大型馬達在運轉的時候會產生震動訊號，我們可藉由檢測此訊號來判斷船舶是否出現損壞故障，以免造成危險。
本專案使用SVM模型方法進行實驗，並且將震動訊號做一階的離散小波轉換，
提取出標準差與絕對中位差兩種重要的資訊當作模型訓練的特徵，最終準確率在實際的船舶上錄製與我們使用小型馬達模擬的兩個震動訊號資料集皆達到100%。

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

| Model | Dataset | Acc |
| :----: | :----: | :----: |
| 離散小波轉換+SVM | 小型馬達模擬 | 100% |
| Frequency Doamin+SVM | 實際船舶錄製 | 100% |

## 資料集介紹
### Vibration-Anomaly-Detection_1  
正常震動訊號有100筆，異常震動訊號有10筆，該資料為室內模擬的小型馬達震動訊號  
<img src="https://github.com/karta13373580/Vibration-Anomaly-Detection/blob/main/github_photo/3.png">  

### Vibration-Anomaly-Detection_2  
正常震動訊號有227筆，異常震動訊號有58筆，該資料為實際於船舶上收集之引擎震動訊號
<img src="https://github.com/karta13373580/Vibration-Anomaly-Detection/blob/main/github_photo/4.png">
<img src="https://github.com/karta13373580/Vibration-Anomaly-Detection/blob/main/github_photo/5.png">

## 實驗環境

* CUDA: 11.3
* Python: 3.8.0
* matplotlib==3.3.4
* PyWavelets==1.1.1
* statsmodels==0.12.2
* joblib==1.1.1
```
pip install -r requirements.txt
```

## 模型使用
### 訓練與測試
```
python main.py
```

## Jetson Nano 2GB
### 板子運行畫面
<img src="https://github.com/karta13373580/Vibration-Anomaly-Detection/blob/main/github_photo/6.png">
### 板子規格
<img src="https://github.com/karta13373580/Vibration-Anomaly-Detection/blob/main/github_photo/7.png">

## 參考資料
* <https://hdl.handle.net/11296/38mzge>
