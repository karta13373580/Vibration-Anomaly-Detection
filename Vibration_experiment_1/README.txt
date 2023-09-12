dataset_20220127.zip該資料集使用0127pump-100筆.zip製作，
因資料過少且無需透過Deep learning訓練與預測，
故直接以dataset_20220127.zip中的main.py做切分與svm的訓練，
採樣頻率為50Hz，因設備影響採樣頻率不穩定為非均勻採樣，
每筆資料維度為(1500, 4)，1500為以採樣頻率50Hz採集30秒，
4為x, y, z與time

