from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn import preprocessing

def CreateModel():
    np.random.seed(7)
    np.set_printoptions(suppress=True)

    df = pd.read_csv("Train.csv", skiprows = 0, delimiter = ",", dtype=float, usecols=[2,3,4,5])
    df.fillna(0, inplace=True)
    candidate_raw_dataset = df.values

    df2 = pd.read_csv("Train.csv", skiprows = 0, delimiter = ",", dtype=float, usecols=[0])
    df2.fillna(0, inplace=True)
    predict_raw_dataset = df2.values

    min_max_scaler = preprocessing.MinMaxScaler()
    scale_candidate = min_max_scaler.fit_transform(candidate_raw_dataset)
    scale_predict = min_max_scaler.fit_transform(predict_raw_dataset)
    normalized_data_1 =pd.DataFrame(scale_candidate)
    normalized_data_2 =pd.DataFrame(scale_predict)
    candidate_set = normalized_data_1.values
    predict_set = normalized_data_2.values

    candidate = candidate_set[:, 2:5]
    predict = predict_set[:, 0]


    model = Sequential()

    model.add(Dense(12, input_dim=2, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(candidate, predict, epochs=200, batch_size=74)

    scores = model.evaluate(candidate, predict)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    model.save('my_model.h5')
    del model
    model=load_model('my_model.h5')



def returnMinandMax():
    df = pd.read_csv("Train.csv", skiprows = 0, delimiter = ",", dtype=float, usecols=[2,3,4,5])
    df.fillna(0, inplace=True)
    candidate_raw_dataset = df.values

    df2 = pd.read_csv("Train.csv", skiprows = 0, delimiter = ",", dtype=float, usecols=[0])
    df2.fillna(0, inplace=True)
    predict_raw_dataset = df2.values
    
    minimal = int(min(predict_raw_dataset))
    maximal = int(max(predict_raw_dataset))

    return minimal, maximal

if __name__ == '__main__':
    this = CreateModel()
