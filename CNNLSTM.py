import os
import numpy as np
import keras
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
file_list = os.listdir(".")

def load_data():
    Lista_Dati = []
    Lista_label = []
    file_list = os.listdir(".")
    for filename in file_list:
        if "txt" in filename:
            f = open(filename, "r")
            for row in f.readlines():
                Row_S = row.strip().split(",")
                #print(Row_S)
                # print(len(Row_S))
                Lista_Dati.append(
                    (float(Row_S[1]), float(Row_S[2]), float(Row_S[3]), float(Row_S[4]), float(Row_S[5]),
                     float(Row_S[6]),
                     float(Row_S[7]), float(Row_S[8]), float(Row_S[9]), float(Row_S[10]),
                     float(Row_S[11]), float(Row_S[12]), float(Row_S[13]), float(Row_S[14]), float(Row_S[15]),
                     float(Row_S[16]),
                     float(Row_S[17]), float(Row_S[18]), float(Row_S[19]),
                     float(Row_S[20]), float(Row_S[21]), float(Row_S[22]), float(Row_S[23]), float(Row_S[24])))
                Lista_label.append(Row_S[25])

    return Lista_Dati, Lista_label

#load_data()

Lista_dati, Lista_label = load_data()
#print(Lista_dati, Lista_label)

def finestre_temporali(Lista_dati, Lista_label):
    Grouped_List = []
    Lista_label_ridotta = []
    for i in range(0, len(Lista_dati), 100):
        group = Lista_dati[i:i + 100]
        group2 = Lista_label[i:i + 100]
        Grouped_List.append(group)
        Lista_label_ridotta.append(group2)
   # print(Grouped_List)
    #print(Lista_label_ridotta)
    Lista_indici = []
    Lista_dati_Corretta = []
    for i in range(len(Lista_label_ridotta)):
        #print(Lista_label_ridotta[i])
        if len(set(Lista_label_ridotta[i])) < 2:
            Lista_indici.append(set(Lista_label_ridotta[i]))
            Lista_dati_Corretta.append(Grouped_List[i])

    Lista_Indici_Finale = []
    for el in Lista_indici:
        lista = list(el)
        for el2 in lista:
            Lista_Indici_Finale.append(el2)

    return Lista_dati_Corretta, Lista_Indici_Finale


X, y =finestre_temporali(Lista_dati, Lista_label)

def do_label_numerica(X,y):
    X_new = []
    y_new = []
    print(len(X))
    for i in range(len(y)):
        if y[i] == "idle":
            continue
        else:
            X_new.append(X[i])
            y_new.append(y[i])

    return X_new, y_new

X_new, y_new = do_label_numerica(X,y)


def do_label_Corretta(y_new):
    Y_new2 = []
    for el in y_new:
        if el =="walk-stairascent" or el =="stairascent" or el =="stairascent-walk":
            Y_new2.append(0)
        elif el == "walk-stairdescent" or el =="stairdescent" or el =="stairdescent-walk":
            Y_new2.append(1)
        elif el == "walk-rampascent" or el =="rampascent" or el =="rampascent-walk":
            Y_new2.append(2)
        elif el == "walk-rampdescent" or el =="rampdescent" or el =="rampdescent-walk":
            Y_new2.append(3)
        else:
            Y_new2.append(4)

    return Y_new2

Y_new2C = do_label_Corretta(y_new)
#print(Y_new2C)
#print(len(Y_new2C))
def correggo(Y_new2C, X_new):
    New_X = []
    New_y = []
    for i in range(len(X_new)):
        if len(X_new[i]) < 100:
            continue
        else:
            New_X.append(X_new[i])
            New_y.append(Y_new2C[i])

    New_X = np.array(New_X)
    New_y = np.array(New_y)
    return New_X, New_y

Dati_X, label_y = correggo(Y_new2C, X_new)

print(len(Dati_X))
model = keras.Sequential([
    layers.Conv1D(128, kernel_size=3, activation='relu',input_shape=(100,24)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(128, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(128, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.LSTM(128, return_sequences=True),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')  # Adjust the number of output units based on your problem (3 for 3 classes)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(Dati_X, label_y, epochs=10)
model.save('CNN_LSTM_lineare.h5')
