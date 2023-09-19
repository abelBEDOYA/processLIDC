import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('./values_loss_epoch_14.pkl', 'rb') as archivo:
    datos = pickle.load(archivo)

print(datos.keys())

train = datos['epoch_loss_history']
val = datos['epoch_val_loss_history']
train_nodulo = datos['nodulo']
val_nodulo = datos['nodulo_val']
train_no_nodulo = datos['no_nodulo']
val_no_nodulo = datos['no_nodulo_val']

epochs = []
for i in range(1, len(train)+1):
    epochs.append(i)
plt.plot(epochs, train, label='train')
plt.plot(epochs,val, label='val')
plt.plot(epochs,train_nodulo, label='train_nodulo')
plt.plot(epochs,val_nodulo, label='val_nodulo')
plt.plot(epochs,train_no_nodulo, label='train_no_nodulo')
plt.plot(epochs,val_no_nodulo, label='val_no_nodulo')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='best', frameon=True)
plt.show()

