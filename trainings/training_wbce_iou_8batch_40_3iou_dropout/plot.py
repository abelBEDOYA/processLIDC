import pickle
import numpy as np
import matplotlib.pyplot as plt


# with open('./values_loss_epoch_12.pkl', 'rb') as archivo:
#     datos = pickle.load(archivo)

# print(datos.keys())
# epochs = [6.5,6.6,6.6, 6.13,5.9, 6.05,5.9,5.9, 6.05, 5.7, 5.97, 5.93,5.6, 6.13,6.05, 5.95, 6.4,6, 6.08,5.9]
val = [6.55,6.7,6.7, 6.13,5.9, 6.05,5.9,5.9, 6.05, 5.7, 5.97, 5.93, 5.6, 6.2, 5.95, 6.18,6, 6.08,5.9,6.45,5.95,6.07, 5.91,6.2]

train = [6.05, 5.9,5.5,5.3,5.15,5.05, 4.9, 4.87,4.76, 4.7,4.6,4.5, 4.4, 4.3, 4.18, 4.14, 4.1, 4.05, 3.95,3.9,3.85,3.75, 3.6,3.5]

epochs = []
for i in range(1, len(train)+1):
    epochs.append(i)
plt.plot(epochs, np.exp(train), label='train')
plt.plot(epochs, np.exp(val), label='val')
# plt.plot(epochs,train_nodulo, label='train_nodulo')
# plt.plot(epochs,val_nodulo, label='val_nodulo')
# plt.plot(epochs,train_no_nodulo, label='train_no_nodulo')
# plt.plot(epochs,val_no_nodulo, label='val_no_nodulo')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='best', frameon=True)
plt.savefig(dpi=500, fname='referencia.png')
plt.show()

