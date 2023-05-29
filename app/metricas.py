import torch
# Mi libreria:
from processLIDC import Patient
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import random
import argparse
import os

def get_confusion_matrix(id_patient, model, threshold = 0.5, batch = 10):
    cm = np.zeros((2,2))
    print(cm)
    patient = Patient(id_patient)
    patient.scale()
    images, mask = patient.get_tensors(scaled = False)
    mask = mask.cpu().detach().numpy()
    n_slices = mask.shape[0]
    slices = (0, int(batch))
    prediccion = patient.predict(model, slices=slices, scaled=True, gpu = True)
    prediccion = np.where(prediccion >= threshold, 1, 0)[:,0,:,:]
    
    for i in range(batch, n_slices, batch):
        
        slices = (i, i+batch)
        # print(i+batch, n_slices)
        pred = patient.predict(model, slices=slices, scaled=True, gpu = True)
        pred_bin = np.where(pred >= threshold, 1, 0)[:,0,:,:]
        prediccion = np.concatenate((prediccion, pred_bin), axis=0)
        

    label = mask.flatten()
    
    prediccion = prediccion.flatten()
    cm_ = confusion_matrix(label, prediccion, labels=(0,1))
    cm = cm + np.array(cm_)
    
    print('terminado')
    return cm


def get_confusion_matrix_list(id_patient, model, threshold = 0.5, batch = 10):
    cm = np.zeros((2,2))
    if isinstance(id_patient,str):
        cm = get_confusion_matrix(id_patient, model, threshold =threshold, batch = batch)
        return cm
    else:
        cm = get_confusion_matrix(id_patient[0], model, threshold =threshold, batch = batch)
        for id in tqdm(id_patient[1:]):
            cm = cm + get_confusion_matrix(id, model, threshold =threshold, batch = batch)
        return cm

def plotNsave(cm, save = None, show = True):
    fig, ax = plt.subplots()

    # Crear mapa de calor utilizando seaborn
    sns.heatmap(np.int32(cm/100), annot=True, fmt="d", cmap="Blues", cbar=False, square=True)  # , xticklabels=labels, yticklabels=labels)

    # Añadir etiquetas a los ejes
    ax.set_xlabel("Etiqueta Predicha")
    ax.set_ylabel("Etiqueta Verdadera")

    # Añadir título
    ax.set_title("Matriz de Confusión")
    if show:
        # Mostrar la figura
        plt.show()
    if save is not None:
        plt.savefig(save+'confusion_matrix.png', dpi=300)


if __name__ == "__main__":
    random.seed(123)
    parser = argparse.ArgumentParser()
    # Agregar los argumentos necesarios
    parser.add_argument('--val', action='store_true', default = True)
    parser.add_argument('--model', type=str, default='./default_model.pt')
    parser.add_argument('--save', type=str, default='./')
    parser.add_argument('--path2dataset', type=str, default='../../manifest-1675801116903/LIDC-IDRI/')
    parser.add_argument('--valsplit', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--batch', type=float, default=5)
    args = parser.parse_args()
    
    print('Buscando los pacientes...', flush= True)
    patients = os.listdir(args.path2dataset)
    archivo = open('./failed_patients.txt', 'r')  # Reemplaza 'nombre_archivo.txt' por el nombre de tu archivo
    failed_patients = []

    for linea in archivo:
        linea = linea.strip()  # Elimina los espacios en blanco al principio y al final de la línea
        failed_patients.append(linea)
    archivo.close()
    patients = [pat for pat in patients if not pat=='LICENSE' and pat not in failed_patients]
    n_val = int(len(patients) * args.valsplit)
    if args.val:
        # Seleciona aleatoriamente el 30% de los patients
        patients_list = random.sample(patients, n_val)
    else:
        val_patients = random.sample(patients, n_val)
        patients_list = [nombre for nombre in patients if nombre not in val_patients]
        
    print('Cargando el modelo...', flush=True)
    model = torch.jit.load(args.model)
    model.to('cuda')
    model.eval()
    cm = get_confusion_matrix_list(patients_list, model, args.threshold, args.batch)
    plotNsave(cm, args.save, show=False)
    
    
    