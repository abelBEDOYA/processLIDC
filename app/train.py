import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import random

# Mi libreria:
from processLIDC import Patient

def train_val_split(patients_list, val_split):
    """TOma la lsita de pacientes list(str) y hace la separacion
    en train y validation segun la proporcion indicada en val_split.
    Args:
        patients_list (list(str)): lista on los id de los pacientes
        val_split (float): proporcion destinada a validation
    Returns:
        train_patients, val_val_patients (list, list): lista de
            nombres de pacientes para train y validation
        """
    n_val = int(len(patients_list) * val_split)

    # Seleciona aleatoriamente el 30% de los patients
    val_patients = random.sample(patients_list, n_val)

    # Crea una lista con los patient que no fueron seleccionados
    train_patients = [nombre for nombre in patients_list if nombre not in val_patients]
    return train_patients, val_patients


def get_val_loss(model, val_patients, loss_fn, batch_size=4):
    if len(val_patients)==0:
        return 0

    loss_batch = np.array([])
    batch_loss_history = np.array([])
    loss_patient = np.array([])

    for id_pat in tqdm(val_patients):
            # Cargamos datos de un paciente:
            patient = Patient(id_pat)
            
            # Escalamos:
            patient.scale()
            
            # Obtenemos los tensores:
            imgs, mask = patient.get_tensors(scaled=True)
            
            # Preparamos tensores para recorrerlos:
            primera = 2
            ultima = 6
            dataset = TensorDataset(imgs[primera:ultima], mask[primera:ultima])
            # dataset = TensorDataset(imgs, mask)
            

            train_loader = DataLoader(dataset,batch_size=batch_size, shuffle=True)
            loss_batch = np.array([])
            for batch_idx, (data, target) in enumerate(train_loader):

                # # Forward pass
                output = model(data)
                # Calcular pérdida
                loss = loss_fn(output[:,0], target)
                loss_batch = np.append(loss_batch, loss.item())
                batch_loss_history = np.append(batch_loss_history, loss.item())

            loss_patient = np.append(loss_patient, np.mean(np.array(loss_batch)))
    val_mean_loss = np.mean(loss_patient)
    return val_mean_loss

def train(model, n_epochs:int =4, 
          batch_size: int =4, 
          val_split: float = 0.2,
          path2dataset: str = '../../manifest-1675801116903/LIDC-IDRI/'):
    """Ejecuta el entrenamiento

    Args:
        model (_type_): modelo a entrenar
        epochs (int, optional): numero de epocas. Defaults to 4.
        batch_size (int, optional): batch de imagenes (no pacientes) a evaluar antes de haer backprop. Defaults to 4.
        val_split (float, optional): porcentaje del dataset a validation. Defaults to 0.2.
    """
    patients = os.listdir('../../manifest-1675801116903/LIDC-IDRI/')
    patients = [pat for pat in patients if not pat=='LICENSE']

    train_patients, val_patients = train_val_split(patients, val_split)
    # Definir función de pérdida
    loss_fn = nn.BCELoss()

    # Definir optimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    loss_batch = np.array([])
    batch_loss_history = np.array([])

    loss_patient = np.array([])
    patient_loss_history = np.array([])

    epoch_loss_history = np.array([])
    epoch_loss_history = np.array([])
    epoch_val_loss_history = np.array([])
    for epoch in range(n_epochs):
        print(f'Epoch: {epoch+1}/{n_epochs}')
        loss_patient = np.array([])
        random.shuffle(train_patients)
        for id_pat in tqdm(train_patients):
            print('Cargando paciente: {}'.format(id_pat))
            # Cargamos datos de un paciente:
            patient = Patient(id_pat)
            
            # Escalamos:
            patient.scale()
            
            # Obtenemos los tensores:
            imgs, mask = patient.get_tensors(scaled=True)
            
            # Preparamos tensores para recorrerlos:
            primera = 2
            ultima = 6
            dataset = TensorDataset(imgs[primera:ultima], mask[primera:ultima])
            # dataset = TensorDataset(imgs, mask)
            

            train_loader = DataLoader(dataset,batch_size=batch_size, shuffle=True)
            loss_batch = np.array([])
            for batch_idx, (data, target) in enumerate(train_loader):

                # # Forward pass
                output = model(data)
                # Calcular pérdida
                loss = loss_fn(output[:,0], target)

                # # # Calcular gradientes y actualizar parámetros
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_batch = np.append(loss_batch, loss.item())
                batch_loss_history = np.append(batch_loss_history, loss.item())

            loss_patient = np.append(loss_patient, np.mean(np.array(loss_batch)))
            patient_loss_history = np.append(patient_loss_history, np.mean(np.array(loss_batch)))
            
        epoch_loss_history = np.append(epoch_loss_history, np.mean(np.array(loss_patient)))
        
        # Calculemos el loss del val:
        val_loss = get_val_loss(model, val_patients, loss_fn, batch_size)
        epoch_val_loss_history = np.append(epoch_val_loss_history, val_loss)
        
        print('Train Epoch: {}\t Train Loss: {:.6f}. Val Loss: {:.6f}'.format(
            epoch, epoch_loss_history[-1], epoch_val_loss_history))
        print('-----------------------------------')
