import numpy as np
import os

import random
import argparse


random.seed(123)


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



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # Agregar los argumentos necesarios
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--path2dataset', type=str, default='../../manifest-1675801116903/LIDC-IDRI/')
    parser.add_argument('--path2save', type=str, default='./')

    # Obtener los argumentos proporcionados por el usuario
    args = parser.parse_args()

    archivo = open('./failed_patients.txt', 'r')  # Reemplaza 'nombre_archivo.txt' por el nombre de tu archivo
    failed_patients = []
    for linea in archivo:
        linea = linea.strip()  # Elimina los espacios en blanco al principio y al final de la línea
        failed_patients.append(linea)
    archivo.close()
    patients = os.listdir(args.path2dataset)
    patients = [pat for pat in patients if not pat=='LICENSE' and pat not in failed_patients]

    train_patients, val_patients = train_val_split(patients, args.val_split)
    print(train_patients, '\n \n \n ', val_patients)
    # Descargamos el modelo preentrenado:
    with open(f'{args.path2save}val_patients.txt', 'w') as archivo:
        for elemento in val_patients:
            archivo.write(str(elemento) + '\n')
    print(f"La lista se ha guardado en el archivo {args.path2save}val_patients.txt")
        
        
        

