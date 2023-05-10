import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import random
import argparse

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

    for id_pat in val_patients:
            # Cargamos datos de un paciente:
            patient = Patient(id_pat)
            
            # Escalamos:
            patient.scale()
            
            # Obtenemos los tensores:
            imgs, mask = patient.get_tensors(scaled=True)
            if torch.cuda.is_available():
                imgs, mask = imgs.to(torch.cuda.FloatTensor), mask.to(torch.cuda.FloatTensor)
            # Preparamos tensores para recorrerlos:
            # primera = 2
            # ultima = 10
            # dataset = TensorDataset(imgs[primera:ultima], mask[primera:ultima])
            dataset = TensorDataset(imgs, mask)
            

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

def plot(data, show=False, path_save=None):
    epoch_loss_history = data['epoch_loss_history']
    batch_loss_history = data['batch_loss_history']
    patient_loss_history = data['patient_loss_history']
    epoch_val_loss_history = data['epoch_val_loss_history']
    n_epochs = len(epoch_loss_history)
    plt.plot(np.linspace(1, n_epochs, np.array(batch_loss_history).shape[0]), np.log(np.array(batch_loss_history)), label='Train Batch Loss')
    plt.plot(np.linspace(1, n_epochs, np.array(patient_loss_history).shape[0]), np.log(np.array(patient_loss_history)), label='Train Patient Loss')
    plt.plot(np.linspace(1, n_epochs, n_epochs), np.log(np.array(epoch_loss_history)), label='Train Epoch Loss')
    plt.plot(np.linspace(1, n_epochs, n_epochs), np.log(np.array(epoch_val_loss_history)), label='Val. Epoch Loss')
    plt.title('Loss: Binary Cross Entropy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc = 'best', frameon=True)
    if path_save is not None:
        plt.savefig(path_save+'loss_plot.png', dpi=300)
    if show:
        plt.show()



def save_model(model, path='./', model_name='model'):
    if path[-1]=='/':
        # Guardar el modelo
        torch.save(model.state_dict(), path+model_name+'.pth')
        print('Modelo: {}{}.pth guardado.'.format(path, model_name))
    else:
        path = path+'/'
        torch.save(model.state_dict(), path+model_name+'.pth')
        print('Modelo: {}{}.pth guardado.'.format(path, model_name))

def train(model, n_epochs:int =4, 
          batch_size: int = 4, 
          val_split: float = 0.2,
          path2dataset: str = '../../manifest-1675801116903/LIDC-IDRI/',
          path2savefiles: str = './',
          plot_metrics: bool = False,
          save_plots: bool = False,
          save_epochs = None):
    """Ejecuta el entrenamiento

    Args:
        model (_type_): modelo a entrenar
        epochs (int, optional): numero de epocas. Defaults to 4.
        batch_size (int, optional): batch de imagenes (no pacientes) a evaluar antes de haer backprop. Defaults to 4.
        val_split (float, optional): porcentaje del dataset a validation. Defaults to 0.2.
        path2dataset: str = '../../manifest-1675801116903/LIDC-IDRI/',
        plot: bool = False,
        save_plots: bool = False)
    """
    patients = os.listdir(path2dataset)
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
            if torch.cuda.is_available():
                imgs, mask = imgs.to(torch.cuda.FloatTensor), mask.to(torch.cuda.FloatTensor)

            # Preparamos tensores para recorrerlos:
            # primera = 2
            # ultima = 10
            # dataset = TensorDataset(imgs[primera:ultima], mask[primera:ultima])
            dataset = TensorDataset(imgs, mask)

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
            print(os.system('date'))
        epoch_loss_history = np.append(epoch_loss_history, np.mean(np.array(loss_patient)))
        if save_epochs is not None:
            if epoch//save_epochs == epoch/save_epochs:
                save_model(model, path2savefiles, model_name= 'model-epoch{}'.format(epoch))
                if save_plots and epoch>1:
                    data_dict ={
                        'epoch_loss_history': epoch_loss_history,
                        'batch_loss_history': batch_loss_history,
                        'patient_loss_history': patient_loss_history,
                        'epoch_val_loss_history': epoch_val_loss_history
                        }
                    plot(data_dict, show=plot_metrics, path_save=path2savefiles)
        # # Calculemos el loss del val:
        val_loss = get_val_loss(model, val_patients, loss_fn, batch_size)
        epoch_val_loss_history = np.append(epoch_val_loss_history, val_loss)
        print('Train Epoch: {}\t Train Loss: {:.6f}. Val Loss: {:.6f}'.format(
            epoch, epoch_loss_history[-1], epoch_val_loss_history[-1]))
        print('-----------------------------------')
    data_dict ={
                'epoch_loss_history': epoch_loss_history,
                'batch_loss_history': batch_loss_history,
                'patient_loss_history': patient_loss_history,
                'epoch_val_loss_history': epoch_val_loss_history
                }
    save_model(model, path2savefiles, model_name='finalmodel')
    if save_plots:
        plot(data_dict, show=plot_metrics, path_save=path2savefiles)
    else:
        plot(data_dict, show=plot_metrics)
            
def checks_alright(args):
    if not args.n_epochs > 0 and not isinstance(args.n_epochs, int):
        raise ValueError("n_epochs no es entero y >0")
    if not args.batch_size > 0 and not isinstance(args.batch_size, int):
        raise ValueError("batch_size no es entero y >0")
    
    # print(args.val_split > 0 and not args.val_split < 1 and isinstance(args.val_split, float))
    if not args.val_split > 0 and not args.val_split < 1 and not isinstance(args.val_split, float):
        print('fallooooo')
        raise ValueError("val_split no es float y >0 y >1")
    
    try:
        if not os.path.exists(args.path2dataset):
            raise ValueError("La ruta al dataset, path2dataset, esta mal")
    except:
        raise ValueError("La ruta al dataset, path2dataset, esta mal")
    try:
        if not os.path.exists(args.path2savefiles):
            raise ValueError("La ruta a donde se guardaran los archivos, path2savefiles, esta mal")
    except:
        raise ValueError("La ruta a donde se guardaran los archivos, path2savefiles, esta mal")
    
    if not isinstance(args.plot_metrics, bool):
        raise ValueError("save_plots no es booleano")
    
    if not isinstance(args.plot_metrics, bool):
        raise ValueError("save_plots no es booleano")
    
    if not isinstance(args.save_epochs, int):
        if  args.save_epochs > 0 and not args.save_epochs < args.n_epochs:
            raise ValueError("save_epochs no es entero y >0 y menos que n_epochs")



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # Agregar los argumentos necesarios
    parser.add_argument('--n_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--path2dataset', type=str, default='../../manifest-1675801116903/LIDC-IDRI/')
    parser.add_argument('--path2savefiles', type=str, default='./')
    parser.add_argument('--plot_metrics', action='store_true', default = False)
    parser.add_argument('--save_plots', action='store_true', default = True)
    parser.add_argument('--save_epochs', type=int, default=None)

    # Obtener los argumentos proporcionados por el usuario
    args = parser.parse_args()
    checks_alright(args)
    # Descargamos el modelo preentrenado:
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                       in_channels=3, out_channels=1, init_features=32, pretrained=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
    
    # Llamar a la función train con los argumentos
    train(model, n_epochs=args.n_epochs, batch_size=args.batch_size, val_split=args.val_split,
        path2dataset=args.path2dataset, path2savefiles=args.path2savefiles,
        plot_metrics=args.plot_metrics, save_plots=args.save_plots,
        save_epochs=args.save_epochs)
        
        
        

