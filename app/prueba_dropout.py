import torch
from unet import UNet

# Supongamos que tienes un modelo con pesos aleatorios y otro con pesos entrenados
modelo = UNet(in_channels=3, out_channels=1, init_features=32, dropout_rate=0.2)  # Reemplaza "TuModelo" por el nombre de tu clase de modelo
modelo_entrenado = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                       in_channels=1, out_channels=1, init_features=32, pretrained=True)


# Cargar los pesos del modelo entrenado en el modelo aleatorio
modelo.load_state_dict(modelo_entrenado.state_dict())
