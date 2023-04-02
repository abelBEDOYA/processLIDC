# processLIDC
En este repo se incluyen metodos para la visualizacion, posible preprocesado y transformaciones del dataset LIDC, útil para tratar los datos y prepararlos para enternar con una UNET de pytorch.

## USAGE:

Para que la clase `Patient` (o incluso sin ella) pueda acceder a un paciente de LIDC es necesario:

1. Tener descargado uno o varios pacientes.

2. La libreria de `pylidc` que realiza las queries ira a mirar a la ruta indicada en el archivo oculto:
```
/home/abel/.pylidcrc
```
En ella hay que indicar:

```
path = /home/abel/TFM/manifest-1675801116903/LIDC-IDRI
```

De tal forma que el comando proporcione lo siguiente: 
```
ls /home/abel/TFM/manifest-1675801116903/LIDC-IDRI

OUTPUT: LIDC-IDRI-0002  LIDC-IDRI-0005
... 
```

Para usar lo ofrecido en este repo es ncesario seguir los siguietnets dos pasos:

1. `git clone https://github.com/abelBEDOYA/processLIDC`

2. Seguir el notebook: `main.ipynb`. En el se muestran los metodos que se ofrecen la clase `Patient`.


