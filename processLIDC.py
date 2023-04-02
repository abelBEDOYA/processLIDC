import pylidc as pl
from pylidc.utils import consensus
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import torch
import time
import plotly.graph_objects as go
import numpy as np
from mayavi import mlab
import cv2


class Patient():
    def __init__(self, id_patient):
        self.id_patient = id_patient
        self.scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == id_patient).first()
        self.vol = self.scan.to_volume()
        self.mask = self.get_mask()

    def get_mask(self, print_count=False):
        mask = np.zeros_like(self.vol)
        nod_count = 0
        for ann_clust in self.scan.cluster_annotations():
            # print('hola')
            nod_count +=1
            cmask, cbbox, _ = consensus(ann_clust, clevel=0.5,
                                        pad=[(20, 20), (20, 20), (0, 0)])
            mask[cbbox] += cmask
        if print_count is True:
                print(nod_count)
        return mask

    def plot_mask(self):
        # Generar una matriz binaria aleatoria de 10x10x10
        data = self.mask
        x, y, z = np.where(data == 1)
        fig = self.__plot_3d(x,y,z)
        x_min, x_max = 0, self.mask.shape[1]
        y_min, y_max = 0, self.mask.shape[0]
        z_min, z_max = 0, self.mask.shape[2]
        fig.update_layout(scene=dict(xaxis=dict(range=[x_min, x_max]),
                                    yaxis=dict(range=[y_min, y_max]),
                                    zaxis=dict(range=[z_min, z_max])),
                        title={'text': "mask nodulos"})
        # fig.title('mascara tumores')
        fig.show()

    def __plot_3d(self, x, y, z):
        """
        Plotea en 3d (scater plot) a partir de tres arrays: x, y, z
        Siendo estos las coordenadas de cada punto
        """
        # Crea el gráfico de dispersión 3D
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=1,
                opacity=0.1,
                colorscale='Viridis',  # Escala de color
                colorbar=dict(title='Valor de z')  # Leyenda del color
            )
        )])

        # Configura el diseño del gráfico
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Eje x (mm)'),
                yaxis=dict(title='Eje y (mm)'),
                zaxis=dict(title='Eje z (mm)'),
            )
        )
        return fig 


    def reconstruct_body(self, body=True, nodulos=False):
        # vol  = np.transpose( vol , [2,0,1]).astype(np.float32) # each row will be a slice

        points = {'x': np.array([]), 'y': np.array([]), 'z': np.array([])}
        if body is True:
            prob_true = 0.3
            for i in range(self.vol.shape[2]):
                # print(np.max(vol[:, :, i]))
                thresh_value = 600
                ret, thresh = cv2.threshold(
                    self.vol[:, :, i], thresh_value, 1, cv2.THRESH_BINARY)
                x, y = np.where(thresh == 1)
                arr_bool = np.random.choice([True, False], size=len(x), p=[
                                            prob_true, 1-prob_true])
                z = np.ones(x.shape)*i
                x, y, z = x[arr_bool], y[arr_bool], z[arr_bool]
                points['x'] = np.append(points['x'], x)
                points['y'] = np.append(points['y'], y)
                points['z'] = np.append(points['z'], z)

        x, y, z = points['x'], points['y'], points['z']

        if nodulos == True:
            x_m, y_m, z_m = np.where(self.mask == 1)
            x = np.append(x, x_m)
            y = np.append(y, y_m)
            z = np.append(z, z_m)

        x, y, z = x*self.scan.pixel_spacing, y*self.scan.pixel_spacing, z*self.scan.slice_spacing
        print(len(points['x']))
        fig = self.__plot_3d(x, y, z)
        fig.show()

    def get_all_nodules(self, plot = False):
        """
        Aqui se obtienen las anoticaion haciendo la query con un join del scan con 
        annotation, a veces hay mas annotaciones que con scan simplemente. Aunque suele 
        ser nodulos repetidos, etiqeutados ligereamente distintos. 
        Esto es solo para informacion.
        """
        anns = pl.query(pl.Annotation).join(pl.Scan).filter(pl.Scan.patient_id == self.id_patient)
        print(f'Paciente: {self.id_patient}')
        print('___________________________________')
        for ann in anns:
            print(f'Paciente del nodulo: {ann.scan.patient_id}')
            print('Primera slice con el nodulo',ann.contour_slice_indices[0])
            print(f'num. slices nodulo: {ann.boolean_mask().shape[-1]}')
            # # Visualizacion interacctiva con el contour tambien:
            if plot == True:
                ann.visualize_in_scan()
            print('-----------')
        print('___________________________________')
    
    def normalice(self, slices= (0, ), plot = False, with_mask = False):
        imgs = np.copy(self.vol)  # +mask[:, :, i]*1000
        ## Histograma antes:
        imgs1 = imgs[:,:, list(slices)]
        h_antes = imgs1[:, :, 0].reshape(-1)
        
        # Escalado:
        mini = np.min(imgs[imgs >=-2000])
        imgs=imgs-mini
        imgs[imgs <=0] = 0
        if with_mask is True and plot == True:
            imgs = np.log(imgs+1) + self.mask*3
        else:
            imgs = np.log(imgs+1)
        
        # Histograma despues:
        imgs2 = imgs[:,:, list(slices)]
        h_despues = imgs2[:, :, 0].reshape(-1)
        
        if plot == True:
            plt.hist(h_antes)
            plt.title('sin escalar')
            plt.yscale('log')
            plt.show()
                
            plt.hist(h_despues)
            plt.title('escalado')
            plt.yscale('log')
            plt.show()
            
            
            for i in range(len(slices)):
                img1 = imgs1[:, :, i]  # +mask[:, :, i]*1000
                # ## Imagen:
                plt.imshow(img1)
                plt.title('sin escalar')
                plt.title(f'slice: {i+slices[0]}')
                plt.show()
                img2 = imgs2[:, :, i]
                plt.imshow(img2)
                plt.title('escalado')
                plt.title(f'slice: {i+slices[0]}')
                plt.show()
            
            

