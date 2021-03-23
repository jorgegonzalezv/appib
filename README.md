# appib

Práctica de Información Biométrica UAM EPS
Deepfake detection

El repositorio contiene el codigo para ejecutar el entrenamiento de un modelo de red neuronal convolucional sobre los datos provistos en la práctica.

Instrucciones:

Descargar repo 
```python 
 git clone https://github.com/jorgegonzalezv/appib.git
 cd appib
```

Copiar los archivos Task_1 y Task_2_3 en appib
```python 
 cd appib
 cp path-to-files/Task_1 ./Task_1
 cp path-to-files/Task_2_3 ./Task_2_3
```

Instalar dependencias python
```python
 pip install -r requirements.txt
```

Preprocesado de los datos. Extracción de las caras. Dentro del fichero refactor.py se especifica el directorio de imágenes a procesar.
```python
 python refactor.py
```

Entrenamiento modelo
```python
 python src/train.py
```
Los pesos se van guardando en el directorio weights. 
Para sacar la curva roc para un modelo con unos pesos específicos, finalmente. Se genera un fichero plot.png.
```python
 python roc.py
```




