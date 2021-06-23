# PROYECTO-FINAL-PLAIDML
PROYECTO CREADO POR HAWKS:
Proyecto de la materia de Lenguajes y autómatas 2 impartida por el profesor: Parra Hernandez Tomas Abdiel

Se presenta el proyecto: PlaidML

Equipo Hawks:
Avila Gomez Giovanni Arturo:
Github: https://github.com/Giovanni-Avila/PROYECTO-FINAL-PLAIDML

García Pacheco Axel David:
Github: https://github.com/axel-david-garcia-pacheco/PlaidML-ProyectoFinal

Rojas Pérez José Ramón:
Github:https://github.com/RamonRojas9987/PLAIDML-PROYECTOFINAL

# INTRODUCCIÓN

Primero comencemo con conocer que MLIR Evalúa como una infraestructura generalizada que reduce la costó de construcción de compiladores, que describe diversos casos de uso para mostrar la investigación y oportunidades educativas para futuros lenguajes de programación, compiladores, ejecución entornos y arquitectura informática. 

MLIR comenzó al darse cuenta de que los marcos de aprendizaje automático modernos se componen de muchos compiladores, tecnologías de gráficos y sistemas de tiempo de ejecución diferentes, que no compartir una infraestructura o un punto de diseño común, y no todos seguían las mejores prácticas en el diseño de compiladores.

# PLAIDML

PLAIDML, es un compilador que permite al usuario aprender un poco más sobre computadoras portátiles, este compilador forma parte de  uno de los más bajos lo que ayuda al usuario a acceder a cualquier hardware ya que  también PLAIDML es compatible con keras y onnyx 


# KERAS

Keras facilita la ejecución de nuevos experimentos, le permite probar más ideas que su competencia, más rápido. Y así es como se gana. Tambien puede acelerar las cargas de trabajo de entrenamiento con código Tile personalizado o generado automáticamente. Funciona especialmente bien en GPU y no requiere el uso de CUDA / cuDNN en hardware Nvidia, al tiempo que logra un rendimiento comparable.

# IMPLEMENTACIÓN

La implementación que se llevó acabo fue de PLAIDML en dos equipos con el S.O. Wndows un equipo con su procesador INTEL y otro con AMD.

# REQUISITOS

•	Python (se admite v2, se recomienda v3)

•	OpenCL 1.2 o superior

# CÓDIGO

import numpy as np
import os
import time

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import keras.applications as kapp
from keras.datasets import cifar10

(x_train, y_train_cats), (x_test, y_test_cats) = cifar10.load_data()
batch_size = 8
x_train = x_train[:batch_size]
x_train = np.repeat(np.repeat(x_train, 7, axis=1), 7, axis=2)
model = kapp.VGG19()
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Running initial batch (compiling tile program)")
y = model.predict(x=x_train, batch_size=batch_size)

# Now start the clock and run 10 batches
print("Timing inference...")
start = time.time()
for i in range(10):
    y = model.predict(x=x_train, batch_size=batch_size)
print("Ran in {} seconds".format(time.time() - start))

# CONCLUSIÓN 

En la ejecución de PLAIDML se puede ver que se trabaja en el administrador de tareas, forzando a los equipos de computo en un 80%en su aceleración de procesos así esta investigación nos lleva a que el equipo necesita tener una buena capacidad para poder implementar el código ya que se necesita por lo menos de una tarjeta gráfica de 6gb para que podamos observar bien el código 
