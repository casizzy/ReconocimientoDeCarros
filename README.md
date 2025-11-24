Sistema de Detección y Conteo de Vehículos (Basado en ROIs)

Este repositorio contiene la implementación completa de un sistema capaz de detectar y contar vehículos utilizando únicamente técnicas de procesamiento digital de imágenes.
El proyecto analiza un video de entrada de estacionamiento y detecta el paso de autos mediante cambios de intensidad dentro de Regiones de Interés (ROIs), sin usar modelos de inteligencia artificial.

Estructura del Proyecto

src/

│

├── main.py             # Archivo principal para ejecutar el sistema

├── contar_autos.py     # Lógica de detección y análisis de ROIs

├── mpi.py              # Funciones de procesamiento de imagen (grises, filtros, convolución)

├── data/               # Carpeta para el video e imágenes de prueba

└── __pycache__/        # Caché automático de Python


frame_test.jpg          # Fotograma de ejemplo usado para pruebas y documentación

pyvenv.cfg              # Configuración del entorno virtual

.gitignore              # Exclusiones de Git


Contenido del Repositorio
- Código fuente completo
- Carpeta con material de prueba (video/imágenes)
- Reporte final en PDF
- Presentación final en PDF

Descripción del Funcionamiento
- El sistema carga el video y reduce el tamaño de cada cuadro.
- Convierte cada imagen a escala de grises mediante funciones implementadas en mpi.py.
- Analiza la intensidad promedio de dos ROIs horizontales.
- Cuando el cambio en la intensidad es claro y consistente, se detecta el paso de un vehículo.
- Se incrementa el contador y se resalta visualmente la región donde ocurrió el evento.

Requisitos
- Python 3.x
- OpenCV
- NumPy







