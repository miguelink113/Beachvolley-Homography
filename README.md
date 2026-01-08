# Beachvolley-Homography

Proyecto de vision por computador para analizar partidos de voley playa desde video. El sistema detecta la pelota, detecta el campo, realiza la homografia hacia una vista cenital y hace seguimiento de jugadores para generar datos accionables y visualizaciones limpias.

Autores: Miguel, Gorka y Wail

---

## Presentacion del proyecto
Beachvolley-Homography nace para convertir video real de voley playa en informacion medible: trayectorias, posiciones y eventos clave. La idea es ofrecer una base tecnica lista para analisis deportivo, scouting, metricas de rendimiento y futuras aplicaciones de arbitraje asistido.

Valor diferencial:
- Pipeline completo de deteccion y seguimiento con mapeo a un plano de cancha.
- Salidas en formatos estandar (CSV/JSON) listas para analitica.
- Visualizaciones de tracks sobre video y sobre mapa.

---

## Componentes principales

### 1) Deteccion y seguimiento de pelota
Objetivo: localizar la pelota en cada frame y reconstruir su trayectoria.

Flujo general:
- Deteccion por modelo (YOLO) en cada frame.
- Filtrado por confianza y tamano para reducir falsos positivos.
- Asociacion temporal con criterio de distancia/IOU para construir el track.
- Suavizado opcional de la trayectoria para eliminar saltos.

Detalle tecnico paso a paso:
- Preprocesado: el video se lee frame a frame y se ajusta a la resolucion esperada del modelo.
- Inferencia: YOLO devuelve bounding boxes con clase "ball" y una puntuacion de confianza.
- Filtrado: se descartan detecciones fuera de rango de tamano o con score bajo.
- Seleccion: si hay varias detecciones candidatas, se elige la mas coherente con la posicion anterior.
- Tracking: se enlazan detecciones entre frames usando distancia euclidea y/o IOU.
- Interpolacion: cuando falta una deteccion puntual, se interpolan posiciones para mantener continuidad.
- Suavizado: se aplica una media movil o filtro simple para evitar jitter.

Salida tipica:
- CSV con `frame`, `x`, `y`, `score` y/o un track continuo.
- Video de depuracion con bounding box y trayectoria superpuesta.

Notas:
- La pelota es pequeña y rapida: ajustar el umbral de confianza puede cambiar mucho el recall.
- En escenas con poca luz o mucho blur, conviene priorizar recall y limpiar con post-procesado.

---

### 2) Deteccion del campo y homografia
Objetivo: identificar el campo y mapear cualquier punto del video a una vista cenital (mapa).

Flujo general:
- Deteccion de lineas y/o puntos clave del campo.
- Seleccion de correspondencias entre el plano de imagen y el plano del campo.
- Calculo de homografia con RANSAC o ajuste robusto.
- Transformacion de coordenadas de pelota y jugadores al sistema del mapa.

Detalle tecnico paso a paso:
- Extraccion de bordes: se resaltan lineas blancas del campo con filtros (Canny/threshold).
- Deteccion de lineas: Hough u otro metodo para encontrar segmentos y su interseccion.
- Puntos clave: se identifican esquinas y lineas centrales como anclas geometricas.
- Correspondencias: se asocian puntos del video con los del mapa ideal de la cancha.
- Homografia: se estima la matriz 3x3 que transforma coordenadas de imagen a cancha.
- Validacion: se revisa el error de reproyeccion y se ajustan puntos si es necesario.
- Proyeccion: cualquier coordenada (pelota/jugadores) se transforma a `x_map`, `y_map`.

Salida tipica:
- `map_points.json` con puntos de referencia y metadatos.
- Coordenadas transformadas `x_map`, `y_map` en CSV.
- Imagen/overlay de la cancha con la proyeccion de tracks.

Notas:
- La calidad de la homografia depende de la estabilidad de la camara.
- Se recomienda fijar 4+ puntos bien distribuidos (esquinas y lineas centrales).

---

### 3) Seguimiento de jugadores
Objetivo: detectar jugadores, asignarles IDs y mantener su identidad en el tiempo.

Flujo general:
- Deteccion de jugadores por modelo (YOLO o similar).
- Asociacion multi-objeto con criterios de distancia y/o apariencia.
- Manejo de oclusiones y cambios de escala.
- Proyeccion al mapa mediante homografia para analisis espacial.

Detalle tecnico paso a paso:
- Deteccion: se obtienen bounding boxes de la clase "person".
- Filtrado: se descartan cajas fuera del area de juego o con score bajo.
- Asociacion temporal: se calcula distancia centro-a-centro entre frames consecutivos.
- Asignacion de ID: el mejor match hereda el ID anterior; si no hay match, se crea uno nuevo.
- Oclusiones: se permite una ventana de frames sin deteccion antes de cerrar un track.
- Proyeccion: se usa la homografia para ubicar cada jugador en el mapa.
- Agregacion: se pueden generar mapas de calor o zonas de ocupacion.

Salida tipica:
- CSV con `frame`, `id`, `x`, `y`, `x_map`, `y_map`, `score`, `class`.
- Video con bounding boxes e IDs.
- Mapa con zonas de ocupacion y recorridos.

Notas:
- Si hay cambios bruscos de camara, puede fallar la asignacion de IDs.
- La separacion por equipos puede integrarse mas adelante con color/jersey.

---

## Resultados y salidas
Este repositorio genera:
- CSV de tracks de pelota y jugadores.
- JSON con puntos de referencia para la homografia.
- Videos con anotaciones y mapas con trayectorias.

Ejemplos en el repo:
- `ball_track_edges.csv`
- `ball_track_yolo.csv`
- `map_points.json`
- `ball_track_edges.mp4`
- `ball_track_yolo.mp4`

---

## Como usar el proyecto

### Requisitos
Este proyecto usa Python 3.11+. Las dependencias estan en `requirements.txt`:
- numpy, opencv-python, pandas, matplotlib, tqdm
- torch, torchvision
- ultralytics (YOLO)

Instalacion recomendada:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Ejecucion (notebooks)
El flujo principal esta en notebooks:
- `VC_P6.ipynb` (analisis general y demo)
- `ball_tracking.ipynb` (pelota)
- `player_tracking.ipynb` (jugadores)
- `analysis.ipynb` (analisis y visualizaciones)

Abrir con Jupyter o VS Code y ejecutar celda a celda.

### Pesos y modelos
Coloca los pesos en las rutas esperadas:
- `yolo11n.pt` en la raiz del repo.
- `checkpoints/` para pesos auxiliares (ej. SAM).

Si no estan disponibles, el pipeline puede fallar o quedar incompleto.

---

## Indicaciones para imagenes y videos
Deja los recursos visuales en `docs/media/` (sugerencia). Agrega enlaces en estas secciones:

### Capturas del proyecto
```markdown
![Vista general](docs/media/overview.png)
![Deteccion de pelota](docs/media/ball_detection.png)
![Deteccion de campo](docs/media/court_detection.png)
![Seguimiento de jugadores](docs/media/player_tracking.png)
```

### Videos por modulo
```markdown
[Video: Deteccion de pelota](docs/media/ball_demo.mp4)
[Video: Deteccion de campo](docs/media/court_demo.mp4)
[Video: Seguimiento de jugadores](docs/media/player_demo.mp4)
```

### Video final del resultado
```markdown
[Video final](docs/media/final_result.mp4)
```

Tip: si el repositorio no admite archivos grandes, enlaza a un drive o plataforma externa.

---

## Vision a futuro
- Integrar re-identificacion para mejorar el seguimiento en oclusiones.
- Separar equipos y roles por color de camiseta y posicion.
- Deteccion de eventos: saque, remate, bloqueo, puntos y errores.
- Panel web con analiticas y exportacion de clips.
- Optimizacion para ejecucion en tiempo real.

---

## Contacto
Para colaboraciones o dudas, abre un issue o escribe a los autores.
