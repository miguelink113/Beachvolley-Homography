# Beachvolley-Homography

**Proyecto:** Homografía y seguimiento de balón/jugadores en voleyplaya en vídeo

**Autores:** Miguel Castellano Hernández,  Wail Ben El Hassane Boudhar, Gorka Eymard Santana Cabrera 
**Resumen:** Este proyecto implementa pipelines para la detección y seguimiento de balón y jugadores en vídeos de beach volley, estima la homografía entre la pista y una vista superior (mapa), y genera salidas en CSV/JSON con coordenadas mapeadas, tracks y visualizaciones.

---

## 📋 Rápido (Carátula)
- **Lenguaje:** Python 3.11.4
- **Dependencias principales:** OpenCV, NumPy, Pandas, PyTorch (YOLO, SAM)
- **Checkpoints:** `yolo11n.pt`, `checkpoints/sam2.1_hiera_tiny.pt` (no incluidos en PyPI; ver carpeta `checkpoints/`)
- **Datos de salida:** `Output/*.json`, `Output/*.csv`

## ⚡ Quick Start
1. Crear un entorno virtual (recomendado):

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Abrir notebooks principales (ejemplos):
   - `VC_P6.ipynb` — análisis y demo
   - `player_tracking.ipynb` — seguimiento de jugadores
   - `ball_tracking.ipynb` — seguimiento de balón

3. Ejecutar pipelines o scripts (próximamente en `src/` como CLI):

   ```bash
   # ejemplo conceptual
   python -m src.run --video VideosAnalisis/mi_video.mp4 --out Output/tracking_data.csv
   ```

---

## 🧭 Checkpoints y datos grandes
- Si los checkpoints no están en el repositorio, descargarlos según las instrucciones del proyecto o colocarlos en `checkpoints/` y `yolo11n.pt` en la raíz.
- Ten en cuenta que los pesos y algunos vídeos pueden ser grandes; usar Git LFS si es necesario.

## 📤 Salidas esperadas
- `Output/tracking_data.csv` — columnas: `frame`, `id`, `x`, `y`, `x_map`, `y_map`, `score`, `class` (confirmar con los scripts reales)
- `Output/puntos_mapa.json` — puntos de referencia para homografía

---

## 🧪 Recomendaciones
- Mantener notebooks como exploración y mover la lógica reusable a `src/` (propuesta de refactor disponible).
- Añadir tests unitarios para funciones determinísticas (IOU, homografía, merge de tracks).

---

## 📬 Contacto
Si tienes dudas o quieres colaborar, abre un issue o PR en este repositorio.

---

*Fecha de la carátula: 2026-01-08*
