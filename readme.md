# Clasificación de Tejidos 

## Descripción
Este proyecto implementa un sistema de clasificación automática de imágenes histopatológicas para la detección y análisis de tejidos pulmonares y colónicos, con enfoque en patrones asociados a cáncer. Utilizando técnicas avanzadas de deep learning, el modelo clasifica las muestras de tejido en categorías específicas, proporcionando información valiosa para asistir en el diagnóstico.

## Funcionalidad 

La versión del sistema ofrece:

- **Clasificación multiclase** en 6 categorías: adenocarcinoma de colon, colon benigno, carcinoma escamoso pulmonar, adenocarcinoma pulmonar, tejido pulmonar benigno y otros
- **Procesamiento de imágenes** mediante un modelo basado en EfficientNet-B3 preentrenado y optimizado para imágenes histopatológicas
- **Interfaz web intuitiva** que permite cargar imágenes fácilmente y visualizar resultados inmediatos
- **Visualización detallada** con gráficos de barras que muestran los porcentajes de confianza para cada categoría
- **Análisis explicativo** mediante IA que proporciona interpretación detallada de las características visibles en el tejido
- **Indicadores de confianza** que alertan cuando la predicción no alcanza un umbral adecuado

## Instalación y uso
Antes de comenzar, asegúrate de tener instalados los siguientes programas:

1. **Python 3.8 o superior**  
   - Verifica si ya está instalado con el siguiente comando:
     ```bash
     python --version
     ```
   - Si no lo tienes, descárgalo desde [python.org](https://www.python.org/downloads/) e instálalo. **Recuerda marcar la opción “Add Python to PATH”** durante la instalación.

2. **Git (Opcional, pero recomendado para clonar el repositorio)**  
   - Para verificar si está instalado, ejecuta:
     ```bash
     git --version
     ```
   - Si no lo tienes, descárgalo desde [git-scm.com](https://git-scm.com/downloads).

## **1. Descargar el Proyecto**
Tienes dos opciones para obtener los archivos del proyecto:

### **Opción 1: Clonar el Repositorio (Recomendado si tienes Git)**
1. Abre una terminal o línea de comandos y sitúate en la carpeta donde quieres descargar el proyecto.
2. Ejecuta el siguiente comando:
   ```bash
   git clone https://github.com/Saultr21/Clasificacion-de-imagenes-hispatolgicas.git
   cd Clasificacion-de-imagenes-hispatolgicas

### **Opción 2: Descargar el Proyecto como ZIP**
1. Ve a la página del repositorio en GitHub.
2. Haz clic en el botón **"Code"** y luego en **"Download ZIP"**.
3. Extrae el archivo en una carpeta de tu elección.

## **2. Descargar el Modelo Pre-entrenado**
El modelo necesario para la clasificación no está en el repositorio. Debes descargarlo manualmente:

1. Accede a [este enlace de Google Drive](https://drive.google.com/drive/folders/1JFx5KMTbyQyqT29bFfV8iaYK07eVa-R0?usp=sharing).
2. Descarga los archivos del modelo.
3. Copia los archivos descargados en la carpeta raíz del proyecto.

## **3. Crear un Entorno Virtual (Opcional, pero Recomendado)**
Para evitar conflictos con otras instalaciones de Python, es recomendable crear un entorno virtual:
```bash
python -m venv venv
```
Luego, actívalo:
- En **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- En **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

## **4. Instalar las Dependencias**
Una vez dentro de la carpeta del proyecto, ejecuta el siguiente comando para instalar todas las bibliotecas necesarias:
```bash
pip install -r requirements.txt
```

## **5. Configurar el Archivo `.env`**
El proyecto requiere una clave API de OpenRouter. Para configurarla:

1. Ve a [OpenRouter](https://openrouter.ai/settings/keys) y genera una clave API.
2. Descarga el archivo `env` desde el repositorio de GitHub.
3. Renómbralo a `.env` (añadiendo el punto delante del nombre).
4. Abre el archivo con un editor de texto y sustituye `TU_CLAVE_API_AQUI` por la clave API que generaste.

## **6. Ejecutar el Servidor**
Para iniciar la aplicación, ejecuta:
```bash
python web.py
```

Si todo está correcto, verás un mensaje indicando que el servidor está corriendo en `http://127.0.0.1:5000`.

## **7. Acceder a la Aplicación**
Abre un navegador web y accede a la siguiente dirección:
```
http://127.0.0.1:5000
```

Desde ahí podrás interactuar con la aplicación y probar la clasificación de tejidos.

---
### **Notas Adicionales**
- Si encuentras errores de dependencia, asegúrate de tener `pip` actualizado ejecutando:
  ```bash
  python -m pip install --upgrade pip
  ```
- Si tienes problemas con Flask, puedes ejecutarlo de forma explícita con:
  ```bash
  flask run
  ```

---


## Tecnologías utilizadas
- **Backend**: Python, Flask, PyTorch
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Modelos**: EfficientNet-B3 (para clasificación de imágenes), Gemini 2.5 Pro (para análisis explicativo)
- **Visualización**: Chart.js

## Requisitos
- Python 3.7+
- PyTorch
- Flask
- Bibliotecas adicionales: PIL, torchvision, markdown, requests
- Conexión a internet (para las explicaciones mediante API)

## Archivos adicionales
El modelo pre-entrenado (`lung_cancer_model_todos.pth`) y conjunto de imágenes de prueba están disponibles en el siguiente enlace:

[Google Drive - Archivos del proyecto](https://drive.google.com/drive/folders/1JFx5KMTbyQyqT29bFfV8iaYK07eVa-R0?usp=sharing)

---

*Nota: Este proyecto tiene fines educativos e investigativos. No debe utilizarse como herramienta de diagnóstico clínico sin la validación adecuada por profesionales médicos.*
