# Clasificación de Tejidos de Cáncer de Pulmón

## Descripción
Este proyecto implementa un sistema de clasificación automática de imágenes histopatológicas para la detección y análisis de tejidos pulmonares y colónicos, con enfoque en patrones asociados a cáncer. Utilizando técnicas avanzadas de deep learning, el modelo clasifica las muestras de tejido en categorías específicas, proporcionando información valiosa para asistir en el diagnóstico.

## Funcionalidad (v2)

La versión 2 del sistema ofrece:

- **Clasificación multiclase** en 6 categorías: adenocarcinoma de colon, colon benigno, carcinoma escamoso pulmonar, adenocarcinoma pulmonar, tejido pulmonar benigno y otros
- **Procesamiento de imágenes** mediante un modelo basado en EfficientNet-B3 preentrenado y optimizado para imágenes histopatológicas
- **Interfaz web intuitiva** que permite cargar imágenes fácilmente y visualizar resultados inmediatos
- **Visualización detallada** con gráficos de barras que muestran los porcentajes de confianza para cada categoría
- **Análisis explicativo** mediante IA que proporciona interpretación detallada de las características visibles en el tejido
- **Indicadores de confianza** que alertan cuando la predicción no alcanza un umbral adecuado

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

## Instalación y uso
1. Clone este repositorio
2. Descargue el modelo pre-entrenado del enlace de Google Drive y colóquelo en la carpeta raíz
3. Instale las dependencias: `pip install -r requirements.txt`
4. Cree un archivo .env con su clave API de OpenRouter (para la funcionalidad de explicación)
5. Ejecute el servidor: `python web_v2.py`
6. Abra su navegador y vaya a `http://127.0.0.1:5000`

---

*Nota: Este proyecto tiene fines educativos e investigativos. No debe utilizarse como herramienta de diagnóstico clínico sin la validación adecuada por profesionales médicos.*