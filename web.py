#Creado por: [Saúl] - liagsad21@gmail.com 
#Github: https://github.com/Saultr21 

#Cambiar API key de openrouter por la tuya en el archivo .env
import os
import glob
from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
from torchvision.models import EfficientNet_B3_Weights
import torch.nn.functional as F
import requests
import json
import base64
import markdown  
import time  
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import io
import numpy as np
import cv2  # Necesitarás instalar opencv-python
import matplotlib
matplotlib.use('Agg')  # Necesario para usar matplotlib sin GUI

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)

# Asegurar que la carpeta para fotos existe
UPLOAD_FOLDER = 'static/foto'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Definición del modelo (debe ser la misma que usamos para entrenar)
class LungCancerModel(nn.Module):
    def __init__(self, num_classes):
        super(LungCancerModel, self).__init__()
        self.efficientnet = models.efficientnet_b3(weights=None) 
        for param in list(self.efficientnet.parameters())[:-30]:
            param.requires_grad = False
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

# Cargar el modelo reentrenado
num_classes = 6  
modelo = LungCancerModel(num_classes=num_classes)
modelo.load_state_dict(torch.load('lung_cancer_model_todos.pth', map_location=torch.device('cpu'))) 
modelo.eval()

# Transformaciones para las imágenes
transformaciones = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Diccionario de clases 
clases = {
    0: 'Adenocarcinoma de Colon',
    1: 'Colon Benigno',
    2: 'Carcinoma escamoso pulmonar',
    3: 'Adenocarcinoma pulmonar',
    4: 'Tejido pulmonar benigno',
    5: 'Otros'
}

def limpiar_carpeta_fotos():
    """Elimina todas las imágenes en la carpeta de fotos"""
    archivos = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
    for archivo in archivos:
        try:
            if os.path.isfile(archivo):
                os.remove(archivo)
                print(f"Archivo eliminado: {archivo}")
        except Exception as e:
            print(f"Error al eliminar {archivo}: {e}")

#Creado por: [Saúl] - liagsad21@gmail.com 
#Github: https://github.com/Saultr21 
def predecir(imagen_path):
    imagen = Image.open(imagen_path).convert('RGB')
    imagen_tensor = transformaciones(imagen).unsqueeze(0)
    with torch.no_grad():
        salida = modelo(imagen_tensor)
        probabilidades = F.softmax(salida, dim=1)[0] * 100
        
        # Imprimir probabilidades detalladas para depuración
        print("\nProbabilidades por clase:")
        for i, prob in enumerate(probabilidades):
            clase_nombre = clases.get(i, f"Índice {i} desconocido")
            print(f"  Clase {i} ({clase_nombre}): {prob.item():.2f}%")
        
        confianza, prediccion = torch.max(probabilidades, 0)
        print(f"Salida del modelo: {salida}")
        print(f"Probabilidades: {probabilidades}")
        print(f"Predicción (índice): {prediccion.item()}")
    
    # Generar Grad-CAM
    grad_cam_path = None
    try:
        # Usar el tensor de imagen para Grad-CAM
        imagen_tensor = imagen_tensor.to(next(modelo.parameters()).device)
        gradients, activations = get_gradients_and_features(modelo, imagen_tensor, prediccion.item())
        
        # Guardar la imagen Grad-CAM con ruta absoluta
        grad_cam_filename = f"gradcam_{os.path.basename(imagen_path)}"
        grad_cam_absolute_path = os.path.join(UPLOAD_FOLDER, grad_cam_filename)
        generate_grad_cam(gradients, activations, imagen_path, grad_cam_absolute_path)
        
        # Crear URL relativa para el navegador (sin la barra inicial)
        grad_cam_path = f"foto/{grad_cam_filename}"  # Quitar el /static/ inicial
        print(f"Grad-CAM generado en: {grad_cam_absolute_path}")
        print(f"URL para el navegador (grad_cam_path): {grad_cam_path}")
    except Exception as e:
        print(f"Error al generar Grad-CAM: {e}")
    
    return prediccion.item(), confianza.item(), probabilidades.tolist(), grad_cam_path

# Funciones para Grad-CAM (añadir después de las definiciones de clase y antes de las rutas)
def get_gradients_and_features(model, img, predicted_class):
    activation_output = None
    gradient_output = None

    # Registrar los hooks en la última capa convolucional
    final_conv_layer = model.efficientnet.features[-1]  # Última capa convolucional
    
    def save_activation_hook(module, input, output):
        nonlocal activation_output
        activation_output = output
    
    def save_gradient_hook(module, grad_input, grad_output):
        nonlocal gradient_output
        gradient_output = grad_output[0]
    
    handle_activation = final_conv_layer.register_forward_hook(save_activation_hook)
    handle_gradient = final_conv_layer.register_backward_hook(save_gradient_hook)

    # Realizar una pasada hacia adelante - ELIMINAR unsqueeze(0)
    outputs = model(img)  # Eliminar el .unsqueeze(0) aquí
    
    # Realizar la retropropagación
    model.zero_grad()
    outputs[0, predicted_class].backward()

    # Desregistrar los hooks
    handle_activation.remove()
    handle_gradient.remove()

    return gradient_output, activation_output

def generate_grad_cam(gradients, activations, img_path, save_path):
    # Calcular la importancia de las activaciones por clase
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    grad_cam = torch.sum(weights * activations, dim=1, keepdim=True)

    # Aplicar ReLU para visualizar las activaciones positivas
    grad_cam = F.relu(grad_cam)
    grad_cam = grad_cam.squeeze().cpu().detach().numpy()
    grad_cam -= np.min(grad_cam)
    grad_cam /= np.max(grad_cam)  # Normalizar entre 0 y 1

    # Cargar la imagen original para superponer
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0

    # Redimensionar el mapa de calor al tamaño de la imagen
    grad_cam_resized = np.uint8(255 * grad_cam)
    grad_cam_resized = cv2.resize(grad_cam_resized, (img_array.shape[1], img_array.shape[0]))
    
    # Crear el mapa de calor
    heatmap = cv2.applyColorMap(grad_cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Superponer el mapa de calor sobre la imagen original
    superimposed_img = 0.6 * img_array + 0.4 * heatmap
    superimposed_img = np.clip(superimposed_img, 0, 1)
    
    # Guardar la imagen
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_array)
    plt.title("Imagen Original")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title("Imagen Grad-CAM")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return save_path

@app.route('/', methods=['GET', 'POST'])
def index():
    prediccion_texto = None
    confianza = None
    confianzas = None
    imagen_path = None
    explicacion = None
    grad_cam_path = None
    
    if request.method == 'POST':
        if 'imagen' not in request.files:
            return 'No se ha subido ninguna imagen.'
        
        imagen = request.files['imagen']
        if imagen.filename == '':
            return 'No se ha seleccionado ninguna imagen.'
        
        if imagen:
            # Limpiar la carpeta antes de guardar la nueva imagen
            limpiar_carpeta_fotos()
            
            # Guardar la nueva imagen
            imagen_path = os.path.join(UPLOAD_FOLDER, imagen.filename)
            imagen.save(imagen_path)
            
            # Ahora la función predecir también devuelve la ruta del Grad-CAM
            prediccion, confianza, confianzas, grad_cam_path = predecir(imagen_path)
            
            # Manejar el caso donde la predicción no está en el diccionario
            try:
                prediccion_texto = clases.get(prediccion, "Desconocido")
            except Exception as e:
                print(f"Error al obtener la clase: {e}")
                prediccion_texto = "Desconocido"
    
    # Si tienes confianzas, prepara un texto formateado
    texto_confianzas = None
    if confianzas:
        texto_confianzas = []
        for i, conf in enumerate(confianzas):
            nombre_clase = clases.get(i, f"Clase {i}")
            texto_confianzas.append({
                "nombre": nombre_clase,
                "valor": f"{conf:.2f}%"
            })
    
    return render_template('index.j2', 
                          prediccion=prediccion_texto, 
                          confianza=confianza, 
                          confianzas=confianzas,
                          texto_confianzas=texto_confianzas,
                          imagen_path=imagen_path,
                          grad_cam_path=grad_cam_path,
                          explicacion=explicacion,
                          clases=clases)

# Añadir esta ruta después de la ruta index y antes de la función generar_explicacion
@app.route('/generar_explicacion', methods=['POST'])
def obtener_explicacion():
    try:
        datos = request.json
        clasificacion = datos.get('clasificacion')
        confianza = float(datos.get('confianza'))
        imagen_path = datos.get('imagen_path')
        
        print(f"Imagen path recibido: {imagen_path}")
        
        # Validar que tenemos todos los datos necesarios
        if not all([clasificacion, confianza, imagen_path]):
            return jsonify({"error": "Faltan datos requeridos"}), 400
        
        # Limpiar espacios extras en la ruta
        imagen_path = imagen_path.replace('static/foto', 'static/foto/')
        
        # Eliminar espacios extras entre la ruta y el nombre del archivo
        partes = imagen_path.split('/')
        if len(partes) >= 3:
            partes[-1] = partes[-1].strip()  # Eliminar espacios al inicio y fin del nombre
            imagen_path = '/'.join(partes)
        
        print(f"Imagen path corregido: {imagen_path}")
        
        # Si el archivo no existe, buscar todos los archivos en la carpeta para depuración
        if not os.path.exists(imagen_path):
            print(f"Listado de archivos en carpeta: {os.listdir('static/foto/')}")
            
            # Intento de búsqueda alternativa por nombre similar
            nombre_archivo = partes[-1].replace(' ', '')  # Eliminar todos los espacios
            archivos_en_carpeta = os.listdir('static/foto/')
            
            for archivo in archivos_en_carpeta:
                if nombre_archivo.lower() in archivo.lower().replace(' ', ''):
                    imagen_path = os.path.join('static/foto', archivo)
                    print(f"¡Archivo encontrado por nombre similar!: {imagen_path}")
                    break
        
        # Verificar que el archivo existe
        if not os.path.exists(imagen_path):
            return jsonify({"error": f"Imagen no encontrada en: {imagen_path}"}), 404
        
        # Generar la explicación
        explicacion = generar_explicacion(clasificacion, confianza, imagen_path)
        
        # Convertir Markdown a HTML
        explicacion_html = markdown.markdown(explicacion)
        
        return jsonify({"explicacion": explicacion_html})
    except Exception as e:
        print(f"Error al generar explicación: {e}")
        return jsonify({"error": str(e)}), 500
    
#Creado por: [Saúl] - liagsad21@gmail.com 
#Github: https://github.com/Saultr21 
def generar_explicacion(clasificacion, confianza, imagen_path):
    """Generar una explicación para la clasificación utilizando OpenRouter"""
    # Modificar la pregunta para que analice independientemente la imagen
    pregunta = f"""Analiza la siguiente imagen y clasifícala en una de las siguientes categorías:
        - Adenocarcinoma de Colon
        - Colon Benigno
        - Carcinoma escamoso pulmonar
        - Adenocarcinoma pulmonar
        - Tejido pulmonar benigno
        - Otro tejido diferente (relacionado con patología)
        - Otro tipo de imagen (no patológica)

        Si estás seguro de una categoría patológica, indica cuál es y argumenta brevemente tu decisión basándote en las características visuales.
        Si dudas entre varias categorías patológicas, indica los porcentajes de probabilidad estimados para cada una.
        Si identificas 'Otro tejido diferente', descríbelo brevemente si es posible y a que podría pertenecer.
        Si es 'Otro tipo de imagen', describe de qué se trata.
        Mantén la respuesta en español y entre 15 y 30 líneas.
    """
    
    max_intentos = 5  # Número máximo de intentos
    intentos = 0
    tiempo_espera = 2  # Segundos iniciales de espera entre intentos
    
    while intentos < max_intentos:
        try:
            respuesta = ask_openrouter_with_base64_es(pregunta, imagen_path)
            
            # Si la respuesta indica que no se recibió contenido, reintentar
            if respuesta == "No se recibió respuesta":
                intentos += 1
                print(f"Intento {intentos}/{max_intentos} falló. Reintentando en {tiempo_espera} segundos...")
                time.sleep(tiempo_espera)
                tiempo_espera *= 2  # Incremento exponencial del tiempo de espera
            else:
                # Si tenemos una respuesta válida, salir del bucle
                return respuesta
                
        except Exception as e:
            intentos += 1
            print(f"Error en intento {intentos}/{max_intentos}: {e}. Reintentando en {tiempo_espera} segundos...")
            time.sleep(tiempo_espera)
            tiempo_espera *= 2  # Incremento exponencial del tiempo de espera
    
    # Si llegamos aquí es porque agotamos todos los intentos
    return "No se pudo generar una explicación después de varios intentos. Por favor, inténtelo de nuevo más tarde o consulte con un especialista para interpretar estos resultados."

def ask_openrouter_with_base64_es(question, image_path):
    """Versión que incluye mensaje de sistema para forzar respuesta en español"""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}", # Cambia por tu API key en el archivo .env
        "Content-Type": "application/json",
    }
    
    # Leer la imagen y convertirla a base64
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Formato base64 para la API
    base64_image = f"data:image/jpeg;base64,{encoded_string}"

    #Creado por: [Saúl] - liagsad21@gmail.com 
    #Github: https://github.com/Saultr21 
    # Incluir mensaje de sistema para forzar español
    messages = [
        {"role": "system", "content": "Eres un asistente experto en patología, especializado en el análisis de imágenes histológicas. Tu tarea es examinar la imagen proporcionada y clasificarla según las categorías especificadas por el usuario. Debes basar tu análisis únicamente en las características visuales presentes en la imagen. Responde siempre en español y asegúrate de que tu respuesta total sea entre 15 y 30 líneas. Prioriza la precisión y, si no estás seguro, indica las probabilidades. Si la imagen no corresponde a ninguna categoría patológica relevante, identifícala como tal."},
        {"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": base64_image}}
        ]}
    ]

    data = json.dumps({
        "model": "google/gemini-2.0-flash-exp:free",
        "messages": messages
    })

    response = requests.post(url, headers=headers, data=data)
    # Extraer el contenido del mensaje del asistente
    response_json = response.json()
    # Verificar si la respuesta es válida
    if response.status_code != 200:
        print(f"Error HTTP: {response.status_code}")
        print(f"Respuesta: {response_json}")
        return "No se recibió respuesta"
    content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "No se recibió respuesta")
    return content


if __name__ == '__main__':
    app.run(debug=True)
