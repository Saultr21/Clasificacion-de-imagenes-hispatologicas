// Función para validar el formulario
function validateForm() {
    const fileInput = document.getElementById('inputGroupFile02');
    const errorAlert = document.getElementById('error-no-file');
    
    if (fileInput.files.length === 0) {
        errorAlert.style.display = 'block';
        return false;
    }
    
    return true;
}
// Creado por: [Saúl] - liagsad21@gmail.com 
//  Github: https://github.com/Saultr21 
// Cuando el documento esté listo
document.addEventListener('DOMContentLoaded', function() {
    // Limpiar mensaje de error cuando se selecciona un archivo
    const fileInput = document.getElementById('inputGroupFile02');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            document.getElementById('error-no-file').style.display = 'none';
        });
    }
    
    // Inicializar el gráfico si existe el elemento
    const chartCanvas = document.getElementById('prediccionChart');
    if (chartCanvas) {
        // Comprobar si existen las variables necesarias
        if (typeof confianzas === 'undefined') {
            console.warn("Variable 'confianzas' no definida");
            confianzas = [0, 0, 0, 0, 0, 0];
        }
        
        if (typeof labels === 'undefined') {
            console.warn("Variable 'labels' no definida");
            labels = [
                'Adeno. de colon', 
                'Colon benigno', 
                'Carc. escamoso\npulmonar', 
                'Adeno. pulmonar', 
                'Tejido pulm.\nbenigno', 
                'Otros'
            ];
        }
        
        const ctx = chartCanvas.getContext('2d');
        var chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    data: confianzas,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.7)',    // Rojo para adenocarcinoma de colon
                        'rgba(255, 205, 86, 0.7)',    // Amarillo para colon benigno
                        'rgba(54, 162, 235, 0.7)',    // Azul para carcinoma escamoso pulmonar
                        'rgba(153, 102, 255, 0.7)',   // Morado para adenocarcinoma pulmonar
                        'rgba(75, 192, 192, 0.7)',    // Verde azulado para tejido pulmonar benigno
                        'rgba(201, 203, 207, 0.7)'    // Gris para otros
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(255, 205, 86, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(153, 102, 255, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(201, 203, 207, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: {
                    padding: {
                        bottom: 10  // Aumentar padding inferior
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.raw.toFixed(1) + '%';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        },
                        title: {
                            display: true,
                            text: 'Confianza'
                        }
                    },
                    x: {
                        title: {
                            display: false,
                            text: ''
                        },
                        ticks: {
                            maxRotation: 0,
                            minRotation: 0
                        }
                    }
                }
            }
        });
    }
});

// Añadir al final del archivo

// Manejo de consultas a la IA
document.addEventListener('DOMContentLoaded', function() {
    const sendQueryBtn = document.getElementById('sendQueryBtn');
    const queryInput = document.getElementById('aiQueryInput');
    const responseContainer = document.getElementById('responseContainer');
    const responseText = document.getElementById('responseText');
    const responseStatus = document.getElementById('responseStatus');
    
    if (sendQueryBtn && queryInput) {
        sendQueryBtn.addEventListener('click', function() {
            const question = queryInput.value.trim();
            if (!question) {
                alert('Por favor, introduce una pregunta.');
                return;
            }
            // Creado por: [Saúl] - liagsad21@gmail.com 
            //  Github: https://github.com/Saultr21 
            // Obtener la ruta de la imagen actual
            const imagePath = document.querySelector('.img-analysis')?.src;
            if (!imagePath) {
                alert('No hay imagen para analizar.');
                return;
            }
            
            // Mostrar el contenedor de respuesta y indicador de carga
            responseContainer.classList.remove('d-none');
            responseText.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Cargando...</span></div>';
            responseStatus.innerText = 'Procesando...';
            responseStatus.style.backgroundColor = 'rgba(255, 193, 7, 0.2)';
            
            // Realizar la solicitud al servidor
            fetch('/query_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    image_path: imagePath.replace(window.location.origin + '/', '')
                })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Error en la respuesta del servidor');
                }
                return response.json();
            })
            .then(data => {
                // Actualizar el contenedor de respuesta
                responseText.innerText = data.response;
                responseStatus.innerText = 'Completado';
                responseStatus.style.backgroundColor = 'rgba(40, 167, 69, 0.2)';
            })
            .catch(error => {
                console.error('Error:', error);
                responseText.innerText = 'Ocurrió un error al procesar tu consulta. Por favor, inténtalo de nuevo.';
                responseStatus.innerText = 'Error';
                responseStatus.style.backgroundColor = 'rgba(220, 53, 69, 0.2)';
            });
        });
        
        // También permitir enviar con Enter
        queryInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendQueryBtn.click();
                e.preventDefault();
            }
        });
    }
});

// Añadir al final del archivo

// Función para solicitar explicación
function solicitarExplicacion(clasificacion, confianza, imagen_path) {
    // Mostrar mensaje de carga
    document.getElementById('explicacion-loading').style.display = 'block';
    document.getElementById('explicacion-content').innerHTML = '';
    document.getElementById('explicacion-btn').disabled = true;
    
    // Limpiar la ruta de la imagen si es necesario
    // Si la ruta tiene la URL completa (http://...), extraer solo la parte relativa
    if (imagen_path && imagen_path.includes('://')) {
        const urlObj = new URL(imagen_path);
        imagen_path = urlObj.pathname;
    }
    
    // Asegurarse de que la ruta tiene el formato correcto para la carpeta static/foto
    if (imagen_path.includes('static/foto') && !imagen_path.includes('static/foto/')) {
        imagen_path = imagen_path.replace('static/foto', 'static/foto/');
    }
    
    // Realizar la petición AJAX
    fetch('/generar_explicacion', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            clasificacion: clasificacion,
            confianza: parseFloat(confianza),
            imagen_path: imagen_path
        })
    })
    .then(response => response.json())
    .then(data => {
        // Ocultar mensaje de carga
        document.getElementById('explicacion-loading').style.display = 'none';
        
        // Mostrar la explicación
        if (data.explicacion) {
            document.getElementById('explicacion-content').innerHTML = data.explicacion;
        } else if (data.error) {
            document.getElementById('explicacion-content').innerHTML = '<p class="text-danger">Error: ' + data.error + '</p>';
        }
        // Creado por: [Saúl] - liagsad21@gmail.com 
        //  Github: https://github.com/Saultr21 
        // Habilitar el botón nuevamente
        document.getElementById('explicacion-btn').disabled = false;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('explicacion-loading').style.display = 'none';
        document.getElementById('explicacion-content').innerHTML = '<p class="text-danger">Error al conectar con el servidor</p>';
        document.getElementById('explicacion-btn').disabled = false;
    });
}