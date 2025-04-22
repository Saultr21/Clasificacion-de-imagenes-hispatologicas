@echo off
setlocal enabledelayedexpansion
title Configuración del Proyecto - Clasificación de Tejidos
:: Habilitar colores ANSI si no están activados
reg query HKCU\Console /v VirtualTerminalLevel >nul 2>&1 || (
    reg add HKCU\Console /v VirtualTerminalLevel /t REG_DWORD /d 1 /f >nul
)

:: Paso 0: Verificar si el archivo .env existe
if not exist .env (
    echo ❌ ERROR: No se encontró el archivo .env
    echo Crea uno con la línea:
    echo OPENROUTER_API_KEY=tu_clave_de_api
    pause
    exit /b
)

:: Paso 0.1: Verificar si la clave API ha sido configurada (distinta de 'tu_clave_de_api')
set "APICONFIG=0"
for /f "usebackq tokens=* delims=" %%a in (".env") do (
    set "linea=%%a"
    echo !linea! | findstr /R "^OPENROUTER_API_KEY=sk-or-" >nul
    if !errorlevel! == 0 (
        set "APICONFIG=1"
    )
)

:: Paso 1: Crear entorno virtual
echo [1/5] Creando entorno virtual...
python -m venv venv
if errorlevel 1 (
    echo Error al crear el entorno virtual. ¿Tienes Python instalado?
    pause
    exit /b
)

:: Paso 2: Activar entorno virtual
echo [2/5] Activando entorno virtual...
call venv\Scripts\activate

:: Paso 3: Actualizar pip
echo [3/5] Actualizando pip...
python -m pip install --upgrade pip

:: Paso 4: Instalar dependencias
echo [4/5] Instalando dependencias...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error al instalar dependencias. Revisa el archivo requirements.txt.
    pause
    exit /b
)

:: Paso 5: Verificar y mostrar estado de la API
if "%APICONFIG%"=="0" (
    echo =========================================================
    echo =========================================================
    echo =========================================================
    echo [91mATENCION: No se encontro una clave OPENROUTER_API_KEY configurada en el archivo .env[0m
    echo [91mPuedes obtenerla aqui: https://openrouter.ai/settings/keys[0m
    echo [91mAbre el archivo .env y edita la siguiente linea:[0m
    echo [91mOPENROUTER_API_KEY=tu_clave_aqui[0m
    echo =========================================================
    echo =========================================================
    echo =========================================================
) else (
    echo =========================================================
    echo =========================================================
    echo =========================================================
    echo [91mClave API detectada correctamente en el archivo .env[0m
    echo =========================================================
    echo =========================================================
    echo =========================================================
)

:: Paso 6: Ejecutar aplicación
echo [5/5] Iniciando servidor Flask...
python web.py



:: Mantener ventana abierta
pause
