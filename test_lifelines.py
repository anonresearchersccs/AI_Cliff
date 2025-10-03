# %%
# Diagnóstico completo de lifelines
import sys
import os
import subprocess

print("=== DIAGNÓSTICO DE LIFELINES ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

print("\n1. Verificando instalación con pip...")
try:
    result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                          capture_output=True, text=True, check=True)
    pip_packages = result.stdout
    if "lifelines" in pip_packages:
        for line in pip_packages.split('\n'):
            if 'lifelines' in line.lower():
                print(f"✅ Encontrado en pip: {line}")
    else:
        print("❌ lifelines NO encontrado en pip list")
except Exception as e:
    print(f"❌ Error ejecutando pip list: {e}")

print("\n2. Verificando sys.path...")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

print("\n3. Intentando importar lifelines...")
try:
    import lifelines
    print(f"✅ lifelines importado exitosamente!")
    print(f"   Versión: {lifelines.__version__}")
    print(f"   Ubicación: {lifelines.__file__}")
    
    print("\n4. Probando importación específica...")
    from lifelines import KaplanMeierFitter, CoxPHFitter
    print("✅ KaplanMeierFitter y CoxPHFitter importados correctamente")
    
    print("\n5. Probando funcionalidad básica...")
    import numpy as np
    import pandas as pd
    
    # Datos de prueba
    T = np.array([1, 2, 3, 4, 5])
    E = np.array([1, 1, 1, 1, 1])
    
    kmf = KaplanMeierFitter()
    kmf.fit(T, E)
    print("✅ KaplanMeierFitter funciona correctamente")
    
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    print("\n🔧 INTENTANDO SOLUCIONES...")
    
    # Solución 1: Reinstalar
    print("\nSolución 1: Reinstalando lifelines...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "lifelines", "--force-reinstall"], 
                      check=True)
        print("✅ Reinstalación completada")
        
        # Probar importación después de reinstalar
        import lifelines
        print(f"✅ Ahora funciona! Versión: {lifelines.__version__}")
    except Exception as reinstall_error:
        print(f"❌ Error en reinstalación: {reinstall_error}")
        
        # Solución 2: Instalar sin dependencias y luego con dependencias
        print("\nSolución 2: Instalación por pasos...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "lifelines", "--no-deps"], 
                          check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "lifelines"], 
                          check=True)
            import lifelines
            print(f"✅ Solución 2 exitosa! Versión: {lifelines.__version__}")
        except Exception as e2:
            print(f"❌ Solución 2 falló: {e2}")

except Exception as e:
    print(f"❌ Error inesperado: {e}")

print("\n6. Verificando entorno de conda/anaconda...")
try:
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'No detectado')
    print(f"Entorno conda: {conda_env}")
    
    if 'conda' in sys.executable.lower() or 'anaconda' in sys.executable.lower():
        print("✅ Usando entorno conda/anaconda")
        print("💡 Sugerencia: Probar 'conda install -c conda-forge lifelines'")
    else:
        print("ℹ️ No parece ser un entorno conda")
        
except Exception as e:
    print(f"Error verificando conda: {e}")

print("\n=== FIN DEL DIAGNÓSTICO ===")
print("\nSi lifelines sigue sin funcionar después de este diagnóstico:")
print("1. Reinicia completamente Jupyter/VSCode")
print("2. Verifica que estás usando el kernel correcto")
print("3. Considera usar: conda install -c conda-forge lifelines")

