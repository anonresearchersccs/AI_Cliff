# %%
# CELDA DE VERIFICACIÓN DE LIFELINES - Copia esto en tu notebook

print("=== VERIFICACIÓN LIFELINES ===")

try:
    # Intento de importación
    import lifelines
    from lifelines import KaplanMeierFitter, CoxPHFitter
    
    print(f"✅ lifelines {lifelines.__version__} importado correctamente")
    print("✅ KaplanMeierFitter y CoxPHFitter disponibles")
    
    # Prueba rápida de funcionalidad
    import numpy as np
    np.random.seed(42)
    T = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    E = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1, 0])
    
    kmf = KaplanMeierFitter()
    kmf.fit(T, E)
    
    print(f"✅ Prueba funcional exitosa")
    print(f"   Tiempo mediano de supervivencia: {kmf.median_survival_time_:.2f}")
    print("\n🎉 LIFELINES FUNCIONA PERFECTAMENTE")
    print("   Puedes proceder con el análisis de supervivencia")
    
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    print("\n🔧 SOLUCIONES:")
    print("1. Reinicia el kernel de Jupyter/VSCode completamente")
    print("2. Si persiste, ejecuta en una celda:")
    print("   import sys")
    print("   !{sys.executable} -m pip install --force-reinstall lifelines")
    print("3. Luego reinicia el kernel nuevamente")
    
except Exception as e:
    print(f"❌ Error inesperado: {e}")
    print("💡 Verifica que todas las dependencias estén instaladas")

print("="*50)

