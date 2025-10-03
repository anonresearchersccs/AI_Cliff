# %%
# Verificación de lifelines - Celda de diagnóstico
print("=== VERIFICACIÓN DE LIFELINES ===")

try:
    import lifelines
    print(f"✅ lifelines {lifelines.__version__} importado correctamente")
    
    from lifelines import KaplanMeierFitter, CoxPHFitter
    print("✅ KaplanMeierFitter y CoxPHFitter importados")
    
    # Prueba rápida con datos sintéticos
    import numpy as np
    import pandas as pd
    
    # Datos de prueba
    np.random.seed(42)
    n = 100
    T = np.random.exponential(10, n)  # tiempos de supervivencia
    E = np.random.binomial(1, 0.8, n)  # eventos observados
    
    # Probar Kaplan-Meier
    kmf = KaplanMeierFitter()
    kmf.fit(T, E)
    median_survival = kmf.median_survival_time_
    print(f"✅ KM funciona - Tiempo mediano de supervivencia: {median_survival:.2f}")
    
    # Probar Cox PH
    df = pd.DataFrame({
        'T': T,
        'E': E,
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.binomial(1, 0.5, n)
    })
    
    cph = CoxPHFitter()
    cph.fit(df, duration_col='T', event_col='E')
    print("✅ Cox PH funciona correctamente")
    
    print("\n🎉 LIFELINES ESTÁ COMPLETAMENTE FUNCIONAL")
    print("Puedes proceder con el análisis de supervivencia en tu notebook")
    
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    print("💡 Solución: Reinicia el kernel de Jupyter/VSCode")
    
except Exception as e:
    print(f"❌ Error inesperado: {e}")
    print("💡 Verifica la instalación de las dependencias")

print("\n" + "="*50)

