# Definir operaciones de preparación para posterior ejecución en secuencia

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import chi2_contingency

pd.set_option('future.no_silent_downcasting', True)

# ------------------------------------------------------------------------------------------------------------
def tx_depurar_datos_nulos(df):
    # Solo estas columnas ameritan borrado completo del registro si son nulas
    columnas_requeridas = [
        'CALIFICACION', 'FECHA_INST', 'FECHA_CALIFICACION', 'DIAMETRO', 'LONGITUD', 'PROF_BATEA', 'PROF_BATE1',
    ]
    df = df.dropna(subset=columnas_requeridas)

    # Columnas categoricas no requeridas se rellenan con N/D (No disponible)
    columnas_categoricas = df.select_dtypes(include=['object']).columns
    
    # df[columnas_categoricas] = df[columnas_categoricas].fillna('N/D') genera warning. cambio por df.loc
    df.loc[:, columnas_categoricas] = df[columnas_categoricas].fillna('N/D')


    return df

# ------------------------------------------------------------------------------------------------------------
def tx_convertir_tipo_datos(df):
    df['CALIFICACION'] = df['CALIFICACION'].astype(int)
    df['FECHA_INST'] = pd.to_datetime(df['FECHA_INST'])
    df['FECHA_CALIFICACION'] = pd.to_datetime(df['FECHA_CALIFICACION'])

    return df

# ------------------------------------------------------------------------------------------------------------
def tx_validar_rango_datos(df):

    # Solo son válidos registros con CALIFICACION entre 1 y 5
    df = df[df['CALIFICACION'].between(1, 5)]

    # Rango válido de diametro
    df = df[df['DIAMETRO'].between(100, 2000)]

    # Rango válido de longitud
    df = df[df['LONGITUD'].between(1, 250)]

    # Rango de profundidades de batea
    df = df[df['PROF_BATEA'] <= 15]
    df = df[df['PROF_BATE1'] <= 15]

    # Rango de fechas
    fecha_actual = datetime.now()
    df = df[df['FECHA_INST'].between('1923-01-01', fecha_actual)]
    df = df[df['FECHA_CALIFICACION'].between('1923-01-01', fecha_actual)]

    return df

# ------------------------------------------------------------------------------------------------------------
def tx_unificar_unidades(df):
    df['DIAMETRO'] = df['DIAMETRO'] / 1000 # Convertir milimetros a metros

    return df

# ------------------------------------------------------------------------------------------------------------
def tx_agregar_edades(df):
    df['EDAD'] = abs((df['FECHA_CALIFICACION'] - df['FECHA_INST'])).dt.days / 365.25

    #fecha_actual = datetime.now()
    #segundos_del_ano = 365 * 24 * 60 * 60
    #df['EDAD'] = df['FECHA_INST'].apply(
    #    lambda fecha: (fecha_actual - fecha).total_seconds() / segundos_del_ano
    #)

    return df
# ------------------------------------------------------------------------------------------------------------
def tx_agregar_areas(df):
    df['AREA'] = df['LONGITUD'] * df['DIAMETRO'] * np.pi

    return df

# ------------------------------------------------------------------------------------------------------------
def tx_recalcular_pendientes(df):
    df['PENDIENTE'] = (df['PROF_BATEA'] - df['PROF_BATE1']) / df['LONGITUD']
    return df

# ------------------------------------------------------------------------------------------------------------
def tx_binarizar_calificacion(df):
    pd.set_option('future.no_silent_downcasting', True)
    df['DETERIORADO'] = df['CALIFICACION'].replace({1: False, 2: False, 3: False, 4: True, 5: True}).astype(bool)
    return df

# ------------------------------------------------------------------------------------------------------------
def tx_reasignar_materiales(df):
    reemplazo_materiales = {
        'NOVAFORT': 'PVC',
        'NOVALOC' : 'PVC',
        'PVC'     : 'PVC',
        'CONCRETO CLASE 1'  : 'CONCRETO SIMPLE',
        'CONCRETO CLASE 2'  : 'CONCRETO SIMPLE',
        'CONCRETO CLASE 3'  : 'CONCRETO SIMPLE',
        'CONCRETO CLASE I'  : 'CONCRETO REFORZADO Y OTROS',
        'CONCRETO CLASE II' : 'CONCRETO REFORZADO Y OTROS',
        'CONCRETO CLASE III': 'CONCRETO REFORZADO Y OTROS',
        'CONCRETO CLASE IV' : 'CONCRETO REFORZADO Y OTROS',
        'CONCRETO CLASE V'  : 'CONCRETO REFORZADO Y OTROS',
    }

    df['MATERIAL'] = np.where(
        df['MATERIAL'].isin(reemplazo_materiales.keys()),
        df['MATERIAL'].map(reemplazo_materiales),
        'CONCRETO REFORZADO Y OTROS'
    )

    return df

# ------------------------------------------------------------------------------------------------------------
def tx_reasignar_fabricantes(df):
    reemplazo_fabricantes = {
        'INDUSTRIAS DIQUE S.A.'    : 'INDUSTRIAS DIQUE S.A.',
        'MEXICHEM COLOMBIA S.A.S.' : 'MEXICHEM COLOMBIA S.A.S.',
    }

    df['FABRICANTE'] = np.where(
        df['FABRICANTE'].isin(reemplazo_fabricantes.keys()),
        df['FABRICANTE'].map(reemplazo_fabricantes),
        'OTROS'
    )

    return df

# -----------------------------------------------------------------------------------------------------------
def tx_reasignar_tipo_agua(df):
    reemplazo_tipo_agua = {
        'COMBINADAS'         : 'COMBINADAS',
        'LLUVIAS'            : 'LLUVIAS Y DESCARGAS',
        'DESCARGA DE PLANTA' : 'LLUVIAS Y DESCARGAS',
        'DESCARGA DE TANQUE' : 'LLUVIAS Y DESCARGAS',
    }

    df['TIPO_AGUA'] = np.where(
        df['TIPO_AGUA'].isin(reemplazo_tipo_agua.keys()),
        df['TIPO_AGUA'].map(reemplazo_tipo_agua),
        'OTRAS'
    )

    return df

# ------------------------------------------------------------------------------------------------------------
def tx_reasignar_tipo_red(df):
    df['TIPO_RED'] = df['TIPO_RED'].replace({
        'COLECTOR'    : 'COLECTOR O INTERCEPTOR',
        'INTERCEPTOR' : 'COLECTOR O INTERCEPTOR',
    })

    return df
# ------------------------------------------------------------------------------------------------------------
# Binarizar zona por ser solo dos categorias: Sur/Norte y otros
def tx_reasignar_zonas(df):
    df['ZONA_SUR'] = df['ZONA'].map({'SUR ALCANTARILLADO' : True}).fillna(False).astype(bool)    
    return df

# ------------------------------------------------------------------------------------------------------------
# Binarizar municipio por ser solo dos categorias: MEDELLIN/Otros
def tx_reasignar_municipios(df):
    df['MUNICIPIO_MEDELLIN'] = df['MUNICIPIO'].map({'MEDELLÍN' : True}).fillna(False).astype(bool)    
    return df

# ------------------------------------------------------------------------------------------------------------
# Binarizar estado por ser solo dos categorias (OPERACION/Otros)
def tx_reasignar_estados(df):
    df['ESTADO_OPERACION'] = df['ESTADO'].map({'OPERACION': True}).fillna(False).astype(bool)
    return df

# ------------------------------------------------------------------------------------------------------------
def tx_reasignar_arranque(df):
    df['ARRANQUE'] = df['ARRANQUE'].map({'SI': True}).fillna(False).astype(bool)
    return df

# ------------------------------------------------------------------------------------------------------------
def tx_normalizar_numericas(df):
    columnas_numericas = df.select_dtypes(include=['int', 'float']).columns
    #scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = StandardScaler()
    df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])

    return df

# ------------------------------------------------------------------------------------------------------------
def tx_seleccionar_columnas(df):
    columnnas_utiles = [
        'DETERIORADO',
        'EDAD',
        'DIAMETRO',
        'LONGITUD',
        'AREA',
        'PROF_BATEA',
        'PROF_BATE1',
        'PENDIENTE',
        #'TIPO_RED', # eliminada
        'TIPO_AGUA',
        'MATERIAL',
        'FABRICANTE',
        'ARRANQUE',
        #'CAMARA_CAI', # eliminada
        'ZONA_SUR',
        'MUNICIPIO_MEDELLIN',
        'ESTADO_OPERACION',
    ]

    return df[columnnas_utiles]

# ------------------------------------------------------------------------------------------------------------
def tx_codificar(df):
    return pd.get_dummies(df)

# ------------------------------------------------------------------------------------------------------------
def tx_aplicar_transformaciones(df):
    transformaciones = {
        # Transformaciones de limpieza
        'Depurar datos nulos         ' : tx_depurar_datos_nulos,
        'Convertir tipo de datos     ' : tx_convertir_tipo_datos,
        'Validar rango de datos      ' : tx_validar_rango_datos,
        'Unificar unidades a metros  ' : tx_unificar_unidades,
        #'Reasignar tipo de red       ' : tx_reasignar_tipo_red, # columna eliminada
        'Reasignar arranque          ' : tx_reasignar_arranque,
        'Reasignar tipo de agua      ' : tx_reasignar_tipo_agua,
        'Reasignar materiales        ' : tx_reasignar_materiales,
        'Reasignar fabricantes       ' : tx_reasignar_fabricantes,
        'Reasignar zonas             ' : tx_reasignar_zonas,
        'Reasignar municipios        ' : tx_reasignar_municipios,
        'Reasignar estados           ' : tx_reasignar_estados,

        # Transformaciones de curado
        'Agregar edad de tramos      ' : tx_agregar_edades,
        'Aregar área de tramos       ' : tx_agregar_areas,
        'Recalcular pendientes       ' : tx_recalcular_pendientes,
        'Binarizar calificacion      ' : tx_binarizar_calificacion,
        'Seleccionar columnas útiles ' : tx_seleccionar_columnas,

        # Transformaciones de estandarización
        'Normalizar numéricas        ' : tx_normalizar_numericas,
    }

    df_transformado = df.copy()
    for nombre_transformacion, funcion_transformacion in transformaciones.items():
        print(f'Aplicando transformación: {nombre_transformacion}', end='... ')
        df_transformado = df_transformado.pipe(funcion_transformacion)
        print('Terminada.')

    return df_transformado

# --------------------------------------------------------------------------------------------------------------

# Obtener la fecha y hora actual
fecha_hora_actual = datetime.now()

# Imprimir la fecha y hora en un formato específico
formato = "%Y-%m-%d %H:%M:%S"  # Formato: Año-Mes-Día Hora:Minuto:Segundo
fecha_hora_formateada = fecha_hora_actual.strftime(formato)

print("Fecha y Hora de ejecución de módulo:", fecha_hora_formateada)
