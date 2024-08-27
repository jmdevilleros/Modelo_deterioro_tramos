*************************************************************************
# Modelo  para predicción de deterioro en tramos de tubería

## Version alterna 01 (alt01)

- Simplificación de variables categóricas con solamente dos valores: Convertirlas a binarias.
- Reevaluación de modelos
- Se agrega entrenamiento de modelo (.fit) antes de cross validation
- Se cambia distribución de datos para GridSearch (train) y Cross Validate (test)


## Resultados: 
- Mejor modelo = Histogram Gradient Boosting Classifier.
- Hiperparámetros:
    - 'categorical_features': 'warn'
    - 'class_weight': {False: 0.6978099115541204, True: 1}
    - 'early_stopping': False
    - 'interaction_cst': None
    - 'l2_regularization': 1
    - 'learning_rate': 0.015, 
    - 'loss': 'log_loss', 
    - 'max_bins': 255, 
    - 'max_depth': None, 
    - 'max_features': 1.0, 
    - 'max_iter': 100, 
    - 'max_leaf_nodes': 63, 
    - 'min_samples_leaf': 5, 
    - 'monotonic_cst': None, 
    - 'n_iter_no_change': 10, 
    - 'random_state': None, 
    - 'scoring': 'loss', 
    - 'tol': 1e-07, 
    - 'validation_fraction': 0.1, 
    - 'verbose': 0, 
    - 'warm_start': False

    ![alt text](Gráficas\resultados_barras.png)

    ![alt text](Gráficas\resultados_tiempos.png)

    ![alt text](Gráficas\resultados_tabla.png)

*************************************************************************