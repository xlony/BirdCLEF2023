# Diseño Técnico: Clasificación de Sonidos de Aves

## Fundamentos del Procesamiento de Audio

### Por qué Espectrogramas Mel
El sistema auditivo humano y de las aves percibe el sonido de manera no lineal, con mayor sensibilidad a cambios en frecuencias bajas. Los espectrogramas Mel reflejan esta característica natural:

- Mejor representación de características relevantes del canto de aves
- Reducción efectiva de dimensionalidad manteniendo información crucial
- Similitud con el procesamiento auditivo biológico

## Estructura de Datos

### Archivos CSV del Dataset

#### 1. train_metadata.csv
Contiene la información principal de etiquetado:
- **primary_label**: Especie principal identificada en la grabación
- **secondary_labels**: Especies adicionales presentes en el audio
- Permite implementación de clasificación multi-etiqueta
- Base para el mapeo entre archivos de audio y especies

#### 2. eBird_Taxonomy_v2021.csv
Proporciona la taxonomía estandarizada de especies:
- Jerarquía taxonómica completa de las especies
- Garantiza consistencia en la identificación de especies
- Referencia para la validación de etiquetas

#### 3. sample_submission.csv
Estructura del formato de predicciones:
- Define el formato esperado para las predicciones del modelo
- Guía para la generación de resultados en producción

## Arquitectura del Modelo

### Redes Neuronales Convolucionales (CNN)
La elección de CNNs como base del modelo se fundamenta en:

1. **Patrones Locales**: Los cantos de aves contienen patrones espectrales localizados
2. **Invariancia a la Traslación**: Las características del canto son relevantes independientemente de su posición temporal
3. **Jerarquía de Características**: Las CNNs pueden aprender desde características básicas (tonos) hasta patrones complejos (secuencias de canto)

### Mecanismo de Atención
La incorporación de atención multi-cabeza mejora el modelo por:

1. **Dependencias Temporales**: Captura relaciones entre diferentes partes del canto
2. **Foco Selectivo**: Permite al modelo concentrarse en segmentos relevantes
3. **Procesamiento Paralelo**: Las múltiples cabezas capturan diferentes aspectos del audio

```python
# Ejemplo de la implementación de atención
self.attention = nn.MultiheadAttention(256, num_heads=8)
```

## Decisiones de Diseño

### Arquitectura CNN
- **Capas Convolucionales**: Incremento gradual de canales (64 → 128 → 256)
  - Permite extraer características cada vez más complejas
- **Normalización por Lotes**: Mejora la estabilidad del entrenamiento
- **Pooling Máximo**: Reduce dimensionalidad manteniendo características importantes

### Clasificación Final
- **Pooling Global Promedio**: Reduce la variabilidad en la longitud de las secuencias
- **Capas Fully Connected**: Mapeo final a probabilidades de especies
- **Dropout**: Previene el sobreajuste

## Ventajas del Diseño

1. **Eficiencia Computacional**
   - Arquitectura balanceada entre profundidad y anchura
   - Uso efectivo de pooling para reducción de dimensionalidad

2. **Robustez**
   - Múltiples mecanismos contra sobreajuste
   - Capacidad de manejar variabilidad en los datos

3. **Flexibilidad**
   - Adaptable a diferentes números de especies
   - Arquitectura modular para futuros ajustes

## Referencias y Base Científica

1. Transformers en Audio:
   - "Attention Is All You Need" (Vaswani et al., 2017)
   - Adaptación para procesamiento de audio

2. Procesamiento de Señales de Audio:
   - Análisis de Mel-espectrogramas
   - Técnicas de aumento de datos para audio

3. Bioacústica:
   - Estudios sobre vocalizaciones de aves
   - Patrones espectrales en cantos de aves

## Conclusión

La arquitectura diseñada combina principios probados de procesamiento de señales con técnicas modernas de deep learning. La integración de CNNs con mecanismos de atención permite un procesamiento robusto y efectivo de sonidos de aves, considerando tanto las características locales como las relaciones temporales en los cantos.

La documentación técnica continuará evolucionando con el proyecto, incorporando nuevos hallazgos y mejoras basadas en resultados experimentales.