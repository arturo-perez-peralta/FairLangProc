# Documentación del Paquete FairLangProc

Este documento combina las guías de demostración para los módulos de Datasets, Métricas y Procesadores.

---

# Demo: Datasets de Equidad (Fairness)

Este notebook muestra los datasets disponibles en el módulo. Echaremos un vistazo a la tarea que propone cada uno, así como a su formato.

Los conjuntos de datos se descargaron del repositorio de GitHub [https://github.com/i-gallegos/Fair-LLM-Benchmark](https://github.com/i-gallegos/Fair-LLM-Benchmark).

**Referencia:** Gallegos, I. O., Rossi, R. A., Barrow, J., Tanjim, M. M., Kim, S., Dernoncourt, F., ... & Ahmed, N. K. (2024). Bias and fairness in large language models: A survey. Computational Linguistics, 1-79.
**Preprint:** [https://arxiv.org/abs/2309.00770](https://arxiv.org/abs/2309.00770).

## BBQ (Bias Benchmark for QA)

El conjunto de datos BBQ está diseñado para medir el sesgo en modelos de lenguaje en tareas de Pregunta-Respuesta (QA).

## StereoSet

El conjunto de datos StereoSet mide el sesgo estereotípico en modelos de lenguaje.

## CrowS-Pairs

El conjunto de datos CrowS-Pairs es un recurso para medir el grado en que los Modelos de Lenguaje (LMs) enmascarados se basan en estereotipos sociales.

## WinoGender

El conjunto de datos WinoGender es un recurso para medir el sesgo de género en la resolución de correferencias.

---

# Demo: Métricas de Equidad (Fairness)

En este notebook mostramos las métricas disponibles en el submódulo `FairnessMetrics`. Básicamente, existen tres tipos diferentes de métricas para evaluar el sesgo en LLMs:

1.  **Basadas en Embeddings:** Se basan en pruebas de asociación sobre los embeddings tanto de palabras sensibles como de palabras con ciertos atributos (profesiones, ocupaciones,...).
2.  **Basadas en Probabilidad:** Se calculan utilizando un modelo de lenguaje enmascarado (MLM) para computar las probabilidades de los tokens enmascarados.
3.  **Basadas en Texto Generado:** Cuentan el léxico utilizado en las generaciones de ciertos modelos.

## Métricas basadas en Embeddings

### WEAT (Word Embedding Association Test)

Mide la asociación entre dos conjuntos de palabras objetivo (por ejemplo, 'Flores' y 'Herramientas') y dos conjuntos de palabras de atributo (por ejemplo, 'Agradable' y 'Desagradable').

### SEAT (Sentence Embedding Association Test)

Similar a WEAT, pero opera a nivel de embeddings de oraciones.

## Métricas basadas en Probabilidad

### CrowS-Pairs

Utiliza pares de oraciones que difieren mínimamente para evaluar la presencia de estereotipos sociales.

### StereoSet

Evalúa el sesgo estereotípico en tres categorías: género, profesión y raza, utilizando tareas de asociación intra-oración.

## Métricas basadas en Texto Generado

### HONEST

Mide el lenguaje hiriente en las completaciones de texto generadas por un modelo.

---

# Demo: Procesadores de Equidad (Fairness)

En este notebook mostraremos los diferentes procesadores de equidad que hemos implementado, mostrando un caso de uso simple en el que eliminamos el sesgo (debias) del modelo *BERT*.

Los procesadores de equidad se pueden clasificar según la parte del pipeline de machine learning en la que se introducen:

1.  **Pre-procesadores:** si se introducen antes de que el modelo haya sido entrenado.
2.  **In-procesadores:** si se introducen durante el proceso de entrenamiento del modelo.
3.  **Post-procesadores:** si se introducen después del paso de entrenamiento.
4.  **Intra-procesadores:** adicionalmente, hablamos de *intra-procesadores* al referirnos a métodos de equidad que no modifican los parámetros de un modelo. Esta noción se solapa con la de post-procesadores y puede considerarse equivalente.

Para mostrar la implementación de estos métodos, los ejecutaremos en el conjunto de datos IMDB sin más consideraciones, ya que solo pretende servir como prueba de concepto.

## Pre-procesamiento: CDA (Counterfactual Data Augmentation)

CDA genera ejemplos de datos contrafácticos para aumentar el conjunto de entrenamiento.

## In-procesamiento: Debiasing de Embeddings

Este método aplica técnicas de eliminación de sesgo directamente a la capa de embeddings durante el entrenamiento.

## Intra-procesamiento: EAT (Entropy-based Adversarial Training)

EAT introduce un hook (gancho) en el modelo para aplicar un entrenamiento adversarial basado en entropía sin modificar la arquitectura central.

## Post-procesamiento: Diff

Este método ajusta las representaciones de salida del modelo para mitigar el sesgo después del entrenamiento.