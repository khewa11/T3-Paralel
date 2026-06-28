# Análisis de Resultados

## Contexto experimental

Se evaluaron dos implementaciones CUDA para el cálculo de la matriz de covarianza de un conjunto de imágenes del dataset DIV2K:

- **Experimento 1**: ejecución tradicional en un solo stream, cargando todo el dataset contiguamente en GPU.
- **Experimento 2**: ejecución con múltiples CUDA Streams, particionando el dataset en batches y usando memoria pinned para habilitar copias asíncronas.

La plataforma usada para las mediciones fue:

- **GPU**: NVIDIA GeForce GTX 1650 Ti
- **VRAM disponible**: 3715 MB
- **Sistema**: Fedora 44

Las imágenes del dataset fueron redimensionadas durante la carga para controlar el consumo de memoria. Se probaron tres resoluciones:

- `64x48`
- `96x72`
- `112x84`

En todos los casos se usaron 100 imágenes y `TILE_SIZE = 16`. Para el experimento 2 se usó `S = 4`.

---

## Recordatorio del costo computacional

El problema tiene tres etapas:

1. Cálculo del vector promedio.
2. Centrado de las imágenes.
3. Cálculo de la covarianza.

El punto crítico es la covarianza:

```text
C = (1/m) * V̄^T * V̄
```

donde:

- `m` es el número de imágenes
- `n = ancho * alto * canales`
- `C` tiene tamaño `n x n`

Como consecuencia, el costo de memoria y de cómputo crece aproximadamente con `n²`, por lo que pequeños aumentos de resolución producen incrementos muy grandes en el tamaño de la matriz de covarianza.

---

## Resultados obtenidos

### 1. Resolución `64x48`

#### Experimento 1

- `n = 9216`
- tamaño del dataset: ~`3 MB`
- tiempo de copia H→D: `0.74336 ms`
- tiempo de cómputo: `100.794 ms`
- tiempo de copia D→H: `51.4052 ms`
- tiempo total: `152.932 ms`

#### Experimento 2

- `n = 9216`
- tamaño total del dataset: ~`3 MB`
- tiempo total GPU: `1944.81 ms`
- copia D→H del resultado: `52.8463 ms`
- tiempo total global: `1997.66 ms`

#### Observación

En esta resolución, el experimento 1 fue mucho más rápido que el experimento 2. El uso de streams no logró ocultar suficiente latencia como para compensar el costo adicional de coordinación, sincronización y manejo de pinned memory.

---

### 2. Resolución `96x72`

#### Experimento 1

- `n = 20736`
- tamaño del dataset: ~`7 MB`
- tiempo de copia H→D: `1.47299 ms`
- tiempo de cómputo: `1007.95 ms`
- tiempo de copia D→H: `540.124 ms`
- tiempo total: `1549.55 ms`

#### Experimento 2

- `n = 20736`
- tamaño total del dataset: ~`7 MB`
- tiempo total GPU: `2978.75 ms`
- copia D→H del resultado: `531.57 ms`
- tiempo total global: `3510.32 ms`

#### Observación

Al aumentar la resolución, ambos experimentos se vuelven considerablemente más lentos, lo cual era esperado porque la matriz de covarianza escala cuadráticamente con `n`. Sin embargo, el experimento 2 sigue siendo más lento que el experimento 1. La diferencia relativa se reduce respecto a `64x48`, pero aún no hay ventaja en tiempo total.

---

### 3. Resolución `112x84`

#### Experimento 1

- `n = 28224`
- tamaño del dataset: ~`10 MB`
- resultado: **out of memory**

El experimento 1 no pudo reservar la memoria necesaria para la covarianza completa en la GPU.

#### Experimento 2

- `n = 28224`
- tamaño total del dataset: ~`10 MB`
- tiempo total GPU: `3026.42 ms`
- copia D→H del resultado: `498.38 ms`
- tiempo total global: `3524.8 ms`

#### Observación

En esta resolución el enfoque tradicional deja de ser viable en la GPU disponible, mientras que el experimento 2 todavía logra ejecutarse gracias al particionamiento del procesamiento en lotes. Esto muestra la principal virtud del enfoque multi-stream: **viabilidad de memoria**.

---

## Comparación resumida

| Resolución | `n` | Covarianza aprox. | Experimento 1 | Experimento 2 (S=4) | Comentario |
|---|---:|---:|---:|---:|---|
| `64x48` | `9,216` | `~327.6 MiB` | `152.9 ms` | `1997.7 ms` | Ambos corren; `exp1` domina |
| `96x72` | `20,736` | `~1.72 GiB` | `1549.6 ms` | `3510.3 ms` | Ambos corren; el costo crece fuerte |
| `112x84` | `28,224` | `~3.05 GiB` | **OOM** | `3524.8 ms` | `exp1` ya no cabe; `exp2` sí corre |

---

## Análisis del comportamiento

### 1. El aumento de resolución no mejora el tiempo absoluto

Subir la resolución incrementa el número de elementos por imagen y, por tanto, incrementa el tamaño de `n`. Como la covarianza es una matriz `n x n`, el costo de memoria y cómputo crece de manera muy agresiva.

Esto explica por qué:

- `64x48` es relativamente liviano.
- `96x72` multiplica drásticamente el costo.
- `112x84` ya satura la memoria de la GPU para el experimento 1.

### 2. El experimento 2 gana viabilidad de memoria, no necesariamente velocidad

El enfoque de streams permite procesar el dataset por partes. Eso reduce la presión sobre la VRAM y permite ejecutar configuraciones que el experimento 1 ya no soporta.

Sin embargo, en las mediciones obtenidas:

- el solapamiento entre copias y kernels fue bajo,
- la sincronización agregada fue costosa,
- la acumulación con atomics también introdujo overhead,
- y el tamaño del problema aún no fue lo bastante grande como para amortizar todo ese costo.

Por eso el experimento 2:

- **sí mejora la escalabilidad de memoria**,
- pero **no mejora el tiempo total** frente al experimento 1 en los casos donde ambos caben.

### 3. El cuello de botella no es solo el PCIe

Aunque el objetivo del experimento 2 era ocultar latencia de transferencia, los perfiles muestran que el tiempo no se explica únicamente por la copia H→D o D→H.

También influyen:

- `cudaHostAlloc`
- `cudaStreamSynchronize`
- `cudaMemcpyAsync`
- los lanzamientos de kernels adicionales
- y la sincronización implícita asociada al diseño de la implementación

En otras palabras: la complejidad de orquestación supera al beneficio del solapamiento en este caso concreto.

---

## Lectura de profiling con Nsight Systems

Al analizar los reportes generados con `nsys`, se observó lo siguiente:

### Experimento 1

- El timeline muestra una secuencia ordenada:
  - copia H→D
  - kernels de cómputo
  - copia D→H
- El overlap entre copy y kernels fue esencialmente nulo.
- La mayor parte del tiempo del GPU quedó concentrada en la covarianza.

### Experimento 2

- El timeline muestra múltiples streams.
- Sin embargo, el overlap observado fue muy pequeño.
- La suma de tiempo solapado entre memoria y kernels fue mínima comparada con el tiempo total.

### Advertencia de Nsight

Nsight emitió un aviso indicando que el trazado de eventos CUDA puede aumentar el overhead y crear dependencias falsas entre streams. Esto significa que, aunque el profiling es útil para inspección visual del timeline, hay que interpretar con cautela los tiempos absolutos de una corrida con profiling activado.

---

## Conclusiones principales

1. **La resolución influye de forma crítica** en el costo total porque la covarianza crece con `n²`.
2. **El experimento 1 es más rápido** mientras la matriz de covarianza todavía cabe cómodamente en la memoria de la GPU.
3. **El experimento 2 es más escalable en memoria**, porque divide el problema en batches y usa streams concurrentes.
4. **El experimento 2 no mostró ganancia de tiempo total** en las configuraciones probadas, porque el solapamiento efectivo fue demasiado pequeño y el overhead de coordinación fue alto.
5. **`112x84` marca un umbral importante**:
   - `experiment1` ya no puede reservar la memoria necesaria.
   - `experiment2` sí logra completar la ejecución.
6. **El principal valor del enfoque multi-stream aquí no es acelerar el caso pequeño**, sino permitir procesar un problema que ya no cabe en el enfoque monolítico.

---

## Conclusión final redactable para el informe

Al comparar las distintas resoluciones, se observa que el aumento del tamaño de las imágenes incrementa de manera muy marcada el costo del problema debido al crecimiento cuadrático de la matriz de covarianza. En la resolución `64x48`, ambos experimentos completan la ejecución, pero el enfoque tradicional resulta ampliamente más rápido. En `96x72`, el costo de cómputo y transferencia aumenta considerablemente, aunque el experimento 1 sigue siendo superior en tiempo total. Finalmente, en `112x84`, el experimento 1 deja de ser viable por falta de memoria en la GPU, mientras que el experimento 2 aún logra ejecutarse gracias al procesamiento por lotes y al uso de múltiples streams. Esto demuestra que la estrategia de streams no necesariamente mejora el tiempo absoluto en este caso, pero sí amplía la escalabilidad del problema al permitir trabajar con resoluciones que exceden la capacidad del enfoque tradicional.

---

## Datos útiles para gráficas del informe

### Tiempo total por resolución

- `64x48`
  - Exp1: `152.932 ms`
  - Exp2: `1997.66 ms`
- `96x72`
  - Exp1: `1549.55 ms`
  - Exp2: `3510.32 ms`
- `112x84`
  - Exp1: `OOM`
  - Exp2: `3524.8 ms`

### Speedup de Exp2 respecto a Exp1

Solo se puede calcular donde ambos corren:

- `64x48`: `152.932 / 1997.66 = 0.0766`
- `96x72`: `1549.55 / 3510.32 = 0.4414`

Interpretación:

- valores menores que 1 indican que Exp2 es más lento que Exp1
- el valor se acerca a 1 al aumentar la resolución, pero aún no la supera

---

## Cierre

Este conjunto de resultados permite argumentar que la estrategia con CUDA Streams mejora la capacidad de escalar en memoria, pero no necesariamente el tiempo total en esta configuración específica. La resolución intermedia `96x72` sirve como buen punto de comparación entre costo y viabilidad, mientras que `112x84` evidencia el límite práctico del enfoque tradicional. En un informe final, conviene destacar que el beneficio de la orquestación con streams aparece más claramente cuando el problema es suficientemente grande para que la latencia de transferencia sea dominante y el overhead de coordinación quede amortizado.

