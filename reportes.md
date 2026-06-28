# Resultados de Profiling con NVIDIA Nsight Systems

## Entorno de Ejecución

- GPU: NVIDIA GeForce GTX 1650 Ti
- VRAM: 3715 MB
- Dataset: DIV2K_valid_LR_bicubic_X4
- Número de imágenes: 100
- Resolución utilizada: 64 × 48 × 3
- Tamaño vectorizado por imagen:

  n = 64 × 48 × 3 = 9216

- Número de Streams: 4
- Tile Size: 16

---

# Resumen CUDA (API / Kernels / MemOps)

Reporte obtenido mediante:

```bash
nsys stats --report cuda_api_gpu_sum streams_report.nsys-rep
```

| Categoría   | Operación                  | Tiempo Total (ns) | % Tiempo |
| ----------- | -------------------------- | ----------------: | -------: |
| CUDA_API    | cudaHostAlloc              |       121,930,477 |    26.5% |
| CUDA_API    | cudaStreamSynchronize      |       113,116,190 |    24.6% |
| CUDA_KERNEL | accumulateCovKernel<16>    |       112,813,689 |    24.5% |
| CUDA_API    | cudaMemcpy                 |        52,885,074 |    11.5% |
| MEMORY_OPER | CUDA memcpy Device-to-Host |        52,741,260 |    11.5% |
| MEMORY_OPER | CUDA memcpy Host-to-Device |         1,175,072 |     0.3% |
| MEMORY_OPER | CUDA memset                |         1,811,231 |     0.4% |

## Observaciones

- La transferencia Host→Device representa una fracción mínima del tiempo total.
- La transferencia Device→Host es significativamente mayor debido al tamaño de la matriz de covarianza resultante.
- El costo de sincronización (`cudaStreamSynchronize`) es comparable al costo computacional principal.

---

# Resumen de Kernels GPU

Reporte obtenido mediante:

```bash
nsys stats --report cuda_gpu_kern_sum streams_report.nsys-rep
```

| Kernel                  | Instancias | Tiempo Total (ns) | % Tiempo GPU |
| ----------------------- | ---------: | ----------------: | -----------: |
| accumulateCovKernel<16> |          4 |       112,813,689 |        99.9% |
| centerBatchKernel       |          4 |            69,280 |         0.1% |
| accumulateMeanKernel    |          4 |            25,504 |        <0.1% |
| finalizeMeanKernel      |          1 |             1,728 |        <0.1% |

---

# Análisis de Kernels

## Kernel de Promedio

Calcula:

\[
\mu*j = \frac{1}{m}\sum*{k=0}^{m-1}v_j^{(k)}
\]

Su costo computacional es despreciable frente al resto de la aplicación.

---

## Kernel de Centrado

Calcula:

\[
\bar{v}\_j^{(k)} = v_j^{(k)} - \mu_j
\]

También presenta un costo extremadamente bajo.

---

## Kernel de Covarianza

Calcula:

\[
C=\frac{1}{m}\bar{V}^{T}\bar{V}
\]

Representa aproximadamente el **99.9% del tiempo de ejecución GPU**, constituyéndose como el cuello de botella principal del sistema.

---

# Análisis del Uso de Streams

El objetivo de CUDA Streams es ocultar la latencia de las transferencias PCIe mediante la ejecución concurrente de:

- Transferencias Host→Device.
- Ejecución de kernels.
- Transferencias Device→Host.

Sin embargo, el perfilado muestra que:

| Operación   |    Tiempo |
| ----------- | --------: |
| Host→Device |   1.17 ms |
| Covarianza  | 112.81 ms |

Por lo tanto:

\[
\frac{1.17}{112.81}\approx 1\%
\]

La transferencia de datos representa aproximadamente el 1% del tiempo consumido por el kernel principal.

En consecuencia, existe muy poca latencia de comunicación que pueda ocultarse mediante overlap.

---

# Conclusiones

1. El kernel de covarianza domina completamente la ejecución, representando aproximadamente el 99.9% del tiempo GPU.

2. Los kernels de promedio y centrado tienen un impacto prácticamente nulo sobre el tiempo total.

3. Las transferencias Host→Device son demasiado pequeñas para convertirse en un cuello de botella.

4. El uso de múltiples CUDA Streams no produce mejoras significativas debido a que el tiempo total está dominado por el cálculo de la covarianza.

5. El costo de sincronización (`cudaStreamSynchronize`) es comparable al tiempo de cómputo observado, reduciendo aún más los beneficios potenciales del paralelismo mediante streams.

6. El rendimiento de la aplicación está limitado principalmente por el cálculo de:

\[
C=\frac{1}{m}\bar{V}^{T}\bar{V}
\]

cuyo costo computacional crece proporcionalmente a:

\[
O(n^2m)
\]

siendo esta la operación dominante del algoritmo.
