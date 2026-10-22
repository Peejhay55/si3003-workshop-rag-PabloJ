# Respuestas del workshop RAG

Estas respuestas resuelven los ejercicios sin incluir capturas. Las tablas que
dependen de NVIDIA o de una medición local se marcan como **esperadas**: la
API no está disponible en este entorno, así que no se presentan como
mediciones ejecutadas.

## Bloque 0 · Teoría

### T1 · Cuántos nodos viven en cada capa

Con `M = 16`, `P(l >= k) = 16^-k`. Para `N = 1.000.000`:

| Capa | Cálculo | Esperados |
|---|---:|---:|
| >= 1 | `1.000.000 / 16` | 62.500 |
| >= 2 | `1.000.000 / 16^2` | 3.906,25 |
| >= 3 | `1.000.000 / 16^3` | 244,14 |

La simulación con semilla 0 debe quedar cerca de esos valores; la diferencia
es muestreo aleatorio. Tener pocos hubs arriba reduce el número de candidatos
que se visitan en las capas superiores y permite saltar grandes distancias
antes de refinar la búsqueda en la capa 0.

### T2 · `ef`, `k` y el tamaño de la cola

1. Con `k = 10` y `ef = 5`, la cola no puede mantener suficientes candidatos
para devolver diez vecinos fiables; en implementaciones que exigen
`ef >= k`, la configuración es inválida o se corrige/degrada.
2. Conviene empezar con `ef` aproximadamente igual a `k`, medir recall y
latencia, y subirlo gradualmente si el recall es insuficiente.
3. `ef` se puede cambiar en producción sin reconstruir el índice: afecta la
búsqueda. `ef_construction` se usa al insertar y cambiarlo exige reconstruir
el grafo para que el nuevo parámetro tenga efecto retrospectivo.

### T3 · Inserción frente a búsqueda

| Pregunta | Inserción | Búsqueda |
|---|---|---|
| ¿Se sortea la capa? | Sí, para asignar el nivel máximo del nuevo vector. | No se crean niveles nuevos. |
| `ef` en capas altas | `ef_construction`, para conectar bien el nuevo nodo. | `efSearch`, para explorar candidatos. |
| Parámetro de la fase principal | `ef_construction` | `efSearch` |
| ¿Modifica el índice? | Sí, añade el vector y enlaces. | No, solo consulta. |

En el notebook 03, `vectorstore.add_documents(all_docs)` dispara inserción;
`retriever.invoke(pregunta)` dispara búsqueda.

### T4 · El invariante de `FAISSVectorStore`

Antes de borrar: `0 -> A`, `1 -> B`, `2 -> C`, `3 -> D`. Si se borra `B`
solo de `self.documents`, el índice devuelve los identificadores `[1, 2, 3]`,
pero la lista ahora interpreta `1 -> C`, `2 -> D` y `3` queda fuera de rango.
Por tanto, `1` es un error silencioso (devuelve `C` en lugar de `B`) y `3`
puede lanzar `IndexError`. El índice FAISS no sabe que la lista cambió.
Qdrant evita depender de una posición paralela mantenida manualmente: cada
punto tiene un identificador y su payload/documento asociado.

### T5 · El costo de la fuerza bruta

`768.000.000` multiplicaciones por consulta por `100` consultas/segundo =
`76.800.000.000` multiplicaciones/s = **7,68 × 10^10**. La fuerza bruta
compara cada consulta contra todos los vectores y crece linealmente con el
tamaño del corpus, por lo que deja de ser práctica a esa escala. ANN sacrifica
la garantía de exactitud/recall perfecto y gana mucha menor latencia y coste.

## Bloque A · App RAG empresarial

### A1 · Ejecución y preguntas

La app carga [company_documents.json](data/company_documents.json), crea
chunks y muestra fuentes con score. Tabla reproducible para las cuatro
preguntas de ejemplo y una pregunta fuera del corpus:

| Pregunta | Respuesta esperada | Fuente |
|---|---|---|
| ¿Cuánto cuesta CloudSync Pro Enterprise? | `$49/mes por usuario` con compromiso anual. | CloudSync Pro — Precios |
| ¿Cuántos días de trabajo remoto se permiten? | Hasta 3 días por semana, con aprobación del jefe. | Política de trabajo remoto |
| ¿Qué ocurre con un reembolso mayor a $100? | Requiere aprobación; después tarda 3–5 días hábiles. | Política de reembolsos |
| ¿Qué se hace durante la primera semana? | Se completan módulos de seguridad, cumplimiento y cultura. | Onboarding |
| ¿Cuál es la política de vacaciones? | `No encuentro esa información en los documentos.` | Ninguna |

La última respuesta es la conducta correcta: el prompt actualizado ordena no
inventar cuando no existe evidencia.

### A2 · Corpus propio

El corpus contiene ahora 8 documentos, cada uno con `id`, `title`, `content` y
las claves de metadata `department` y `type`. Se usan cinco preguntas
respondibles (precios, remoto, reembolsos, descuentos y soporte) y dos no
respondibles (vacaciones y salario). Los documentos son cortos, por lo que
`chunk_size=400` y `chunk_overlap=40` evitan dividirlos innecesariamente,
manteniendo margen para documentos algo más largos.

### A3 · Ingeniería del prompt

Antes: el prompt pedía usar solo el contexto, reconocer la falta de
información y responder concisamente. Después exige citar el título entre
corchetes, usar exactamente `No encuentro esa información en los documentos.`
cuando falte evidencia y limitar la respuesta a tres oraciones.

| Tipo de pregunta | Antes | Después |
|---|---|---|
| Fácil | Responde con el dato correcto, normalmente sin título. | Dato correcto y título citado. |
| Ambigua | Puede responder parcialmente según los chunks recuperados. | Debe ceñirse al contexto y ser más corta. |
| Fuera de contexto | Puede decir que no sabe, pero no con formato fijo. | Respuesta fija, sin inventar. |

La instrucción más efectiva es restringir la respuesta a la evidencia y dar una
frase fija para ausencia de contexto. El límite de tres oraciones es una
restricción de formato que un modelo puede incumplir ocasionalmente.

### A4 · Scores y umbral

El notebook devuelve `vectorstore` y `chat()` muestra tres scores. Sin
ejecutar embeddings no es responsable inventar números; el procedimiento es
registrar tres preguntas buenas y dos fuera de contexto y elegir un umbral por
debajo del menor score bueno y por encima de los scores irrelevantes. Un punto
de partida razonable para este corpus es `0,30`, pero debe sustituirse por el
valor observado y validarse con las siete preguntas de A2.

## Bloque B · RAG sobre papers

### B1 · Quiz y ground truth

| # | Resumen | Respuesta correcta | Paper correcto |
|---|---|---|---|
| Q1 | BLEU del Transformer big en WMT14 En→De | 28,4 BLEU. | Attention, sí |
| Q2 | Memoria paramétrica/no paramétrica | BART-large + índice denso de Wikipedia recuperado por DPR; se entrenan conjuntamente. | RAG, sí |
| Q3 | Escala `sqrt(dk)` | Divide por `sqrt(dk)` para evitar logits grandes y saturación de softmax/gradientes pequeños. | Attention, sí |
| Q4 | Variantes de RAG | RAG-Sequence usa un documento para toda la secuencia; RAG-Token puede cambiar de documento por token. | RAG, sí |
| Q5 | Cabezas y dimensiones | Base: 8 cabezas; `dk = dv = 512/8 = 64`. | Attention, sí |
| Bonus | Generador y arquitectura | BART-large, un seq2seq basado en Transformer. | RAG y Attention |

La bonus debe recuperar fragmentos del paper RAG y, para justificar
Transformer, idealmente también del paper Attention; si MMR devuelve solo RAG,
la respuesta puede seguir ser correcta por la cita a BART, pero la evidencia
de los dos papers sería incompleta.

### B2 · `chunk_size`, `k` y `score_threshold`

Resultado esperado, no medido por falta de NVIDIA: chunks más pequeños
aumentan el número de vectores y pueden perder contexto; chunks de 1500
reducen el número de vectores, pero mezclan temas. `k=3` suele ser preciso,
`k=5` es un compromiso y `k=8` aporta cobertura con más ruido. Un umbral
bajo (`0,15`) recupera más, `0,25` es intermedio y `0,40` puede dejar la
respuesta sin evidencia.

| Experimento | Valores | Predicción |
|---|---|---|
| `chunk_size` | 400 / 900 / 1500 | 900 suele equilibrar contexto y precisión. |
| `k` | 3 / 5 / 8 | 5 suele equilibrar cobertura y ruido. |
| threshold | 0,15 / 0,25 / 0,40 | 0,25 suele ser el punto inicial; 0,40 maximiza abstenciones. |

La conclusión debe usar los nueve conteos reales al ejecutar; no se sustituyen
por cifras inventadas.

### B3 · Preguntas trampa

Para una pregunta fuera del corpus, con el retriever actual sin un umbral
estricto el modelo puede recibir chunks tangencialmente parecidos y mezclar
verdad con invención. La corrección es usar un `score_threshold` validado,
imprimir los documentos recuperados y mantener el prompt de abstención. La
clasificación esperada es: pregunta respondible → admite la evidencia;
pregunta no respondible con chunks irrelevantes → riesgo de mezcla/invención;
pregunta no respondible sin chunks → admite que no sabe.

### B4 · Preguntas en español

| Pregunta | Chunks esperados | Calidad esperada |
|---|---|---|
| ¿Qué BLEU obtuvo el Transformer grande? | Principalmente Attention; puede no coincidir exactamente con EN. | Buena si el embedding es multilingüe; menor confianza posible. |
| ¿Para qué sirve `1/sqrt(dk)`? | Attention, sección 3.2.1. | La respuesta debe conservar la explicación, quizá con más ruido. |
| ¿Cuáles son las dos variantes de RAG? | RAG, sección 2.1. | Buena si el modelo alinea EN↔ES; puede recuperar menos preciso. |

La hipótesis es que el embedding puede representar consultas
semánticamente cercanas entre idiomas, pero los papers son monolingües y la
alineación puede ser peor que la coincidencia en inglés. Se comprueba
comparando IDs de chunks, scores y hit@k con la misma consulta traducida.

### B5 · Quiz propio

Tres preguntas y ground truth propuestos:

1. **BERT:** “¿Qué objetivo permite a BERT usar contexto izquierdo y derecho?”
   Respuesta: masked language modeling; fuente: sección 3.3 del paper BERT.
2. **Attention:** “¿Cómo se concatenan y proyectan las salidas de las
   cabezas?” Respuesta: se concatenan y pasan por `W^O`; fuente: sección
   3.2.2.
3. **Cruce RAG + Attention:** “¿Por qué BART-large puede considerarse
   Transformer en el contexto de RAG?” Respuesta: RAG usa BART-large como
   seq2seq preentrenado y BART está basado en la arquitectura Transformer;
   fuentes: sección 2.3 de RAG y el paper Attention.

## Reflexión

1. Cuando el contexto no contiene suficiente información, el resultado
   depende tanto del umbral como del prompt. Con un umbral alto y contexto
   vacío, el comportamiento esperado es abstenerse; con chunks irrelevantes,
   un modelo generativo puede alucinar, por eso la abstención debe probarse.
2. En la bonus deben aparecer chunks de RAG, porque allí se menciona BART y
   FAISS/HNSW, y chunks de Attention si el retriever captura la referencia a
   Transformer. La lista `sources` del notebook permite comprobarlo.
3. `chunk_size` cambia la granularidad, `k` cambia cobertura/ruido y el
   umbral controla abstención. En producción mediría hit@k y calidad de
   respuesta, en vez de asumir que un parámetro es universal.
4. Separar embeddings y generación permite cambiar el LLM sin reindexar y
   optimizar cada coste por separado. El embedding sirve para recuperar y el
   generador para redactar; tienen latencias, precios y criterios de calidad
   diferentes.
5. Para producción añadiría Qdrant persistente/gestionado, autenticación y
   autorización por metadata, evaluación offline y regresiones, trazas de
   retrieval/generación, límites de coste, caché, backups, versionado de
   prompts y monitorización de latencia, errores y scores.

## Retos opcionales

### C1 · HNSW contra fuerza bruta

`IndexFlatL2` es la verdad exacta. HNSW debe aumentar recall y normalmente
latencia al subir `efSearch`; aumentar `M` mejora conectividad y recall, pero
eleva memoria y tiempo de construcción. Con datos aleatorios de 64
dimensiones, el problema es difícil porque no hay estructura semántica:
es esperable necesitar `efSearch` alto para superar 0,95. El valor exacto de
`efSearch` y el `n` de cruce deben salir de la ejecución, no se pueden afirmar
sin medir.

### C2 · Tercer paper HNSW

La modificación requerida es añadir:

```python
"hnsw_paper.pdf": "https://arxiv.org/pdf/1603.09320"
```

Preguntas con ground truth:

1. La capa máxima se sortea con `l = floor(-ln(U) * mL)` y
   `mL ≈ 1/ln(M)`.
2. La probabilidad de alcanzar al menos la capa `k` es aproximadamente
   `M**(-k)`.
3. `M` controla la conectividad máxima por capa: subirlo suele mejorar recall
   a cambio de memoria y construcción.

Pregunta cruzada: “¿Cómo se relaciona el HNSW usado por FAISS en RAG con la
jerarquía del paper HNSW?” Respuesta: ambos explotan navegación aproximada
small-world; FAISS encapsula la estructura y expone `efSearch`, mientras el
paper describe niveles, `M` y la construcción. La notación puede variar
(`ef`/`efSearch`, `mL`), pero el trade-off es el mismo.

### C3 · Gradio con filtro

El notebook 03 ya incluye una interfaz Gradio que filtra por
`metadata.source`, muestra título, archivo y score, y ofrece “Todos” más los
papers disponibles. Sin filtro una pregunta puede usar ambos papers; con
`attention_paper.pdf` solo usa evidencia de Attention y con
`rag_paper.pdf` solo evidencia de RAG. Esto mejora trazabilidad, aunque puede
empeorar una respuesta que necesita cruzar documentos.

### C4 · Evaluación automática

La métrica `hit@k` es:

```python
ok = any(
    row["palabra_clave"].lower() in doc.page_content.lower()
    and doc.metadata["source"] == row["fuente_esperada"]
    for doc in retriever.invoke(row["pregunta"])
)
```

El resultado es `hits / len(rows)`, calculado para `k=3` y `k=5` con
`chunk_size` 400, 900 y 1500. Limitaciones: una palabra clave puede aparecer
sin que el chunk responda realmente la pregunta, y una respuesta correcta
puede no contener la palabra exacta por paráfrasis o extracción imperfecta
del PDF. Los porcentajes deben completarse al ejecutar las 10 filas.

### C5 · MMR frente a similitud pura

MMR con `k=5`, `fetch_k=20`, `lambda_mult=0,5` primero considera candidatos
similares y luego penaliza redundancia. Ayuda en preguntas amplias o que
requieren cobertura de varias secciones; similitud pura puede devolver cinco
chunks casi duplicados. MMR puede empeorar una pregunta muy puntual porque
reemplaza un segundo fragmento muy relevante por otro diverso pero menos
útil. El notebook 03 ya usa MMR en su retriever principal.

### C6 · Persistencia

El notebook 03 usa `QdrantClient(path="./qdrant_data")`. La primera ejecución
calcula embeddings e indexa; la reapertura solo necesita crear el cliente y
construir `QdrantVectorStore` sobre la colección existente, evitando pagar de
nuevo los embeddings. La tabla de tiempos debe medirse en el entorno de
ejecución. Persistencia local todavía no resuelve concurrencia, autenticación,
backups, replicación, control de acceso ni supervivencia del disco efímero de
Colab; para eso hace falta almacenamiento duradero o un servicio Qdrant.
