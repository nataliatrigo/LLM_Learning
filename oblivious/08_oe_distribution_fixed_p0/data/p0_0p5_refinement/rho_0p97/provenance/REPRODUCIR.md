# Refinamiento de rho = 0.97, p0 = 0.5

Se conserva el mismo modelo, parámetros y tolerancias: `tol=1e-7`,
`dp_tol=1e-10`, estacionariedad `1e-11`, cuadratura de orden 256 y auditoría
con orden 512. Se refinaron las grillas uniformes 51 → 101 → 151 → 201.
Los certificados corresponden a las grillas discretas; la comparación entre
grillas es evidencia numérica de estabilidad, no una prueba del límite continuo.

## Ruta de búsqueda y certificación

El runner genérico comenzó con la política interpolada de la grilla anterior.
En 101, el intento local se estancó y comenzó la continuación; se detuvo
únicamente después de encontrar y certificar una propuesta alternativa. En
151 se hizo lo mismo con el intento local interpolado. Esos procesos propios
se interrumpieron antes de adjuntar resultados para evitar escrituras simultáneas.
Ningún candidato sin certificado fue publicado.

Para 101 y 151 se exploraron políticas usando la ley conjunta estacionaria
**positiva**. Los nodos mixtos se resolvieron con `brentq`. La elección de una
cara activa solo genera una propuesta: `e.evaluate` comprueba nuevamente todas
las desigualdades de acción, indiferencia mixta, optimalidad, cierre marginal
y estacionariedad conjunta positiva.

En 201, la exploración usa una ley estacionaria **signed**, obtenida con un
sistema disperso de la cadena de saltos y una sola columna del lado derecho.
Esa ley solamente guía la búsqueda. El resultado pasó al primer intento el
certificado completo original de `e.evaluate`, que usa estacionariedad positiva
GTH. No fue necesario el pulido de Newton previsto en el helper. La búsqueda
no alteró la demanda, las transiciones ni los umbrales de certificación.

## Comandos y orden utilizados

Desde la raíz del repositorio, con `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` y `MPLCONFIGDIR=/tmp/rho097_mpl`:

```bash
.venv/bin/python -B oblivious/08_oe_distribution_fixed_p0/code/refine_case.py --rho .97
```

Los helpers guardan sus propuestas certificadas en `/tmp/`, sin sobrescribir
los checkpoints históricos. Estas son las recetas de la ruta alternativa;
los scripts, logs y propuestas exactos están copiados en esta carpeta:

```bash
.venv/bin/python -B oblivious/08_oe_distribution_fixed_p0/data/p0_0p5_refinement/rho_0p97/provenance/positive_face097_one.py
.venv/bin/python -B oblivious/08_oe_distribution_fixed_p0/data/p0_0p5_refinement/rho_0p97/provenance/attach_verified_proposal.py

.venv/bin/python -B oblivious/08_oe_distribution_fixed_p0/code/refine_case.py --rho .97 --resume
.venv/bin/python -B oblivious/08_oe_distribution_fixed_p0/data/p0_0p5_refinement/rho_0p97/provenance/positive_face_refine097_n151.py 151 101
.venv/bin/python -B oblivious/08_oe_distribution_fixed_p0/data/p0_0p5_refinement/rho_0p97/provenance/attach_followup_proposal.py 151

.venv/bin/python -B oblivious/08_oe_distribution_fixed_p0/data/p0_0p5_refinement/rho_0p97/provenance/sparse_face_refine097_n201.py
.venv/bin/python -B oblivious/08_oe_distribution_fixed_p0/data/p0_0p5_refinement/rho_0p97/provenance/attach_followup_proposal_n201.py 201

.venv/bin/python -B oblivious/08_oe_distribution_fixed_p0/code/refine_case.py --rho .97 --resume
```

Antes de cada adjunción se copiaron los archivos `/tmp/rho097_positive_face_nNNN.{json,npz}`
a esta carpeta; 101 usa el nombre `rho097_positive_face_n101`. Los helpers de
adjunción protegen los resultados existentes y requieren el manifest en el
estado anterior correspondiente: estos comandos documentan la ejecución, no
son una orden de sobrescribir la corrida terminada. Los scripts de búsqueda
contienen las rutas originales de esta corrida y deben ajustarse si se reproduce
en otra ubicación o con otro nombre de salida.

La adjunción de 101 repitió `e.evaluate`. En 151 y 201 se reutilizó el certificado
positivo recién obtenido, después de comprobar hashes de fuentes y la misma
configuración; además se comprobaron de nuevo el residuo del kernel y las
condiciones de acción con el kernel recién construido. Cada adjunción hizo
una nueva auditoría de demanda, resolvió y certificó el monopolio, evaluó los
valores descontados en el juego conjunto real y verificó las identidades de
flujos, ventas y beneficios. Los dos archivos del checkpoint se publicaron
antes del commit atómico del manifest. El último `--resume` solo audita hashes
y genera tablas y selección de estabilidad, pues las tres grillas ya están completas.

## Tiempos y procedencia

`elapsed_seconds` en estos checkpoints mide la etapa de adjunción y comprobación
final. No incluye la búsqueda alternativa ni los intentos de búsqueda interrumpidos.
Los tiempos de respuestas exploratorias están en `competition.history` y en los
logs; en 201, los tiempos de certificación positiva también aparecen en
`competition.positive_polish_history`. El primer certificado completo de 201
empleó 137.76 s después de siete respuestas dispersas exploratorias de unos 1.8 s.
Las fuentes, scripts, propuestas y logs se registran con SHA-256 en el manifest.
