# POC – Analítica táctica de regatas (TP SOFTWARE · Entrega 2)

Flujo completo con datos reales: **ingesta → variables → viento → indicadores → predicción → visualización**.

```bash
pip install -r requirements.txt
python run_poc.py <carpeta con telemetry.csv.gz, races.csv, routes.csv, ...> results
```
Tarda ~3 min en 1 CPU con las 16 regatas del dataset chico (`tractrac_raw_optimist`).

## Estructura (monolito modular, como en el PRD)
| Módulo | Qué hace |
|---|---|
| `src/ingest.py` | Limpieza, normalización (m locales, nudos, rumbo 0-360), recorrido y asignación de piernas |
| `src/wind.py` | Estimador de viento **separado** del modelo, con confianza y protocolo de validación (Fase 0) |
| `src/analytics.py` | Viradas/trasluchadas, costo, variables relativas a la flota, ventanas de ganancia/pérdida |
| `src/model.py` | Modelo predictivo (GBM) + referencias + variante sin viento |
| `src/viz.py` | Trayectoria coloreada verde (ganó) → rojo (perdió) |

## Qué se hizo con los datos
- **Limpieza** (`results/cleaning_report.json`): 82 outliers de velocidad (>15 kn), 33 saltos de GPS, 63 barcos con <200 puntos fuera. Rumbo con signo → 0-360. Velocidad km/h → nudos. lat/lon → metros locales por regata.
- Se excluyó 1 regata ("Memorial", recorrido con marcas inconsistentes: 1ª pierna de 32 m). Quedan **15 regatas, 803 participaciones, ~134 mil snapshots** (uno cada 15 s por barco).
- **Piernas** a partir de `routes.csv` + `control_passings.csv`; tipo de pierna (U/R/D) según el rumbo respecto del eje salida→primera marca.

## Viento (estimador propio)
En cada ventana causal de 4 min se separan los rumbos estables de la flota en dos bordadas (babor/estribor) y el viento es la **bisectriz**. Confianza = tamaño de muestra × plausibilidad del ángulo entre bordadas.
- Semi-ángulo entre bordadas en ceñida: **48-57°** (lo esperado en Optimist).
- Consistencia: partiendo la flota en dos mitades al azar, las estimaciones difieren **0,9° de mediana** (máx. 2,8° en las regatas de 14 barcos).
- Cobertura: 53-98 % de las ventanas con estimación válida (mediana 73 %). Confianza mediana 0,8.
- Popa: casi no hay trasluchadas (van derecho a la boya) → ahí no hay información y no se usa.
- **Intensidad**: no se intentó (sigue siendo "deseable").

## Indicadores
- 10.064 viradas, 145 trasluchadas, 3.498 redondeos de boya, 773 giros "otros".
- Costo de virada: mediana ~1,3 m / 1,7 s (**cota baja**: GPS cada 3 s suaviza el frenazo).
- 8.833 ventanas de ganancia/pérdida (umbral ±16 m en 60 s): 4.765 de ganancia y 4.068 de pérdida. Las de pérdida tienen 1,03 viradas en promedio y las de ganancia 0,57.
- **Referencia de flota y sesgo de supervivencia.** La ganancia se mide contra la mediana de la flota *todavía en carrera*. Cuando los líderes terminan y salen, esa mediana se corre y los barcos que quedan parecen "ganar" sin hacerlo (con <50 % de la flota activa, la ganancia media pasaba de ~+1 m a +50 m o más). Por eso la ganancia solo se calcula mientras siga en carrera >= 75 % de los barcos (`MIN_FLEET_FRAC` en `src/analytics.py`); fuera de ese rango el gráfico muestra el tramo en **gris** ("sin referencia confiable") y esos instantes no entran al modelo.

## Modelo predictivo
Target: metros que el barco gana/pierde contra la mediana de la flota en los **próximos 60 s**. Evaluado siempre en regatas no vistas.

| Modelo | Regatas no vistas (MAE m) | R² | Spearman | Evento entero (MAE m) | R² |
|---|---|---|---|---|---|
| Constante | 22,3 | -0,00 | -0,03 | 22,3 | -0,00 |
| Persistencia (últ. 60 s) | 20,6 | 0,15 | 0,35 | 20,7 | 0,13 |
| GBM sin viento | **17,2** | 0,37 | 0,61 | 19,4 | 0,26 |
| GBM con viento | 17,4 | 0,37 | 0,60 | **18,3** | 0,31 |

**Lectura honesta:** el modelo mejora claramente a las dos referencias en ambos esquemas. **El viento todavía no mejora el modelo de forma concluyente:** empeora levemente (17,2 -> 17,4) al validar con regatas no vistas y mejora al dejar afuera un evento entero (19,4 -> 18,3). Con 15 regatas no alcanza para afirmar nada; es lo que debe resolver la Entrega 3 con 50-100 regatas (`tractrac_raw_optimist_history`). La acción (`act_tack_30`) es una entrada, no una recomendación: no hay causalidad.

## Resultados (`results/`)
`telemetria_limpia.csv.gz` (dataset limpio y normalizado) · `features_snapshots.csv.gz` · `maniobras.csv.gz` · `ventanas_ganancia_perdida.csv` · `validacion_viento.csv` · `metricas_modelo.csv` · `importancia_variables.csv` · `trayectoria_verde_rojo.html` (índice con selector de las 15 regatas; cada una en `trayectorias/`, con desplegable de todos los barcos ordenados por llegada; boyas rotuladas por función; pasar el mouse muestra viento, ganancia y predicción).

## Limitaciones / para el MVP
- Se asume que la 1ª pierna es de ceñida (eje salida→marca 1). Si una regata no cumple, la confianza baja sola.
- Marcas fijas (mediana de posición); en el MVP conviene usar su posición en el tiempo.
- Falta: intensidad del viento, viento local, calibrar costo de viradas con más resolución, tests y CI.
