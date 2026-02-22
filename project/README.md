# Detector de anomalías en rendimientos logarítmicos para FOREX

Proyecto en Python para descargar precios de Forex desde Yahoo Finance (vía `yfinance`) o cargarlos desde CSV, limpiar los datos, calcular log returns y detectar anomalías con un enfoque combinado **MAD robusto + EWMA volatility**.

## Qué hace el pipeline

1. Descarga/lectura de datos (Yahoo o CSV).
2. Limpieza y validación de timestamps/duplicados/NaN.
3. Marcado de posibles problemas de feed (`data_issue`) por heurística de spike-reversión.
4. Construcción de serie de precio (`close` o `mid` con bid/ask).
5. Cálculo de `log_return`.
6. Detección de barras anómalas con `z_mad` y `z_ewma`.
7. Filtro por spread (si existe) para separar problemas de liquidez.
8. Agrupación de barras en eventos con cooldown.
9. Clasificación heurística de eventos:
   - `liquidity/spread`
   - `spike/reversion`
   - `break/event`
   - `regime_shift`
   - `unknown`
10. Exportación de resultados en CSV.

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

### Ejemplo 1 (Yahoo 5m)

```bash
python main.py --source yahoo --symbol EURUSD=X --interval 5m --period 30d --price_mode close --output_dir output
```

### Ejemplo 2 (Daily max)

```bash
python main.py --source yahoo --symbol GBPUSD=X --interval 1d --period max --price_mode close --output_dir output
```

### Ejemplo 3 (CSV)

```bash
python main.py --source csv --input data.csv --price_mode close --output_dir output
```

## Salidas

Dentro de `output_dir`:

- `raw_prices.csv` (si fuente Yahoo)
- `anomaly_bars.csv`
- `anomaly_events.csv`

Además, el CLI imprime:
- Número total de eventos.
- Top 5 eventos por `max_abs_return`.

## Nota importante sobre Yahoo intradía

Los datos intradía de Yahoo/yfinance normalmente **no superan ~60 días** y el intervalo **1m** suele estar limitado a alrededor de **~7 días**.
