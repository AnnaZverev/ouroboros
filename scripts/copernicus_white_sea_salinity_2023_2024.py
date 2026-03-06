#!/usr/bin/env python3
"""
Скрипт для скачивания и анализа данных о солёности в Белом море
из продукта Copernicus Marine ARCTIC_ANALYSISFORECAST_PHY_002_001
за период 2023-2024.

Если OPeNDAP недоступен или требуется авторизация, можно использовать
альтернативные методы (CLI copernicusmarine или прямая загрузка).
"""

import os
import sys
import subprocess
from datetime import datetime

# Установка зависимостей, если нет
def install_deps():
    required = ['xarray', 'netcdf4', 'numpy', 'pandas', 'matplotlib', 'requests']
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            print(f"Установка {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', pkg])

install_deps()

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Настройка
OUTPUT_DIR = 'data/white_sea_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Возможные OPeNDAP endpoints для ARCTIC_ANALYSISFORECAST_PHY_002_001
# Приоритет: сначала вероятный NRT endpoint, затем старый.
OPENDAP_CANDIDATES = [
    "https://nrt.cmems-du.eu/thredds/dodsC/cop_merc/phy-oi-025-025/ARCTIC_ANALYSISFORECAST_PHY_002_001/latest",
    "https://nrt.cmems-du.eu/thredds/dodsC/ARCTIC_ANALYSISFORECAST_PHY_002_001/latest",
    "https://nrt.cmems-du.eu/thredds/dodsC/dataset-opendap/ARCTIC_ANALYSISFORECAST_PHY_002_001/latest",
]

# Белое море bbox: [lon_min, lat_min, lon_max, lat_max]
# Приблизительные координаты (в градусах, WGS84):
WHITE_SEA_BBOX = [30, 64, 40, 70]  # [30°E, 64°N, 40°E, 70°N]

def try_open_dataset():
    for url in OPENDAP_CANDIDATES:
        print(f"Пробуем OPeNDAP: {url}")
        try:
            ds = xr.open_dataset(url, decode_times=True)
            print(f"Успешно открыто: {url}")
            return ds
        except Exception as e:
            print(f"Не удалось: {e}")
            continue
    raise RuntimeError("Не удалось открыть ни один из OPeNDAP endpoints")

def download_and_analyze():
    # Открыть датасет
    try:
        ds = try_open_dataset()
    except Exception as e:
        print(f"Ошибка подключения к OPeNDAP: {e}")
        print("Возможно, требуется авторизация или endpoint изменился.")
        print("Альтернатива: использовать CLI `copernicusmarine` с токеном.")
        print("Инструкция: pip install copernicusmarine, затем copernicusmarine login.")
        return

    print("Датасет открыт. Переменные:", list(ds.variables.keys()))

    # Переменная солёности (обычно 'so' или 'salinity')
    salinity_var = None
    for cand in ['so', 'salinity', 'SAL', 'SALT']:
        if cand in ds:
            salinity_var = cand
            break
    if salinity_var is None:
        print("В датасете не найдена переменная солёности.")
        return

    salinity = ds[salinity_var]
    print(f"Используем переменную: {salinity_var}")
    print("Размерности:", salinity.dims)
    print("Доступные координаты:", list(salinity.coords.keys()))

    # Временной диапазон 2023-2024
    start_date = '2023-01-01'
    end_date = '2024-12-31'
    # Попробуем обрезать
    try:
        salinity = salinity.sel(time=slice(start_date, end_date))
    except Exception as e:
        print(f"Ошибка выбора временного слоя: {e}")
        print("Возможно, временная ось называется иначе.")
        # Попробуем найти временную ось
        time_coord = None
        for coord in ['time', 'time_counter', 'valid_time']:
            if coord in salinity.coords:
                time_coord = coord
                break
        if time_coord:
            print(f"Найдена временная координата: {time_coord}")
            salinity = salinity.sel({time_coord: slice(start_date, end_date)})
        else:
            print("Не удалось определить временную координату.")
            return

    if salinity.time.size == 0:
        print("Нет данных за указанный период.")
        return

    print(f"Выбран период: {salinity.time.min().values} — {salinity.time.max().values}")

    # Определение координатных переменных
    lon_var = None
    lat_var = None
    depth_var = None
    for var in ['longitude', 'lon', 'x', 'nav_lon']:
        if var in salinity.coords:
            lon_var = var
            break
    for var in ['latitude', 'lat', 'y', 'nav_lat']:
        if var in salinity.coords:
            lat_var = var
            break
    for var in ['depth', 'lev', 'z', 'deptht', 'depthu', 'depthv']:
        if var in salinity.coords:
            depth_var = var
            break

    print(f"Координаты: lon={lon_var}, lat={lat_var}, depth={depth_var}")

    if lon_var is None or lat_var is None:
        print("Не удалось найти координаты longitude/latitude.")
        return

    lon = salinity[lon_var]
    lat = salinity[lat_var]

    # Copernicus Arctic вероятно использует 0-360? Проверим диапазон
    if lon.min() >= 0 and lon.max() > 180:
        # Преобразуем bbox в 0-360
        bbox = [30, 64, 40, 70]  # остаётся
    else:
        bbox = WHITE_SEA_BBOX

    # Фильтрация по региону Белого моря
    region = salinity.where(
        (lon >= bbox[0]) & (lon <= bbox[2]) &
        (lat >= bbox[1]) & (lat <= bbox[3]),
        drop=True
    )
    print(f"После фильтрации по региону: {region.sizes}")

    if region.size == 0:
        print("Регион не содержит данных. Проверьте bbox.")
        return

    # Агрегация по пространству (среднее)
    spatial_dims = [lat_var]
    if lon_var not in spatial_dims:
        spatial_dims.append(lon_var)
    spatial_mean = region.mean(dim=spatial_dims)

    # Если есть глубина, выбираем поверхность (ближайшую к 0)
    if depth_var and depth_var in spatial_mean.dims:
        # Найти уровень, ближайший к 0
        depth_vals = spatial_mean[depth_var].values
        idx = np.argmin(np.abs(depth_vals))
        surface = spatial_mean.isel({depth_var: idx})
        timeseries = surface
    else:
        timeseries = spatial_mean

    # Преобразуем в pandas DataFrame
    # timeseries может иметь несколько измерений, кроме времени
    # Упростим: если это DataArray, загрузим значения
    if isinstance(timeseries, xr.DataArray):
        df = timeseries.to_dataframe().reset_index()
    else:
        # Dataset? Берём первую переменную
        var_name = list(timeseries.data_vars)[0]
        df = timeseries[var_name].to_dataframe().reset_index()

    # Переименуем столбец со значением
    value_col = [c for c in df.columns if c not in ['time', 'latitude', 'longitude', 'lon', 'lat', 'depth']][0]
    df = df.rename(columns={value_col: 'salinity'})

    # Сохраняем CSV
    csv_path = os.path.join(OUTPUT_DIR, 'white_sea_salinity_2023_2024.csv')
    df.to_csv(csv_path, index=False)
    print(f"Сохранён CSV: {csv_path}")

    # График временного ряда
    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['salinity'], linewidth=1)
    plt.title('Солёность в Белом море (средняя по поверхности), 2023-2024')
    plt.xlabel('Дата')
    plt.ylabel('Солёность (PSU)')
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(OUTPUT_DIR, 'salinity_timeseries_2023_2024.png')
    plt.savefig(plot_path, dpi=100)
    plt.close()
    print(f"Сохранён график: {plot_path}")

    # Сезонная агрегация (среднее по месяцам)
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.to_period('M')
    seasonal = df.groupby('month')['salinity'].mean().reset_index()
    seasonal_csv = os.path.join(OUTPUT_DIR, 'white_sea_salinity_seasonal_2023_2024.csv')
    seasonal.to_csv(seasonal_csv, index=False)
    print(f"Сохранён сезонный CSV: {seasonal_csv}")

    # График сезонных средних
    plt.figure(figsize=(10, 5))
    plt.plot(seasonal['month'].astype(str), seasonal['salinity'], marker='o', linewidth=2)
    plt.title('Среднемесячная солёность в Белом море, 2023-2024')
    plt.xlabel('Месяц')
    plt.ylabel('Солёность (PSU)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    seasonal_plot = os.path.join(OUTPUT_DIR, 'salinity_seasonal_2023_2024.png')
    plt.savefig(seasonal_plot, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Сохранён график сезонных средних: {seasonal_plot}")

    # Создание markdown отчёта
    stats = df['salinity'].describe()
    report = f"""# Анализ солёности в Белом море за 2023-2024

Данные: Copernicus Marine Service, продукт `ARCTIC_ANALYSISFORECAST_PHY_002_001`

## Методология

- Источник: OPeNDAP endpoint (попытка подключения к нескольким кандидатам)
- Период: {start_date} — {end_date}
- Регион: Белое море, bbox {bbox}
- Параметр: солёность ({salinity_var})
- Агрегация: среднее по пространству (широта/долгота) и по глубине (поверхность, если есть вертикальные уровни)

## Результаты

- Количество временных шагов: {len(df)}
- Средняя солёность за период: {df['salinity'].mean():.3f} PSU
- Стандартное отклонение: {df['salinity'].std():.3f} PSU
- Минимум: {df['salinity'].min():.3f} PSU
- 25%: {stats['25%']:.3f} PSU
- Медиана: {stats['50%']:.3f} PSU
- 75%: {stats['75%']:.3f} PSU
- Максимум: {df['salinity'].max():.3f} PSU

### Графики

![Временной ряд]({os.path.basename(plot_path)})
![Сезонные средние]({os.path.basename(seasonal_plot)})

## Примечания

- Copernicus Marine OPeNDAP может требовать авторизации (учётная запись). Если скрипт не смог подключиться, используйте CLI `copernicusmarine` следующей командой:
  ```
  pip install copernicusmarine
  copernicusmarine login --username YOUR_USER --password YOUR_PASS
  copernicusmarine describe --dataset-id ARCTIC_ANALYSISFORECAST_PHY_002_001
  ```
  и адаптируйте скрипт для загрузки через `copernicusmarine` API.
- Для больших объёмов данных рекомендуется использовать subsetting на стороне сервера (через OPeNDAP constraints).
- Регион Белого моря может требовать уточнения bbox; координаты даны приблизительно.

## Технические детали

Скрипт автоматически устанавливает зависимости: xarray, netcdf4, numpy, pandas, matplotlib, requests.
"""
    report_path = os.path.join(OUTPUT_DIR, 'report_2023_2024.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Создан отчёт: {report_path}")

    print("Анализ завершён.")

if __name__ == '__main__':
    download_and_analyze()
