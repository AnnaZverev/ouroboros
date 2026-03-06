#!/usr/bin/env python3
"""
Скрипт для скачивания и анализа данных о солёности в Белом море
из продукта Copernicus Marine ARCTIC_ANALYSISFORECAST_PHY_002_001
за период 2023-2024.
"""

import os
import sys
import subprocess
from datetime import datetime

def install_deps():
    required = ['xarray', 'netcdf4', 'numpy', 'pandas', 'matplotlib', 'requests']
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            print(f"Установка {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', pkg])

def get_opendap_candidates():
    return [
        "https://nrt.cmems-du.eu/thredds/dodsC/cop_merc/phy-oi-025-025/ARCTIC_ANALYSISFORECAST_PHY_002_001/latest",
        "https://nrt.cmems-du.eu/thredds/dodsC/ARCTIC_ANALYSISFORECAST_PHY_002_001/latest",
        "https://nrt.cmems-du.eu/thredds/dodsC/dataset-opendap/ARCTIC_ANALYSISFORECAST_PHY_002_001/latest",
    ]

def try_open_dataset():
    candidates = get_opendap_candidates()
    for url in candidates:
        print(f"Пробуем OPeNDAP: {url}")
        try:
            import xarray as xr
            ds = xr.open_dataset(url, decode_times=True)
            print(f"Успешно открыто: {url}")
            return ds
        except Exception as e:
            print(f"Не удалось: {e}")
            continue
    raise RuntimeError("Не удалось открыть ни один из OPeNDAP endpoints")

def find_salinity_var(ds):
    for cand in ['so', 'salinity', 'SAL', 'SALT']:
        if cand in ds:
            return cand
    raise KeyError("В датасете не найдена переменная солёности (so, salinity, SAL, SALT)")

def find_coord_vars(ds):
    lon_var = lat_var = depth_var = None
    for var in ['longitude', 'lon', 'x', 'nav_lon']:
        if var in ds:
            lon_var = var
            break
    for var in ['latitude', 'lat', 'y', 'nav_lat']:
        if var in ds:
            lat_var = var
            break
    for var in ['depth', 'lev', 'z', 'deptht', 'depthu', 'depthv']:
        if var in ds:
            depth_var = var
            break
    if lon_var is None or lat_var is None:
        raise ValueError("Не удалось найти координаты longitude/latitude")
    return lon_var, lat_var, depth_var

def select_time_range(var, start_date='2023-01-01', end_date='2024-12-31'):
    # Найти временную координату
    time_coord = None
    for coord in ['time', 'time_counter', 'valid_time']:
        if coord in var.coords:
            time_coord = coord
            break
    if time_coord is None:
        raise ValueError("Не удалось определить временную координату")
    try:
        sliced = var.sel({time_coord: slice(start_date, end_date)})
    except Exception as e:
        raise ValueError(f"Ошибка выбора временного слоя: {e}")
    if sliced.sizes.get(time_coord, 0) == 0:
        raise ValueError("Нет данных за указанный период")
    return sliced

def filter_region(var, lon_var, lat_var, bbox=(30, 64, 40, 70)):
    lon = var[lon_var]
    lat = var[lat_var]
    region = var.where(
        (lon >= bbox[0]) & (lon <= bbox[2]) &
        (lat >= bbox[1]) & (lat <= bbox[3]),
        drop=True
    )
    if region.size == 0:
        raise ValueError("Регион не содержит данных. Проверьте bbox.")
    return region

def compute_timeseries(region, depth_var):
    spatial_dims = [lat_var]
    if lon_var not in spatial_dims:
        spatial_dims.append(lon_var)
    spatial_mean = region.mean(dim=spatial_dims)
    if depth_var and depth_var in spatial_mean.dims:
        depth_vals = spatial_mean[depth_var].values
        idx = int(np.argmin(np.abs(depth_vals)))
        surface = spatial_mean.isel({depth_var: idx})
        return surface
    return spatial_mean

def to_dataframe(timeseries, salinity_var_name):
    import xarray as xr
    import pandas as pd
    if isinstance(timeseries, xr.DataArray):
        df = timeseries.to_dataframe().reset_index()
    else:
        other_vars = [v for v in timeseries.data_vars if v != salinity_var_name]
        if other_vars:
            var_name = other_vars[0]
        else:
            var_name = salinity_var_name
        df = timeseries[var_name].to_dataframe().reset_index()
    value_cols = [c for c in df.columns if c not in ['time', 'latitude', 'longitude', 'lon', 'lat', 'depth']]
    value_col = value_cols[0] if value_cols else salinity_var_name
    df = df.rename(columns={value_col: 'salinity'})
    return df

def save_outputs(df, output_dir, salinity_var):
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'white_sea_salinity_2023_2024.csv')
    df.to_csv(csv_path, index=False)
    print(f"Сохранён CSV: {csv_path}")

    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['salinity'], linewidth=1)
    plt.title('Солёность в Белом море (средняя по поверхности), 2023-2024')
    plt.xlabel('Дата')
    plt.ylabel('Солёность (PSU)')
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(output_dir, 'salinity_timeseries_2023_2024.png')
    plt.savefig(plot_path, dpi=100)
    plt.close()
    print(f"Сохранён график: {plot_path}")

    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.to_period('M')
    seasonal = df.groupby('month')['salinity'].mean().reset_index()
    seasonal_csv = os.path.join(output_dir, 'white_sea_salinity_seasonal_2023_2024.csv')
    seasonal.to_csv(seasonal_csv, index=False)
    print(f"Сохранён сезонный CSV: {seasonal_csv}")

    plt.figure(figsize=(10, 5))
    plt.plot(seasonal['month'].astype(str), seasonal['salinity'], marker='o', linewidth=2)
    plt.title('Среднемесячная солёность в Белом море, 2023-2024')
    plt.xlabel('Месяц')
    plt.ylabel('Солёность (PSU)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    seasonal_plot = os.path.join(output_dir, 'salinity_seasonal_2023_2024.png')
    plt.savefig(seasonal_plot, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Сохранён график сезонных средних: {seasonal_plot}")

    stats = df['salinity'].describe()
    report = f"""# Анализ солёности в Белом море за 2023-2024

Данные: Copernicus Marine Service, продукт `ARCTIC_ANALYSISFORECAST_PHY_002_001`

## Методология

- Источник: OPeNDAP endpoint (попытка подключения к нескольким кандидатам)
- Период: 2023-01-01 — 2024-12-31
- Регион: Белое море, bbox (30, 64, 40, 70)
- Параметр: солёность ({salinity_var})
- Агрегация: среднее по пространству (широта/долгота) и по глубине (поверхность)

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

- Copernicus Marine OPeNDAP может требовать авторизацию. Если скрипт не смог подключиться, используйте CLI `copernicusmarine`.
- Для больших объёмов данных рекомендуется серверный subsetting.
- Регион Белого моря может требовать уточнения bbox.

## Технические детали

Скрипт автоматически устанавливает зависимости: xarray, netcdf4, numpy, pandas, matplotlib, requests.
"""
    report_path = os.path.join(output_dir, 'report_2023_2024.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Создан отчёт: {report_path}")

def main():
    OUTPUT_DIR = 'data/white_sea_analysis'
    start_date = '2023-01-01'
    end_date = '2024-12-31'
    bbox = (30, 64, 40, 70)  # [lon_min, lat_min, lon_max, lat_max]

    install_deps()
    import numpy as np
    import xarray as xr

    try:
        ds = try_open_dataset()
    except Exception as e:
        print(f"Ошибка подключения к OPeNDAP: {e}")
        print("Возможно, требуется авторизация или endpoint изменился.")
        print("Альтернатива: использовать CLI `copernicusmarine` с токеном.")
        print("Инструкция: pip install copernicusmarine, затем copernicusmarine login.")
        return

    print("Датасет открыт. Переменные:", list(ds.variables.keys()))

    try:
        salinity_var_name = find_salinity_var(ds)
        salinity = ds[salinity_var_name]
    except KeyError as e:
        print(e)
        return

    print(f"Используем переменную: {salinity_var_name}")
    print("Размерности:", salinity.dims)
    print("Доступные координаты:", list(salinity.coords.keys()))

    try:
        lon_var, lat_var, depth_var = find_coord_vars(salinity)
        print(f"Координаты: lon={lon_var}, lat={lat_var}, depth={depth_var}")
    except ValueError as e:
        print(e)
        return

    try:
        time_slice = select_time_range(salinity, start_date, end_date)
    except ValueError as e:
        print(e)
        return

    if time_slice.time.size == 0:
        print("Нет данных за указанный период.")
        return

    print(f"Выбран период: {time_slice.time.min().values} — {time_slice.time.max().values}")

    try:
        region = filter_region(time_slice, lon_var, lat_var, bbox)
    except ValueError as e:
        print(e)
        return

    print(f"После фильтрации по региону: {region.sizes}")

    try:
        timeseries = compute_timeseries(region, depth_var)
    except Exception as e:
        print(f"Ошибка агрегации: {e}")
        return

    try:
        df = to_dataframe(timeseries, salinity_var_name)
    except Exception as e:
        print(f"Ошибка преобразования в DataFrame: {e}")
        return

    try:
        save_outputs(df, OUTPUT_DIR, salinity_var_name)
        print("Анализ завершён успешно.")
    except Exception as e:
        print(f"Ошибка сохранения результатов: {e}")

if __name__ == '__main__':
    main()
