import pandas as pd
from numpy import ceil


def cutting(column_name, path_in, path_out):
    for column in column_name:
        df = pd.read_csv(f'input\\{path_in}')[column]
        len_df = len(df)

        step = 8760

        df[0:step].to_csv(
                f'input\\annual segments\\{path_out}\\{column}\\{column}_0.csv',
                index=False)

        i = 1
        while i < int(ceil(len_df / step)):
            df[step * i:step * (i + 1)].to_csv(
                    f'input\\annual segments\\{path_out}\\{column}\\{column}_{i}.csv',
                    index=False)

            i = i + 1


# column_name = ['AEP_hourly', 'COMED_hourly', 'DAYTON_hourly', 'DEOK_hourly',
#               'DOM_hourly', 'DUQ_hourly', 'EKPC_hourly', 'FE_hourly', 'NI_hourly',
#               'PJM_Load_hourly', 'PJME_hourly', 'PJMW_hourly'
#               ]

# column_name = ['dew_point_kelvins', 'dew_point_degrees', 'feels_like_kelvins',
#                'feels_like_degrees', 'temp_min_kelvins', 'temp_min_degrees',
#                'temp_max_kelvins', 'temp_max_degrees', 'pressure', 'humidity', 'wind_speed',
#                'wind_deg', 'rain_1h', 'snow_1h', 'leftovers', 'clouds_all', 'weather_id',
#                'weather_description'
#                ]

# column_name = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
#                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

column_name = ['Total Cooling (kW)', 'Total Heating (kW)', 'Total Mechanical (kW)',
               'Total Lighting (kW)', 'Total Plug Loads (kW)', 'Total Data Center (kW)',
               'Total Building (kW)', 'PV (kW)', 'Building Net (kW)']

name = 'nrel-rsf-measured-data-2011'
path_in = f'{name}.csv'
path_out = f'energy\\{name}'
cutting(column_name, path_in, path_out)
