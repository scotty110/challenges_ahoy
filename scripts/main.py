'''
Read in CSV fils (no geopandas)
'''

import pandas as pd
import numpy as np
import os 
import requests
from os.path import join
from pathlib import Path
from functools import reduce

from shapely.geometry import Point
from shapely import wkt

def make_dir(dir_path:str):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return


def get_csv(file_path:str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def calc_ndvi(nir:float, red:float):
    return (nir - red) / (nir + red)


def filter_pos(a,b) -> bool:
    return a == b


def soil_weights(hzdept:float, hzdepb:float, comppct:float) -> float:
    return ( abs(hzdept - hzdepb) / hzdepb ) * (comppct/100.)


def get_fips_code(lonlat:tuple):
    lonlat = lonlat[0]
    lon = lonlat[0]
    lat = lonlat[1]
    url = 'https://geo.fcc.gov/api/census/block/find?latitude=%s&longitude=%s&format=json' % (lat, lon)
    response = requests.get(url)
    data = response.json()
    state = data['State']['FIPS']
    county = data['County']['FIPS'][2:]
    return int(state+county)


def challenge_1(crop_csv:str, output_dir:str) -> str:
    '''
    Read in data from crop.csv. Extract data from year 2021,
    Then save a csv with the following cols (field_id, field_geometry, crop_type)
    Inputs:
        - crop_csv (str): Path to CSV containing the crop data
        - output_dir (str): path to save the csv to
    '''
    output_path = join(output_dir, 'crop.csv') 
    
    df = get_csv( crop_csv )
    df = df[ df['year'] == 2021 ]
    df = df[['field_id', 'field_geometry', 'crop_type']]
    df.to_csv(output_path, index=False)
    return output_path


def challenge_2(csv_path:str, output_dir:str) -> str:
    '''
    Normalize the difference Vegetation index, Then find Peak of season (max value each year),
    then date the POS for each Tile
    Inputs:
        - csv_path (str): Path where csv is located
        - output_dir (str): Path to save csv
    '''
    out_path = join(output_dir, 'spectral_pos.csv')
    
    df = get_csv( csv_path )
    df['ndvi'] = df[['nir','red']].apply(lambda x: calc_ndvi( *x), axis=1)

    max_df = df.groupby('tile_id')['ndvi'].max('ndvi').reset_index()
    max_df.rename(columns = {'ndvi':'max_ndvi'}, inplace=True)

    both_df = pd.merge(df, max_df, on='tile_id', how='inner')
    both_df['to_filter'] = both_df[['ndvi','max_ndvi']].apply(lambda x: filter_pos( *x), axis=1)
    both_df = both_df[ both_df['to_filter'] == True]
   
    both_df = both_df[['tile_id','tile_geometry','date','ndvi']]
    both_df.rename(columns = {'ndvi':'pos', 'date':'pos_date'}, inplace=True)
    both_df.to_csv(out_path, index=False)
    return out_path


def challenge_3(csv_path:str, output_dir:str) -> str:
    '''
    Calc average for horizonatal components, then compute the weighted average of components for each map unit
    Inputs:
        - csv_path (str): Path where csv is located
        - output_dir (str): Path to save csv
    '''
    out_path = join(output_dir, 'soil.csv')
    
    df = get_csv( csv_path )
    df['layer_weight'] = df[['hzdept','hzdepb','comppct']].apply(lambda x: soil_weights(*x), axis=1)

    df['om_weight'] = df['layer_weight'] * df['om']
    df['cec_weight'] = df['layer_weight'] * df['cec']
    df['ph_weight'] = df['layer_weight'] * df['ph']

    #Weighted OM
    og_df = df.groupby('mukey')['om_weight'].agg(np.mean).reset_index()
    og_df.rename(columns = {'om_weight':'om'}, inplace=True)
    
    #Weighted CEC 
    cg_df = df.groupby('mukey')['cec_weight'].agg(np.mean).reset_index()
    cg_df.rename(columns = {'cec_weight':'cec'}, inplace=True)
    
    #Weighted PH 
    pg_df = df.groupby('mukey')['ph_weight'].agg(np.mean).reset_index()
    pg_df.rename(columns = {'ph_weight':'ph'}, inplace=True)

    # Mukey Geo
    mukey = df[['mukey', 'mukey_geometry']].drop_duplicates(subset=['mukey']).reset_index()

    # Merge all together
    to_return = reduce(lambda x,y: pd.merge(x,y, on='mukey'), [og_df, cg_df, pg_df, mukey])
    to_return = to_return[['mukey', 'mukey_geometry', 'om', 'cec', 'ph']]

    # Save
    to_return.to_csv(out_path, index=False)
    return out_path


def challenge_4(weather_path:str, crop_path:str, output_dir:str) -> str:
    '''
    From the year 2021 computs: total rainfall, temp (min, max, mean)
    Join ...
    Inputs:
        - weather_path (str): Path where weather csv is located
        - crop_path (str): Path where crop csv is located
        - output_dir (str): Path to save csv
    '''
    out_path = join(output_dir, 'weather.csv')
    
    w_df = get_csv( weather_path )
    c_df = get_csv( crop_path )

    ''' Crop Field Fips codes '''
    # Calc field ID fips code
    c_df = c_df.drop_duplicates(subset=['field_id']).reset_index()
    c_df['geom'] = c_df['field_geometry'].apply(wkt.loads)
    c_df['lonlat'] = c_df['geom'].apply(lambda x: x.centroid.coords)
    c_df['fips_code'] = c_df['lonlat'].apply(lambda x: get_fips_code(x) )
    c_df = c_df[['field_id','fips_code']]

    ''' Weather'''
    # Filter to 2021
    w_df = w_df[ w_df['year'] == 2021 ]

    # total_rain
    rain_df = w_df.groupby('fips_code')['precip'].agg(np.sum).reset_index()
    rain_df.rename(columns={'precip':'yr_precip'}, inplace=True)

    #max, min, mean temp
    max_df = w_df.groupby('fips_code')['temp'].agg(np.max).reset_index()
    max_df.rename(columns={'temp':'max_temp'}, inplace=True)

    min_df = w_df.groupby('fips_code')['temp'].agg(np.min).reset_index() 
    min_df.rename(columns={'temp':'min_temp'}, inplace=True)

    mean_df = w_df.groupby('fips_code')['temp'].agg(np.mean).reset_index() 
    mean_df.rename(columns={'temp':'avg_temp'}, inplace=True)
    
    # Agg into one df
    w_df = reduce(lambda x,y: pd.merge(x,y, on='fips_code'), [rain_df, max_df, min_df, mean_df])

    ''' Combine to get weather for field_ids '''
    df = pd.merge(w_df, c_df, on ='fips_code')
    df.rename(columns={'yr_precip':'precip', 'avg_temp':'mean_temp'}, inplace=True) # Probably just don't rename above
    df = df[['field_id','precip', 'min_temp', 'max_temp', 'mean_temp']]
    df.to_csv(out_path, index=False)
    return out_path




if __name__=='__main__':
    out_dir = '../outputs'
    make_dir(out_dir)

    crop_csv = '../inputs/crop.csv'
    challenge_1(crop_csv, out_dir)

    spectral_csv = '../inputs/spectral.csv'
    challenge_2(spectral_csv, out_dir)

    soil_csv = '../inputs/soil.csv'
    challenge_3(soil_csv, out_dir)

    weather_csv = '../inputs/weather.csv'
    challenge_4(weather_csv, crop_csv, out_dir)











