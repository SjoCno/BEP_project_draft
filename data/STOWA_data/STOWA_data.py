def get_STOWA_df():
    import pandas as pd
    import sys
    import os
        
    from reference_columns import reference_df
    
    rdf = reference_df.copy()
    
    file_path = os.path.join(os.path.dirname(__file__), "Wellen_csv.xlsx")
    keep_cols = [
    "DATUM", "LIGGING", "SIT_OMSC", "X_WEL", "Y_WEL", "Z_WEL", "Z_SLKANT", 
    "MEER_WEL", "W_V_1", "W_AARD", "W_KLEUR", "W_ZG_MT", "W_Z_M3", "W_ERNST", 
    "KRATER", "WATERSCH", "WAT_LOB", "Latitude", "Longitude"
]
    
    stowa_df = pd.read_excel(file_path, sheet_name='in')
    del_cols = list(set(stowa_df.columns.tolist())-set(keep_cols))
    stowa_df = stowa_df[keep_cols]
    stowa_df['Country'] = 'The Netherlands'
    stowa_df['X/lon'] = stowa_df['Longitude']
    stowa_df['Y/lat'] = stowa_df['Latitude']
    stowa_df['Coordinate system'] = 'EPSG:4326'
    stowa_df['Location'] =  stowa_df['LIGGING'].astype(str) + " " + stowa_df['SIT_OMSC'].astype(str)
    
    
    stowa_df['DATUM'] = stowa_df['DATUM'].astype(str)
    
    stowa_df['Year'] = stowa_df['DATUM'].str[:4]
    stowa_df['Day+month'] = stowa_df['DATUM'].str[5:11]
    stowa_df['Day+month'] = stowa_df['DATUM'].str[5:11].str.split('-').str[::-1].str.join('-')
    stowa_df = stowa_df.drop(columns=['DATUM'])
    stowa_df['River/coast'] = 'River'
    stowa_df['Levee/dam'] = 'Levee'
    stowa_df['Survival/failure'] = 'Survival'
    stowa_df['Source'] = 'STOWA database'
    
    stowa_df = stowa_df.reindex(columns=rdf.columns.to_list())
    stowa_df['ID'] = stowa_df['Source'].str[:4] + stowa_df.index.astype(str).str.zfill(4)
    
    
    # Remove all values where both ΔH and date not filled in -> not retrievable
    stowa_df['Day+month'] = stowa_df['Day+month'].replace('', pd.NA)
    stowa_df['Year'] = stowa_df['Year'].replace(pd.NaT, pd.NA)
    stowa_df['Year'] = stowa_df['Year'].replace('NaT', pd.NA)
    stowa_df = stowa_df[~((stowa_df['Water_level_diff'].isna()) & (stowa_df['Year'].isna()) & (stowa_df['Day+month'].isna()))]
    
    
    #stowa_df = stowa_df.drop(columns=['X_WEL', 'Y_WEL', 'LIGGING', 'SIT_OMSC', 'Latitude','Longitude'])
    #stowa_df = stowa_df.drop(columns=['Z_WEL', 'Z_SLKANT', 'MEER_WEL', 'W_V_1', 'W_AARD', 'W_KLEUR', 'W_ZG_MT', 'W_Z_M3', 'W_ERNST', 'KRATER', 'WATERSCH', 'WAT_LOB'])
    #stowa_df['Dropped columns'] = str(del_cols+keep_cols)
    
    
    return stowa_df
