import pandas as pd

reference_columns = [
    'ID',
    'Country',
    'Location',
    'Coordinate system',
    'X/lon',
    'Y/lat',
    'Year',
    'Day+month',
    'River/coast',
    'Levee/dam',
    'Survival/failure',
    #Below is used to build a model
    'Water_level_from_NAP',
    'Altitude',
    'Water_level_diff',
    'Seepage_length',
    'Aquifer_thickness',
    'Blanket_thickness',
    'd10',
    'd50',
    'd60',
    'd70',
    'Friction_angle',
    'Hydraulic_conductivity',
    'Hydraulic_conductivity_KC',
    'Porosity',
    'Permeability',
    'Tan_Friction_angle',
    'Bedding_angle',
    'Tan_Bedding_angle',
    'Global_gradient',
    'Uniformity_coefficient',
    
    #metadata
    'Source',
    'Remarks',
    
    #source columns
    'Soil type (source)',
    'Source_Water_level_diff',
    'Source_Seepage_length',
    'Source_Aquifer_thickness',
    'Source_Blanket_thickness',
    'Source_Friction_angle',
    'Source_Hydraulic_conductivity',
    'Source_Porosity',
    'Source_d10', 
    'Source_d50', 
    'Source_d60', 
    'Source_d70',
    
    'Kozeny_Carman',
    
    #BRO data columns
    'BHRG_data',
    'BHRP_data',
    'BHRGT_data',
    'BRO_Blanket_thickness',
    'BRO_Friction_angle',
    'BRO_Hydraulic_conductivity',
    'BRO_Porosity',
    'BRO_d10', 
    'BRO_d50', 
    'BRO_d60', 
    'BRO_d70',
    
    #GeoTOP & REGIS columns
    'GeoTOP_data',
    'GeoTOP_Aquifer_Thickness',
    'GeoTOP_Aquifer_Soils',
    'GeoTOP_Blanket_Thickness',
    'GeoTOP_Friction_angle',
    'GeoTOP_Hydraulic_conductivity',
    'GeoTOP_Porosity',
    'GeoTOP_d10',
    'GeoTOP_d50',
    'GeoTOP_d60',
    'GeoTOP_d70',
    
    'REGIS_data',
    #'REGIS_Aquifer_Thickness',
    #'REGIS_Blanket_Thickness',
    #'REGIS_kh',
    #'REGIS_kv',
    
    #RWS data
    'RWS_Waterlevel_data',
    'FR',
    'FS',
    'FG',
    'H_c_sellmeijer',
    'H_c_schmertmann',
    'H_c_bligh',
    'H_c_BT',
    'H_c_SD',
]                 

reference_df = pd.DataFrame(columns=reference_columns, dtype='object')


imputation_features = ['Water_level_diff', 'Seepage_length', 'Friction_angle', 
                      'Aquifer_thickness', 'Blanket_thickness',
                      'd10', 'd50', 'd60', 'd70', 'Hydraulic_conductivity', 'Porosity']

modelling_features = ['Water_level_diff', 'Seepage_length', 'Friction_angle', 
                      'Aquifer_thickness', 'Blanket_thickness', 'd10', 'd50', 
                      'Uniformity_coefficient', 'Hydraulic_conductivity_KC', 'Porosity',
                      'Bedding_angle']

all_modelling_features = ['Water_level_diff', 'Seepage_length', 'Aquifer_thickness', 'Blanket_thickness',
                          'd10', 'd50', 'd60', 'd70', 'Uniformity_coefficient', 'Hydraulic_conductivity', 
                          'Hydraulic_conductivity_KC', 'Porosity', 'Bedding_angle', 'Tan_Bedding_angle', 
                          'Friction_angle', 'Tan_Friction_angle']
