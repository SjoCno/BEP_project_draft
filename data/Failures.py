def get_failure_df():

    import pandas as pd
    import sys
    import os
        
    from reference_columns import reference_df
    
    rdf = reference_df.copy()
    
    Nieuwkuijk_data = {
        'Country': 'The Netherlands',
        'Location': 'Nieuwkuijk',
        'Year': '1880',
        'Day+month': '29/12',
        'River/coast': 'River',
        'Levee/dam': 'Levee',
        'Survival/failure': 'Failure',
        'Source_Water_level_diff': {'type': 'range', 'min': 1.7, 'max': 3.74},
        'Source_Seepage_length': {'type': 'range', 'min': 27, 'max': 37}, #Buijs said max 32, Kanning said 37
        'Soil type (source)': 'Moderately fine sand',
        'Source': 'Buijs (2013) / Kanning (2012)',
        'Source_Aquifer_thickness': {'type': 'range','min': 38, 'max': 64},
        'Source_Blanket_thickness': {'type': 'range', 'min': 0, 'max': 2.8},
        'Source_Hydraulic_conductivity': {'type': 'range', 'min': 0.000198, 'max': 0.000388},
        'Remarks': '(1) Failed after digging ditches on the outer side [Kanning (2012)]'
    }
    
    Strijenham_data = {
        'Country': 'The Netherlands',
        'Location': 'Strijenham',
        'Year': '1894',
        'Day+month': '29/12',
        'River/coast': 'River',
        'Levee/dam': 'Levee',
        'Survival/failure': 'Failure',
        'Source_Water_level_diff': {'type': 'range', 'min': 0.56, 'max': 3.42},
        'Source_Seepage_length': {'type': 'range', 'min': 40, 'max': 60},
        'Soil type (source)': 'Moderately fine sand',
        'Source': 'Buijs (2013)',
        'Remarks': 'Previous, higher flood',
        'Source_Aquifer_thickness': {'type': 'range', 'min': 60, 'max': 60},
        'Source_Blanket_thickness': {'type': 'range', 'min': 2.5, 'max': 6.7},
        'Source_Hydraulic_conductivity': {'type': 'range', 'min': 9.259e-5, 'max': 0.000150},
    }
    
    Zalk_data = {
        'Country': 'The Netherlands',
        'Location': 'Zalk',
        'Year': '1926',
        'Day+month': '8/1',
        'River/coast': 'River',
        'Levee/dam': 'Levee',
        'Survival/failure': 'Failure',
        'Source_Water_level_diff': {'type': 'range', 'min': 1.15, 'max': 2.13},
        'Source_Seepage_length': {'type': 'range', 'min': 40, 'max': 60},
        'Soil type (source)': 'Coarse sand',
        'Source': 'Buijs (2013)',
        'Source_Aquifer_thickness': {'type': 'range', 'min': 15, 'max': 100},
        'Source_Blanket_thickness': {'type': 'range', 'min': 2.4, 'max': 4.},
        'Source_Hydraulic_conductivity': {'type': 'range', 'min': 0.000368, 'max': 0.000704},
        'Remarks': '(3) Rerouted dike at location of previous failure, failed during high river discharges'
    }
    
    Oosterhout_data = {
        'Country': 'The Netherlands',
        'Location': 'Oosterhout',
        'Year': '1820',
        'Day+month': '23/1',
        'River/coast': 'River',
        'Levee/dam': 'Levee',
        'Survival/failure': 'Failure',
        'Source_Water_level_diff': {'type': 'range', 'min': 3.61, 'max': 5.67},
        'Source_Seepage_length': {'type': 'range', 'min': 32, 'max': 180},
        'Soil type (source)': 'Coarse sand/gravel',
        'Source': 'Buijs (2013)',
        'Source_Aquifer_thickness': {'type': 'range', 'min': 44, 'max': 44},
        'Source_Blanket_thickness': {'type': 'range', 'min': 2.4, 'max': 4.},
        'Source_Hydraulic_conductivity': {'type': 'range', 'min': 0.000610, 'max': 0.001066},
    }
    
    Afferden_data = {
        'Country': 'The Netherlands',
        'Location': 'Afferden',
        'Year': '1784',
        'Day+month': '21/2',
        'River/coast': 'River',
        'Levee/dam': 'Levee',
        'Survival/failure': 'Failure',
        'Source_Water_level_diff': {'type': 'range', 'min': 4.77, 'max': 6.4},
        'Source_Seepage_length': {'type': 'range', 'min': 20, 'max': 38},
        'Soil type (source)': 'Coarse sand',
        'Source': 'Buijs (2013)',
        'Source_Aquifer_thickness': {'type': 'range', 'min': 17, 'max': 45},
        'Source_Blanket_thickness': {'type': 'range', 'min': 0, 'max': 2.},
        'Source_Hydraulic_conductivity': {'type': 'range', 'min': 0.000346, 'max': 0.000969},
    }
    
    Honjo_data = {
        'Country': 'Japan',
        'Location': 'Yabe River, Kyushu Island',
        'Year': '2012',
        'Day+month': '1/7',
        'River/coast': 'River',
        'Levee/dam': 'Levee',
        'Survival/failure': 'Failure',
        'Source_Water_level_diff': {'type': 'point', 'value': 8.36},
        'Soil type (source)': 'Moderately fine sand',
        'Source': 'Honjo (2015)',
        'Remarks': 'Flood duration of 5 hours'
    }
    
    Tholen_data = {
        'Country': 'The Netherlands',
        'Location': 'Tholen',
        'Year': '1894',
        'Day+month': '30-12',
        'River/coast': 'River',
        'Levee/dam': 'Levee',
        'Survival/failure': 'Failure',
        'Source_Water_level_diff': {'type': 'point', 'value': 2.5},
        'Source_Seepage_length': {'type': 'point', 'value': 30},
        'Source_Aquifer_thickness': {'type': 'point', 'value': 20},
        'Soil type (source)': 'Matig vast fijn zand',
        'Source': 'Kanning (2012)',
        'Remarks': '(2) Failure and sand fill river from the river Striene'
    }
    
    Trotters51_data = {
        'Country': 'USA',
        'Location': 'Trotters 51',
        'Year': '1937',
        'Day+month': pd.NA,
        'River/coast': 'River',
        'Levee/dam': 'Levee',
        'Survival/failure': 'Failure',
        'Source_Water_level_diff': {'type': 'point', 'value': 5.3},
        'Source_Blanket_thickness': {'type': 'point', 'value': 3.05},
        'Source_Aquifer_thickness': {'type': 'point', 'value': 30},
        'Source_Seepage_length': {'type': 'point', 'value': 66}, #ja dus of dit +70.8 want ze hadden die berm nog niet in 1937
        'Source_Hydraulic_conductivity': {'type': 'point', 'value': 1e-3},
        'Soil type (source)': '',
        'Source': 'Heemstra (2008)',
        'Remarks': ''
    }
    
    Stovall_data = {
        'Country': 'USA',
        'Location': 'Stovall',
        'Year': '1937',
        'Day+month': pd.NA,
        'River/coast': 'River',
        'Levee/dam': 'Levee',
        'Survival/failure': 'Failure',
        'Source_Water_level_diff': {'type': 'point', 'value': 5.4},
        'Source_Blanket_thickness': {'type': 'point', 'value': 3.2},
        'Source_Aquifer_thickness': {'type': 'point', 'value': 12.1},
        'Source_Seepage_length': {'type': 'point', 'value': 122}, #ja dus of dit +70.8 want ze hadden die berm nog niet in 1937
        'Source_Hydraulic_conductivity': {'type': 'point', 'value': 2.5e-3},
        'Soil type (source)': '',
        'Source': 'Heemstra (2008)',
        'Remarks': 'Beetje onduidelijk of het echt een failure is'
    }
    
    Lower_Jones_tract_data = {
        'Country': 'USA',
        'Location': 'Lower Jones tract, California',
        'Year': '1980',
        'Day+month': '26-09',
        'River/coast': 'River',
        'Levee/dam': 'Levee',
        'Survival/failure': 'Failure',
        'Source_Water_level_diff': {'type': 'point', 'value': 2.7},
        'Source_Seepage_length': {'type': 'point', 'value': 50},
        'Soil type (source)': 'Unknown, probably fine',
        'Source': 'Kanning (2012)',
        'Remarks': '(4) Possibly due to seepage and rodent activities (DRMS, 2009)'
    }
    
    McDonald_Island_data = {
        'Country': 'USA',
        'Location': 'McDonald Island, California',
        'Year': '1982',
        'Day+month': '23-08',
        'River/coast': 'River',
        'Levee/dam': 'Levee',
        'Survival/failure': 'Failure',
        'Source_Water_level_diff': {'type': 'point', 'value': 2.3},
        'Source_Seepage_length': {'type': 'point', 'value': 50},
        'Soil type (source)': 'Unknown, probably fine',
        'Source': 'Kanning (2012)',
        'Remarks': '(5) Possibly due to seepage from dredging at waterside toe (DRMS, 2009)'
    }
    
    Upper_Jones_tract_data = {
        'Country': 'USA',
        'Location': 'Upper Jones tract, California',
        'Year': '2004',
        'Day+month': '03-06',
        'River/coast': 'River',
        'Levee/dam': 'Levee',
        'Survival/failure': 'Failure',
        'Source_Water_level_diff': {'type': 'point', 'value': 2.7},
        'Source_Seepage_length': {'type': 'point', 'value': 65},
        'Soil type (source)': 'Unknown, probably fine',
        'Source': 'Kanning (2012)',
        'Remarks': '(6) Possibly due to high tide, under-seepage, and rodent activity (DRMS, 2009)'
    }
    
    London_Avenue_data = {
        'Country': 'USA',
        'Location': 'New Orleans, London Avenue Canal',
        'Year': '2005',
        'Day+month': pd.NA,
        'River/coast': 'River',
        'Levee/dam': 'Levee',
        'Survival/failure': 'Failure',
        'Source_Water_level_diff': {'type': 'point', 'value': 4.6},
        'Source_Seepage_length': {'type': 'point', 'value': 23},
        'Soil type (source)': 'Fine sand',
        'Source': 'Kanning (2012)',
        'Remarks': '(8) All leakage length is vertical'
    }
    
    Bois_Brule_Kaskaskia_data = {
        'Country': 'USA',
        'Location': 'Bois Brule / Kaskaskia',
        'Year': '1993',
        'Day+month': '22-7 / 25-07',
        'River/coast': 'River',
        'Levee/dam': 'Levee',
        'Survival/failure': 'Failure',
        'Source_Water_level_diff': {'type': 'point', 'value': 7.1},
        'Source_Seepage_length': {'type': 'point', 'value': 30.5},
        'Source_Aquifer_thickness': {'type': 'range', 'min': 30.48, 'max': 39.62},
        'Source_Blanket_thickness': {'type': 'range', 'min': 3.04, 'max': 3.25}, 
        'Source_d50': {'type': 'range', 'min': 0.25e-3, 'max': 0.50e-3}, 
        'Source_Hydraulic_conductivity': {'type': 'point', 'value': 1e-3},
        'Soil type (source)': 'Poorly graded sand or gravel with silt',
        'Source': 'Wiersma (2019), USACE papers'
    }
    
    Mengxi_dike_data = {
        'Country': 'China',
        'Location': 'Mengxi dike',
        'Year': '1998',
        'Day+month': '07-08',
        'River/coast': 'River',
        'Levee/dam': 'Levee',
        'Survival/failure': 'Failure',
        'Source_Water_level_diff': {'type': 'point', 'value': 6.7},
        'Source_Seepage_length': {'type': 'point', 'value': 63},
        'Source_Aquifer_thickness': {'type': 'point', 'value': 2.0},
        'Source_Blanket_thickness': {'type': 'range', 'min': 2.1, 'max': 2.7},
        'Source_d50': {'type': 'range', 'min': 0.25e-3, 'max': 0.50e-3},
        'Source_Hydraulic_conductivity': {'type': 'range', 'min': 1.6e-4, 'max': 6.8e-3}, #was 1.6E-4 – 6.8E-3 
        'Soil type (source)': 'Fine sand',
        'Source': 'van Beek et al. (2013)',
        'Remarks': ''
    }
    
    Paizhou_dike_data = {
        'Country': 'China',
        'Location': 'Paizhou dike',
        'Year': '1998',
        'Day+month': '01-08',
        'River/coast': 'River',
        'Levee/dam': 'Levee',
        'Survival/failure': 'Failure',
        'Source_Water_level_diff': {'type': 'point', 'value': 6.74},
        'Source_Seepage_length': {'type': 'point', 'value': 58},
        'Source_Aquifer_thickness': {'type': 'point', 'value': 30}, #was >30m
        'Source_Blanket_thickness': {'type': 'range', 'min': 3.3, 'max': 5},
        'Source_d50': {'type': 'range', 'min': 0.25e-3, 'max': 0.50e-3},
        'Source_Hydraulic_conductivity': {'type': 'range', 'min': 0.6e-4, 'max': 2.4e-5},
        'Soil type (source)': 'Fine sand',
        'Source': 'van Beek et al. (2013)',
        'Remarks': ''
    }
    
    
    Failure_list = [Nieuwkuijk_data, Strijenham_data, Zalk_data, Oosterhout_data, Afferden_data, Honjo_data, Tholen_data, Trotters51_data, Stovall_data,\
                    Lower_Jones_tract_data, McDonald_Island_data, Upper_Jones_tract_data, Bois_Brule_Kaskaskia_data, London_Avenue_data, Mengxi_dike_data, Paizhou_dike_data]
    Failure_df = pd.DataFrame(Failure_list)
    Failure_df = Failure_df.reindex(columns=rdf.columns.to_list())
    Failure_df['ID'] = Failure_df['Source'].str[:4] + Failure_df.index.astype(str).str.zfill(4)
    
    
    ### There is a little story behind the water level:
    # https://rivergages.mvr.usace.army.mil/WaterControl/stationinfo2.cfm?dt=S&fid=&sid=CE400FDC gauge zero is 341 ft
    # ERDC report toe altitude is 367.5 ft
    # https://www.mvs.usace.army.mil/Media/News-Stories/Article/3996159/many-celebrate-completion-of-bois-brule-levee-project/ 1993 crest peak was 49.7 ft
    # So the water level is 23.2 ft = 7.07 m
    # ERDC report dike width = 100 ft -> 30.48 m
    
    return Failure_df



