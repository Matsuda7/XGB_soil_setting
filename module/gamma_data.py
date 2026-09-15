"""Audit raw density/depth data and prepare both gamma learning tables."""
import json
import numpy as np
import pandas as pd
from module.paths import RAW, TRAINING
from module.soil_tests import load_test_records, project_records
from module.gamma_config import KINDS, load_config, data_signature
from module.gamma_features import load_dem, terrain_at, geographic_context, enrich


def density_targets(frame, gravity=9.80665):
    """Convert independently measured wet/dry densities, preserving missing targets."""
    result = frame.copy()
    for kind in KINDS:
        density = pd.to_numeric(result[kind+'_density'],errors='coerce')
        result[kind+'_density'] = density
        result['gamma_'+kind] = (density*gravity).where(np.isfinite(density) & (density>0))
    result['dry_exceeds_wet'] = result.dry_density > result.wet_density
    result.loc[result.dry_exceeds_wet, ['gamma_wet','gamma_dry']] = np.nan
    return result


def prepare(config=None, raw=None):
    config = config or load_config()
    output = TRAINING/'gamma';output.mkdir(parents=True,exist_ok=True)
    raw = density_targets(load_test_records(RAW/'soiltest/strength') if raw is None else raw,config['gravity_m_s2'])
    raw['record_id'] = np.arange(len(raw))
    considered = raw[['wet_density','dry_density']].notna().any(axis=1)
    frame = raw.loc[considered].copy()
    frame['depth'] = pd.to_numeric(frame.depth_mid,errors='coerce')
    keys=['xml_file','sample_id','sample_no','depth','wet_density','dry_density']
    duplicates = frame.duplicated(keys)
    rejected = [frame.loc[duplicates].assign(exclusion_reason='duplicate')]
    frame = frame.loc[~duplicates].copy()
    invalid = ~np.isfinite(frame.depth) | (frame.depth<0) | frame[['gamma_wet','gamma_dry']].isna().all(axis=1)
    rejected.append(frame.loc[invalid].assign(exclusion_reason='invalid_density_or_depth'))
    frame = frame.loc[~invalid].copy()
    if frame.empty:
        raise ValueError('No usable gamma measurements after density/depth checks')
    frame, bad_coordinates = project_records(frame, RAW/'soiltest/strength', config['spatial']['grid_crs'])
    rejected.append(bad_coordinates.assign(exclusion_reason='invalid_coordinates'))
    if frame.empty:
        raise ValueError('No gamma measurements with valid coordinates')
    dem = load_dem(config)
    outside = terrain_at(frame,dem,config).surface_z.isna()
    if config['exclude_outside_dem']:
        rejected.append(frame.loc[outside].assign(exclusion_reason='outside_dem'))
        frame = frame.loc[~outside].copy()
    if frame.empty:
        raise ValueError('No gamma measurements inside the DEM')
    geology,jshis = geographic_context(config,frame)
    frame = enrich(frame,config,dem,geology,jshis)
    if config['n_comparison']['enabled']:
        from module.n_matching import prepare_n_data
        frame = prepare_n_data(frame, config, output, dem, geology, jshis)
    frame.drop(columns=['_row'],errors='ignore').to_csv(output/'measurements.csv',index=False,encoding='utf-8-sig')
    pd.concat(rejected,ignore_index=True).to_csv(output/'excluded_records.csv',index=False,encoding='utf-8-sig')
    columns=['record_id','boring_id','xml_file','sample_id','sample_no','x','y','latitude','longitude','depth',
             'surface_z','sample_z','slope','curvature','dem_missing','symbol','ser','jshis_jcode','jshis_avs30','jshis_arv']
    if config['n_comparison']['enabled']:
        from module.n_matching import MATCH_COLUMNS
        columns += ['depth_top','depth_bottom','source_boring_id','comparison_hole_id',*MATCH_COLUMNS]
    report={'status':'prepared','density_unit':'g/cm3','target_unit':'kN/m3',
            'raw_density_rows':int(considered.sum()),'raw_density_boreholes':int(raw.loc[considered,'boring_id'].nunique()),
            'duplicate_rows':int(duplicates.sum()),'outside_dem_rows':int(outside.sum()),
            'dry_exceeds_wet_rows':int(raw.dry_exceeds_wet.sum()),'signature':data_signature(config),'variants':{}}
    for kind in KINDS:
        data = frame.loc[frame['gamma_'+kind].notna(),columns].copy()
        data.insert(0,'target',frame.loc[frame['gamma_'+kind].notna(),'gamma_'+kind].to_numpy())
        data.to_csv(output/f'{kind}.csv',index=False,encoding='utf-8-sig')
        report['variants'][kind]={'rows':len(data),'boreholes':int(data.boring_id.nunique()),
            'target_min':float(data.target.min()),'target_max':float(data.target.max()),
            'depth_min':float(data.depth.min()),'depth_max':float(data.depth.max()),
            'missing_features':data[config['numeric_features']+config['categorical_features']].isna().sum().to_dict()}
        if data.boring_id.nunique()<5:
            report['status']='insufficient_data'
    (output/'preparation_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report
