"""Match density intervals to measured SPTs without vertical interpolation."""
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from module.paths import project_path
from module.spt_dataset import load_boring_measurements, reshape_spt_measurements


MATCH_COLUMNS = ['n_measured', 'n_match_method', 'spt_boring_id', 'spt_xy_distance_m',
                 'spt_candidate_holes', 'spt_count', 'spt_depths_m', 'spt_measurement_numbers',
                 'spt_depth_mean_m', 'spt_depth_difference_m', 'spt_max_depth_difference_m',
                 'nearest_spt_depth_m', 'nearest_spt_depth_difference_m']


def load_spt(config):
    path = project_path(config['spt_input'])
    wide = load_boring_measurements(path, 80)
    long, _ = reshape_spt_measurements(wide, 80)
    holes = pd.read_csv(path, encoding='utf-8-sig', usecols=['ファイル名','坑口座標X','坑口座標Y'])
    holes.columns = ['boring_id','x','y']
    holes[['x','y']] = holes[['x','y']].apply(pd.to_numeric, errors='coerce')
    long = long.merge(holes, on='boring_id', validate='many_to_one')
    valid = np.isfinite(long[['x','y','depth','n_value']].to_numpy(dtype=float)).all(axis=1)
    return long.loc[valid].reset_index(drop=True)


def match_measurements(density, spt, config):
    holes = spt[['boring_id','x','y']].drop_duplicates('boring_id').reset_index(drop=True)
    if holes.empty:
        raise ValueError('No finite SPT measurements')
    tree = cKDTree(holes[['x','y']].to_numpy())
    grouped = {str(k): v for k,v in spt.groupby('boring_id')}
    rows = []
    for _, sample in density.iterrows():
        indices = tree.query_ball_point([sample.x, sample.y], config['xy_tolerance_m'])
        row = {c: np.nan for c in MATCH_COLUMNS}
        row.update(n_match_method='no_matching_hole', spt_candidate_holes=len(indices),
                   spt_count=0, spt_depths_m='[]', spt_measurement_numbers='[]', spt_boring_id='')
        if len(indices) > 1:
            row['n_match_method'] = 'ambiguous_hole'
        if len(indices) == 1:
            hole = holes.iloc[indices[0]]
            values = grouped[str(hole.boring_id)]
            row['spt_boring_id'] = str(hole.boring_id)
            row['spt_xy_distance_m'] = float(np.hypot(sample.x-hole.x, sample.y-hole.y))
            distances = (values.depth-sample.depth).abs()
            nearest = values.loc[distances.idxmin()]
            row['nearest_spt_depth_m'] = float(nearest.depth)
            row['nearest_spt_depth_difference_m'] = float(abs(nearest.depth-sample.depth))
            top, bottom = sample.get('depth_top', np.nan), sample.get('depth_bottom', np.nan)
            epsilon = config['depth_tolerance_m']
            if pd.notna(top) and pd.notna(bottom):
                if not np.isfinite([top,bottom]).all() or top < 0 or bottom < top:
                    row['n_match_method'] = 'invalid_sample_interval'
                    rows.append(row)
                    continue
                selected = values.loc[values.depth.between(top-epsilon, bottom+epsilon)]
                method = 'interval_mean'
            else:
                selected = values.loc[distances <= epsilon]
                method = 'exact_depth_mean'
            row['n_match_method'] = method if len(selected) else 'no_spt_in_interval'
            if len(selected):
                row.update(n_measured=float(selected.n_value.mean()), spt_count=len(selected),
                    spt_depths_m=json.dumps(selected.depth.tolist()),
                    spt_measurement_numbers=json.dumps(selected.measurement_no.astype(int).tolist()),
                    spt_depth_mean_m=float(selected.depth.mean()),
                    spt_depth_difference_m=float(selected.depth.mean()-sample.depth),
                    spt_max_depth_difference_m=float((selected.depth-sample.depth).abs().max()))
        rows.append(row)
    result = density.reset_index(drop=True).copy()
    result[MATCH_COLUMNS] = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    # Identity for splitting uses the physical location, not just XML filenames.
    result['source_boring_id'] = result.boring_id
    result['comparison_hole_id'] = location_groups(result, config['xy_tolerance_m'])
    return result


def prepare_n_data(frame, config, output, dem, geology, jshis):
    from module.gamma_features import terrain_at, enrich
    options = config['n_comparison']
    raw_spt = load_spt(options)
    frame = match_measurements(frame, raw_spt, options)
    frame[['record_id','source_boring_id','comparison_hole_id','depth_top','depth_bottom',
           'depth','x','y',*MATCH_COLUMNS]].to_csv(output/'n_matching.csv', index=False)
    # Geographic covariates are target independent. Build them at holes once.
    holes = raw_spt[['boring_id','x','y']].drop_duplicates('boring_id').assign(depth=0.)
    if config['exclude_outside_dem']:
        holes = holes.loc[terrain_at(holes, dem, config).surface_z.notna()].copy()
    if holes.empty:
        raise ValueError('No SPT holes inside the permitted DEM extent')
    enriched = enrich(holes, config, dem, geology, jshis)
    features = enriched.drop(columns=['depth','sample_z','_row'], errors='ignore')
    spt = raw_spt.merge(features, on=['boring_id','x','y'], how='inner', validate='many_to_one')
    spt['n_value_elevation'] = spt.surface_z-spt.depth
    spt['sample_z'] = spt.n_value_elevation
    spt.to_csv(output/'n_spt.csv', index=False)
    report = {'scope':f"soil-test measurements retained for {config.get('model', 'gamma')} training",
              'rows':len(frame), 'matched_rows':int(frame.n_measured.notna().sum()),
              'unmatched_rows':int(frame.n_measured.isna().sum()),
              'methods':frame.n_match_method.value_counts().to_dict(),
              'n_cap':50, 'settings':options,
              'interval_rule':'mean of capped SPT values; not a point measurement at midpoint',
              'nearest_values_are_diagnostic_only':True}
    (output/'n_matching_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return frame


def location_groups(frame, tolerance):
    """Connected components of XY neighbours; every depth of one ID stays together."""
    sites = frame.groupby('boring_id', sort=True)[['x','y']].median()
    parents = list(range(len(sites)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for a,b in cKDTree(sites.to_numpy()).query_pairs(tolerance):
        ra,rb=find(a),find(b)
        parents[max(ra,rb)]=min(ra,rb)
    labels={key:f'location_{find(i)}' for i,key in enumerate(sites.index)}
    return frame.boring_id.map(labels)
