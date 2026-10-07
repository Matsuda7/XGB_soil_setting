"""Step 4-1: local soil inventory/SolphyJ audit only; no model training."""
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import box
from module.paths import ROOT


def main():
    out = ROOT/'gamma_test/soil_inventory_audit'; out.mkdir(parents=True, exist_ok=True)
    folder = ROOT/'data/raw/soil_inventory'
    zips = [p for p in folder.rglob('*.zip') if any(n.lower().endswith('.shp') for n in zipfile.ZipFile(p).namelist())]
    books = list(folder.rglob('*Solphy*.xlsx'))
    if len(zips)!=1 or len(books)!=1:
        raise ValueError('Expected one soil-map archive and one SolphyJ workbook')
    soil = gpd.read_file('zip://'+str(zips[0]), encoding='cp932')
    fields = [c for c in soil if c!='geometry']
    attributes = {c: sorted(soil[c].dropna().unique().tolist()) for c in fields}
    (out/'soil_attributes.json').write_text(json.dumps(dict(file=str(zips[0]),crs=str(soil.crs),polygons=len(soil),unique_values=attributes),ensure_ascii=False,indent=2)+'\n')
    workbook = pd.ExcelFile(books[0])
    dictionary = pd.read_excel(workbook, sheet_name='Soil Series Group')
    dictionary.to_csv(out/'ssg_dictionary.csv', index=False)
    sheets, long = [], []
    columns=['Land use','SSG code','Soil Series Group','Horizon','Number of samples','Mean','SD','Median']
    for name in workbook.sheet_names:
        if name=='Soil Series Group': continue
        raw=pd.read_excel(workbook,sheet_name=name,header=None)
        headers=raw.iloc[3].tolist()
        values=pd.DataFrame({c:raw.iloc[4:,headers.index(c)].to_numpy() for c in columns})
        values['property']=name
        for c in ['Number of samples','Mean','SD','Median']:
            values[c]=pd.to_numeric(values[c],errors='coerce')
        long.append(values)
        sheets.append(dict(file=books[0].name,sheet=name,property_title=str(raw.iloc[0,0]),rows=len(values),
            ssg_codes=values['SSG code'].nunique(),ssg_names=values['Soil Series Group'].nunique(),
            positive_n_mean_rows=int(((values['Number of samples']>0)&values.Mean.notna()).sum()),
            columns_present=';'.join(columns),land_use=';'.join(sorted(values['Land use'].dropna().unique())),
            horizons=';'.join(map(str,sorted(values.Horizon.dropna().unique())))))
    sol=pd.concat(long,ignore_index=True)
    sol.to_csv(out/'solphy_original_scale.csv', index=False)
    pd.DataFrame(sheets).to_csv(out/'solphy_sheets.csv',index=False)
    codes=set(sol['SSG code'].dropna())
    valid=sol.loc[(sol['Number of samples']>0)&sol.Mean.notna()]
    usable=set(valid['SSG code'])
    mapcodes=set(soil.SSerGrCD.dropna())
    matches=mapcodes&codes
    pd.DataFrame([dict(code=c,map_present=c in mapcodes,solphy_present=c in codes,has_any_measured_mean=c in usable) for c in sorted(mapcodes|codes)]).to_csv(out/'code_crosswalk.csv',index=False)
    availability=valid.groupby(['property','SSG code']).size().unstack(fill_value=0)
    availability.to_csv(out/'property_code_availability.csv')
    frame=pd.read_csv(ROOT/'gamma_test/routed_models/augmented.csv')
    spatial=json.loads((ROOT/'config/spatial.json').read_text())
    projected=soil.to_crs(spatial['grid_crs'])
    points=gpd.GeoDataFrame(frame[['record_id','label_source','x','y']],geometry=gpd.points_from_xy(frame.x,frame.y),crs=spatial['grid_crs'])
    joined=gpd.sjoin(points,projected[['SSerGrCD','SoilName','geometry']],how='left',predicate='within')
    extra=len(joined)-len(frame)
    # Do not silently pick a soil when multiple polygons disagree.
    ambiguous=joined.groupby('record_id').SSerGrCD.nunique()
    ambiguous=set(ambiguous[ambiguous>1].index)
    joined=joined.sort_values(['record_id','index_right']).drop_duplicates('record_id')
    joined.loc[joined.record_id.isin(ambiguous),'SSerGrCD']=pd.NA
    joined['soil_category']=joined.SSerGrCD.fillna('UNKNOWN')
    joined['solphy_code_match']=joined.SSerGrCD.isin(codes)
    joined['solphy_any_mean']=joined.SSerGrCD.isin(usable)
    joined.drop(columns=['geometry','index_right']).to_csv(out/'teacher_soil_join.csv',index=False)
    counts=pd.crosstab(joined.soil_category,joined.label_source).reindex(columns=['measured','predicted'],fill_value=0)
    counts.to_csv(out/'teacher_category_counts.csv')
    lookup=joined.set_index('record_id').soil_category
    cv=[]
    for seed in [42,43,44,45,46]:
        base=ROOT/f'gamma_test/stage2_baseline_borehole/borehole_{seed}'
        split=pd.read_csv(base/'measured_split.csv')
        augmented=pd.read_csv(base/'training.csv')
        test=lookup.loc[split.loc[split.split=='test','record_id']]
        for condition, ids in [('measured_only',split.loc[split.split=='train','record_id']),('D_first_augmented',augmented.record_id)]:
            train=lookup.loc[ids];unknown=set(test)-set(train)
            cv.append(dict(seed=seed,condition=condition,train_categories=';'.join(sorted(set(train))),
                test_categories=';'.join(sorted(set(test))),unseen_categories=';'.join(sorted(unknown)),
                test_count=len(test),unseen_count=int(test.isin(unknown).sum()),unseen_rate=float(test.isin(unknown).mean()),
                test_missing=int(test.eq('UNKNOWN').sum())))
    pd.DataFrame(cv).to_csv(out/'borehole_holdout_categories.csv',index=False)
    bounds=spatial['property_mask_bounds']; roi=box(bounds['xmin'],bounds['ymin'],bounds['xmax'],bounds['ymax'])
    geoms=projected.geometry.make_valid()
    covered=shapely.union_all(geoms).intersection(roi)
    matched=shapely.union_all(geoms[projected.SSerGrCD.isin(codes)]).intersection(roi)
    measured_area=shapely.union_all(geoms[projected.SSerGrCD.isin(usable)]).intersection(roi)
    area=dict(rectangle_km2=roi.area/1e6,map_covered_km2=covered.area/1e6,code_matched_km2=matched.area/1e6,
        code_fraction_of_rectangle=matched.area/roi.area,code_fraction_of_map_covered=matched.area/covered.area,
        any_mean_fraction_of_rectangle=measured_area.area/roi.area)
    # Existing prediction mask: area fractions estimated by 10m grid-point membership.
    for geom in [covered,matched,measured_area]: shapely.prepare(geom)
    totals=np.zeros(4,dtype=np.int64)
    h=spatial['property_mask_spacing_m']
    with (ROOT/spatial['property_mask']).open() as stream:
        for row,line in enumerate(stream):
            mask=np.fromstring(line,sep=' ')
            indices=np.flatnonzero(mask!=0)
            x=bounds['xmin']+indices*h; y=np.full(len(x),bounds['ymin']+row*h)
            totals[0]+=len(x)
            for k,g in enumerate([covered,matched,measured_area],1):totals[k]+=np.count_nonzero(shapely.intersects_xy(g,x,y))
    area['prop_mask_gridpoint_counts']=totals.tolist()
    area['prop_mask_approx_km2']=float(totals[0]*h*h/1e6)
    area['prop_mask_code_fraction']=float(totals[2]/totals[0])
    area['prop_mask_any_mean_fraction']=float(totals[3]/totals[0])
    area['prop_mask_method']='10 m nonzero mask grid points, equal area weights; approximate, not exact polygon area'
    summary=dict(map_polygons=len(soil),map_classes=len(mapcodes),solphy_codes=len(codes),solphy_names=sol['Soil Series Group'].nunique(),
        direct_matches=len(matches),unmatched_map=sorted(mapcodes-codes),solphy_only=sorted(codes-mapcodes),
        matching_codes_without_any_mean=sorted(matches-usable),
        overlap_extra_matches=extra,ambiguous_points=len(ambiguous),area=area,
        teacher_counts={label:dict(total=len(d),soil_obtained=int(d.SSerGrCD.notna().sum()),
            solphy_match=int(d.solphy_code_match.sum()),any_mean=int(d.solphy_any_mean.sum())) for label,d in joined.groupby('label_source')})
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='solphy_only'},ensure_ascii=False,indent=2))
    print(pd.DataFrame(cv)[['seed','condition','test_count','unseen_count','unseen_rate']].to_string(index=False))


if __name__=='__main__': main()
