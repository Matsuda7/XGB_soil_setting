"""Mandatory feature-gain and target-frequency reports for fitted XGBoost models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def original_feature(name,numeric,categorical):
    raw=str(name).split('__',1)[-1]
    if raw in numeric: return raw
    for feature in sorted(categorical,key=len,reverse=True):
        if raw==feature or raw.startswith(feature+'_'): return feature
    raise ValueError(f'Cannot map encoded feature to configured predictor: {name}')


def frequency_report(values,output,label,filename='target_distribution'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    raw=np.asarray(values,dtype=float).reshape(-1)
    values=raw[np.isfinite(raw)]
    if not len(values): raise ValueError('No finite training targets for frequency report')
    counts,edges=np.histogram(values,bins=20)
    table=pd.DataFrame({'lower_inclusive':edges[:-1],'upper':edges[1:],
                        'upper_inclusive':[False]*19+[True],'count':counts})
    table.to_csv(output/f'{filename}.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5))
    bars=ax.bar(edges[:-1],counts,width=np.diff(edges),align='edge',edgecolor='white')
    for bar,count in zip(bars,counts):
        if count: ax.annotate(str(count),(bar.get_x()+bar.get_width()/2,bar.get_height()),
                             xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
    ax.set(xlabel=label,ylabel='Number of training records',title='Training target frequency')
    ax.set_ylim(0,max(int(counts.max())*1.2,1));ax.grid(axis='y',alpha=.2)
    ax.text(.98,.95,f'n = {len(values):,}',transform=ax.transAxes,ha='right',va='top')
    fig.tight_layout();fig.savefig(output/f'{filename}.png',dpi=180);plt.close(fig)
    return {'records':len(raw),'finite_records':len(values),'excluded_nonfinite':len(raw)-len(values),
            'minimum':float(values.min()),'maximum':float(values.max()),'bins':20,
            'count_type':'unweighted record counts; not boreholes or predicted grid cells'}


def gain_table(model,preprocessor,numeric,categorical):
    names=preprocessor.get_feature_names_out()
    booster=model.get_booster()
    best=getattr(model,'best_iteration',None)
    if best is not None:
        booster=booster[:int(best)+1]
    scores=booster.get_score(importance_type='gain')
    gains=np.array([scores.get(str(name),scores.get(f'f{i}',0.)) for i,name in enumerate(names)],dtype=float)
    total=gains.sum()
    encoded=pd.DataFrame({'encoded_feature':names,
                          'feature':[original_feature(n,numeric,categorical) for n in names],
                          'gain':gains,'contribution_percent':100*gains/total if total>0 else np.zeros(len(gains))})
    grouped=encoded.groupby('feature',as_index=False)[['gain','contribution_percent']].sum()
    # Include configured predictors dropped during imputation (e.g. all-missing).
    grouped=grouped.set_index('feature').reindex([*numeric,*categorical],fill_value=0).reset_index()
    return encoded.sort_values('contribution_percent',ascending=False),grouped.sort_values('contribution_percent',ascending=False)


def contribution_plot(table,label_column,path,title):
    shown=table.head(30).sort_values('contribution_percent')
    fig,ax=plt.subplots(figsize=(10,max(4,len(shown)*.35+1)))
    ax.barh(shown[label_column],shown.contribution_percent)
    ax.set(xlabel='Normalized gain contribution (%)',title=title)
    ax.grid(axis='x',alpha=.2)
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


def save_model_reports(model,preprocessor,numeric,categorical,targets,output,target_label,
                       population='model fitting records',density_gravity=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    encoded,grouped=gain_table(model,preprocessor,numeric,categorical)
    encoded.to_csv(output/'feature_contribution_encoded.csv',index=False)
    grouped.to_csv(output/'feature_contribution.csv',index=False)
    contribution_plot(encoded,'encoded_feature',output/'feature_contribution_encoded.png','Encoded predictor contributions (top 30)')
    contribution_plot(grouped,'feature',output/'feature_contribution.png','Predictor contributions (top 30)')
    report=frequency_report(targets,output,target_label)
    if density_gravity is not None:
        frequency_report(np.asarray(targets,dtype=float)/density_gravity,output,
                         'Density (g/cm³)','density_distribution')
    report.update(target_label=target_label,population=population,
                  importance_method='XGBoost gain normalized across encoded predictors; summed by original predictor',
                  tree_scope='best_iteration inclusive for early stopping; all trees otherwise',
                  interpretation='model split importance, not a causal effect or SHAP value')
    (output/'report_metadata.json').write_text(json.dumps(report,indent=2)+'\n')
