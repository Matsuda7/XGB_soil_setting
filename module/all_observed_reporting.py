"""Final-model fit diagnostics on all observed records, separate from heldout scores."""
import argparse
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from module.model_reporting import depth_scatter
from xgboost import XGBRegressor
from module.paths import project_path, TRAINING
from module.gamma_features import model_frame
from module.xgb_common import regression_metrics, prepare_feature_frame



def save_comparison_summary(output):
    """Keep heldout evaluation and final-model fit scores explicitly separate."""
    output = Path(output)
    metrics = json.loads((output/'metrics.json').read_text())
    evaluation = metrics if 'test' in metrics else metrics.get('evaluation', {})
    if 'test' not in evaluation or 'all_observed' not in metrics:
        return
    is_cv = 'cross_validation' in evaluation
    test = dict(evaluation['test'], rows=evaluation.get('rows', {}).get('test'),
                population='cross_validation_oof' if is_cv else 'heldout_test',
                model_source='5-fold CV models' if is_cv else 'heldout evaluation model',
                independent_test=True)
    (output/'test_metrics.json').write_text(json.dumps(test, indent=2)+'\n')
    (output/'test_r2.txt').write_text(
        f"Test R2: {test['r2']:.8f}\nRecords: {test['rows']}\n{test['model_source']}; excludes evaluated records from fitting.\n")
    reports = [('cv_oof' if is_cv else 'test', test), ('all_observed', metrics['all_observed'])]
    table = [dict(population=name, rows=report.get('rows'),
                  r2=report['r2'], mae=report.get('mae'), rmse=report.get('rmse'),
                  model_source=report['model_source'], independent_test=report['independent_test'])
             for name, report in reports]
    pd.DataFrame(table).to_csv(output/'comparison_metrics.csv', index=False, encoding='utf-8-sig')


def save_all_observed(frame, predictions, target_column, prediction_column, output,
                      filename, target_label, model_source='final refit model'):
    output = Path(output)
    observed = frame[target_column].to_numpy(dtype=float)
    predicted = np.asarray(predictions, dtype=float).reshape(-1)
    if len(observed) != len(predicted) or len(observed) < 2:
        raise ValueError('All-observed comparison needs matching arrays with at least two rows')
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError('All-observed comparison cannot silently discard nonfinite rows')
    if 'label_source' in frame and not frame.label_source.eq('measured').all():
        raise ValueError('All-observed comparison must not include pseudo targets')
    report = dict(**regression_metrics(observed, predicted), rows=len(observed),
        boreholes=int(frame.boring_id.nunique()) if 'boring_id' in frame else None,
        population='all retained observed records used in final fitting',
        model_source=model_source, evaluation_type='in_sample_final_model_fit',
        independent_test=False, target_label=target_label)
    output.mkdir(parents=True, exist_ok=True)
    main_image = output/filename
    test_image = main_image.with_name(main_image.stem+'_test'+main_image.suffix)
    # Preserve the previous holdout image on the first conversion only.
    if main_image.exists() and not test_image.exists() and not (output/'all_observed_metrics.json').exists():
        shutil.copyfile(main_image, test_image)
    columns = [c for c in ('record_id','boring_id','measurement_no','sample_id','x','y','depth',target_column) if c in frame]
    table = frame[list(dict.fromkeys(columns))].copy()
    table[prediction_column] = predicted
    table['residual'] = observed-predicted
    table.to_csv(output/'all_observed_predictions.csv', index=False, encoding='utf-8-sig')
    (output/'all_observed_metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (output/'all_observed_r2.txt').write_text(f"All observed R2: {report['r2']:.8f}\nRecords: {len(observed)}\nFinal-model fitting-data score; not an independent test score.\n")
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_facecolor('white')
    depth_scatter(ax, observed, predicted, frame.get('depth'), s=9 if len(observed)>2000 else 24,
               alpha=.2 if len(observed)>2000 else .65, edgecolors='none')
    low = min(float(observed.min()), float(predicted.min()))
    high = max(float(observed.max()), float(predicted.max()))
    span = max(high-low, 1.)
    ax.plot([low,high],[low,high],'k--',lw=1)
    ax.set(xlim=(low-.03*span,high+.03*span), ylim=(low-.03*span,high+.03*span),
           xlabel=f'Observed {target_label}', ylabel=f'Predicted {target_label}',
           title='All observed records vs. final-model predictions', aspect='equal')
    ax.text(.04,.96,f"R² (all observed) = {report['r2']:.4f}\nn = {len(observed):,}",
            transform=ax.transAxes,ha='left',va='top',bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
    ax.grid(alpha=.18)
    fig.text(.5,.015,'Includes model-fitting records; not independent test performance.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(main_image,dpi=200,bbox_inches='tight')
    fig.savefig(main_image.with_name(main_image.stem+'_all.png'),dpi=200,bbox_inches='tight')
    plt.close(fig)
    path = output/'metrics.json'
    if path.exists():
        metrics = json.loads(path.read_text())
        metrics['all_observed'] = report
        metrics['comparison_plot_population'] = 'all_observed'
        metrics['comparison_plot_populations'] = ['test', 'all_observed']
        path.write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
        save_comparison_summary(output)
    return report


def rebuild(run, target):
    run = project_path(run)
    if target == 'phi':
        from module.phi_training import load_dataset
        from module.phi_output import artifact_directory, organize_model_output
        output = artifact_directory(run/'model')
        metrics = json.loads((output/'metrics.json').read_text())
        frame = load_dataset(project_path(metrics.get('training_dataset', TRAINING/'phi/model_dataset.csv')),metrics['numeric_features'],
                             metrics['categorical_features'],metrics.get('training_max_depth_m'),
                             metrics.get('training_min_depth_m'), metrics.get('include_n50', True))
        model_path, pre_path = output/'xgb_spt_model.json', output/'xgb_preprocessor.joblib'
        target_column, predicted_column = 'n_value','predicted_n_value'
        filename, label = 'observed_vs_predicted_n.png','SPT N-value (capped at 50)'
    elif target == 'gamma_wet':
        output = run/'models/wet'
        metrics = json.loads((output/'metrics.json').read_text())
        if 'n_value' in metrics['numeric_features']:
            raise ValueError('Predicted-N gamma requires its saved final N features; not supported by this backfill')
        frame = pd.read_csv(TRAINING/'gamma/wet.csv')
        prep = json.loads((TRAINING/'gamma/preparation_summary.json').read_text())
        if metrics.get('preparation_signature') != prep['signature']:
            raise ValueError('Prepared gamma data do not match saved model configuration')
        model_path, pre_path = output/'model.json', output/'preprocessor.joblib'
        target_column, predicted_column = 'target','predicted_gamma_kn_m3'
        filename, label = 'observed_vs_predicted.png','wet gamma [kN/m³]'
    else:
        raise ValueError('Unsupported saved target')
    if len(frame) != metrics['final_fit_rows']:
        raise ValueError('Current dataset size differs from saved final fitting population')
    # Verify the saved holdout observations still agree with the prepared data.
    test = pd.read_csv(output/'test_predictions.csv')
    keys = ['boring_id','measurement_no'] if target == 'phi' else ['boring_id','x','y','depth',target_column]
    current = frame[keys+[target_column]] if target_column not in keys else frame[keys]
    reference = test[keys+[target_column]] if target_column not in keys else test[keys]
    matched = reference.merge(current.drop_duplicates(),on=keys,how='left',indicator=True,suffixes=('_saved','_current'))
    if not matched['_merge'].eq('both').all():
        raise ValueError('Saved holdout observations differ from current dataset')
    if target == 'phi' and not np.allclose(matched[target_column+'_saved'],matched[target_column+'_current']):
        raise ValueError('Saved N targets differ from current dataset')
    model = XGBRegressor(n_jobs=4);model.load_model(model_path)
    pre = joblib.load(pre_path)
    feature_builder = prepare_feature_frame if target == 'phi' else model_frame
    features = feature_builder(frame,metrics['numeric_features'],metrics['categorical_features'])
    predicted = model.predict(pre.transform(features))
    if target == 'phi':
        predicted = np.clip(predicted,0.,50.)
    report = save_all_observed(frame,predicted,target_column,predicted_column,output,filename,label,str(model_path))
    if target == 'phi':
        if output != run/'model':
            shutil.copyfile(output/'observed_vs_predicted_n_all.png', run/'model/observed_vs_predicted_n_all.png')
        organize_model_output(run/'model')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--target',choices=['phi','gamma_wet'],required=True)
    args=parser.parse_args()
    print(json.dumps(rebuild(args.run,args.target),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
