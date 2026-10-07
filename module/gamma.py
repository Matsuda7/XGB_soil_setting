"""Run the selected gamma model in a unique directory and record explicit completion."""
import json
from pathlib import Path
from module.output_layout import create_target_directory
from module.paths import project_path
from module.gamma_config import load_config, selected_kinds
from module.gamma_training import train_both
from module.gamma_prediction import predict_grid


def distribute(config=None,max_chunks=None):
    config=config or load_config()
    kind = selected_kinds(config)[0]
    run = create_target_directory('gamma_' + kind)
    config['output_dir'] = str(run)
    (run/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    status={'status':'running','gamma_kind':kind,'run':run.name,'prediction_depth_m':config['prediction_depth_m']}
    try:
        from module.paths import TRAINING
        from module.gamma_config import data_signature
        prepared = json.loads((TRAINING/'gamma/preparation_summary.json').read_text())
        if prepared['signature'] != data_signature(config):
            raise ValueError('Gamma preparation configuration changed; rerun the prepare stage')
        comparison = None
        if config['n_comparison']['enabled']:
            from module.n_comparison import run_comparison
            comparison = run_comparison(config, run/'n_comparison')
            status['n_comparison'] = comparison
        if config['create_distribution']:
            n_artifact = None
            if config['distribution_model'] == 'predicted_n':
                if comparison is None or comparison['status'] != 'complete':
                    raise ValueError('Cannot deploy predicted_n without completed comparison evaluations')
                from module.gamma_predicted_n import train_selected
                models, n_artifact = train_selected(config, run)
            else:
                models=train_both(config,run/'models')
            prediction=predict_grid(config,models,run/'grid',max_chunks=max_chunks,n_artifact=n_artifact)
            status.update(prediction)
            status['distribution_model'] = config['distribution_model']
        else:
            status.update(status='complete', distribution='not_requested')
        if comparison is not None and comparison['status'] != 'complete':
            status['status'] = 'incomplete'
    except Exception as error:
        status.update(status='failed',error=str(error))
        (run/'run_summary.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
        raise
    (run/'run_summary.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    return {**status,'output':str(run)}
