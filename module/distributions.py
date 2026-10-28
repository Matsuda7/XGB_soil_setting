"""Stage three dispatch for phi, gamma and effective-cohesion Model A."""
from module.output_layout import create_target_directory


def distribute(target, gamma_kind=None):
    if target == 'phi':
        from module import phi_training, phi_prediction
        import json
        output = create_target_directory('phi')
        config = phi_prediction.load_prediction_config(phi_prediction.DEFAULT_CONFIG, output)
        from module.config_snapshot import write_config_snapshot
        write_config_snapshot(config, output, 'phi')
        status = {'status': 'running', 'output': str(output)}
        try:
            phi_training.main(output / 'model')
            phi_prediction.main([], output_dir=output)
            status['status'] = 'complete'
        except Exception as error:
            status.update(status='failed', error=str(error))
            raise
        finally:
            (output / 'run_summary.json').write_text(json.dumps(status, indent=2) + '\n')
        return {'status': 'complete', 'output': str(output),
                'method': 'XGBoost SPT N prediction followed by existing phi conversion'}
    if target == 'gamma':
        from module.gamma import distribute as distribute_gamma
        from module.gamma_config import load_config
        config = load_config()
        if gamma_kind is not None:
            config['gamma_kind'] = gamma_kind
        return distribute_gamma(config)
    if target == 'c':
        from module.c_model_a import distribute as distribute_c
        return distribute_c()
    return {'status': 'not_implemented',
            'reason': f'{target}: 実装待ち（cは変換方法の検討のため一時保留）'}
