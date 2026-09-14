"""Stage three dispatch; c and gamma spatial models are the next development step."""
from module.paths import RESULTS


def distribute(target):
    if target == 'phi':
        from module import phi_training, phi_prediction
        phi_training.main()
        phi_prediction.main([])
        return {'status': 'complete', 'output': str(RESULTS / 'phi'),
                'method': 'XGBoost SPT N prediction followed by existing phi conversion'}
    return {'status': 'not_implemented',
            'reason': f'{target} spatial training/prediction will be implemented after framework integration.'}
