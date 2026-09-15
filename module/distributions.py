"""Stage three dispatch for phi and gamma; cohesion is pending."""
from module.paths import RESULTS


def distribute(target):
    if target == 'phi':
        from module import phi_training, phi_prediction
        phi_training.main()
        phi_prediction.main([])
        return {'status': 'complete', 'output': str(RESULTS / 'phi'),
                'method': 'XGBoost SPT N prediction followed by existing phi conversion'}
    if target == 'gamma':
        from module.gamma import distribute as distribute_gamma
        return distribute_gamma()
    return {'status': 'not_implemented',
            'reason': f'{target}: 実装待ち（cは変換方法の検討のため一時保留）'}
