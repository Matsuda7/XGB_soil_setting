import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from module import collection, pipeline
from module.xgb_common import borehole_train_validation_test_split, build_preprocessor, prepare_feature_frame, build_regressor


class CollectionTests(unittest.TestCase):
    def test_import_is_hardlink_and_repeated_import_preserves_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.xml'
            target = root / 'raw/target.xml'
            source.write_text('<root/>')
            with patch.object(collection, 'ROOT', root):
                self.assertEqual(collection.import_file(source, target)['status'], 'imported')
                self.assertTrue(source.samefile(target))
                self.assertEqual(collection.import_file(source, target)['status'], 'existing')
                other = root / 'other.xml'
                other.write_text('<different/>')
                with self.assertRaises(ValueError):
                    collection.import_file(other, target)
                self.assertEqual(target.read_text(), '<root/>')


class ModelingTests(unittest.TestCase):
    def test_group_split_can_use_gamma_target_without_hole_leakage(self):
        frame = pd.DataFrame({'hole': np.repeat(np.arange(20), 3), 'gamma': np.arange(60)})
        parts = borehole_train_validation_test_split(frame, target_column='gamma', id_column='hole')
        groups = [set(part.hole) for part in parts]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(sum(map(len, parts)), len(frame))

    def test_preprocessor_handles_unseen_category_and_model_fits(self):
        train = pd.DataFrame({'x': [1., 2., np.nan, 4.], 'soil': ['sand', 'clay', 'sand', 'clay']})
        transform = build_preprocessor(['x'], ['soil'])
        x = transform.fit_transform(prepare_feature_frame(train, ['x'], ['soil']))
        model = build_regressor({'n_estimators': 2, 'max_depth': 1, 'n_jobs': 1})
        model.fit(x, [10., 20., 15., 30.])
        unseen = pd.DataFrame({'x': [np.nan], 'soil': ['unseen']})
        pred = model.predict(transform.transform(prepare_feature_frame(unseen, ['x'], ['soil'])))
        self.assertTrue(np.isfinite(pred).all())


class PipelineTests(unittest.TestCase):
    def test_collection_conflict_blocks_downstream_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'config.json'
            config.write_text('{}')
            with patch.object(pipeline, 'RESULTS', root), \
                 patch.object(pipeline, 'collect', side_effect=ValueError('conflicting source')), \
                 patch('module.preparation.prepare_phi') as preparation, \
                 patch('module.distributions.distribute') as distribution:
                self.assertEqual(pipeline.run(config_path=config), 2)
                preparation.assert_not_called()
                distribution.assert_not_called()
            report = json.loads((root / 'run_summary.json').read_text())
            self.assertEqual(report['stages']['distribute']['phi']['status'], 'blocked')

    def test_all_stages_in_order_and_shared_strength_extracted_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'config.json'
            config.write_text('{}')
            events = []
            def collect(*args):
                events.append('collect')
                return {'files': []}
            def prepare_phi():
                events.append('prepare_phi')
                return {'status': 'prepared'}
            def prepare_strength(targets):
                events.append('prepare_strength')
                return {t: {'status': 'prepared'} for t in targets}
            def distribute(target):
                events.append('distribute_' + target)
                return {'status': 'complete' if target == 'phi' else 'not_implemented'}
            with patch.object(pipeline, 'RESULTS', root / 'results'), \
                 patch.object(pipeline, 'collect', side_effect=collect), \
                 patch.object(pipeline, 'missing_inputs', return_value=[]), \
                 patch('module.preparation.prepare_phi', side_effect=prepare_phi), \
                 patch('module.preparation.prepare_strength', side_effect=prepare_strength) as strength, \
                 patch('module.distributions.distribute', side_effect=distribute):
                code = pipeline.run(config_path=config)
            self.assertEqual(code, 2)
            strength.assert_called_once_with(['c', 'gamma'])
            self.assertEqual(events, ['collect', 'prepare_phi', 'prepare_strength', 'distribute_c', 'distribute_phi', 'distribute_gamma'])
            report = json.loads((root / 'results/run_summary.json').read_text())
            self.assertEqual(report['status'], 'incomplete')

    def test_missing_grid_prevents_expensive_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'config.json'
            config.write_text('{}')
            with patch.object(pipeline, 'RESULTS', root), \
                 patch.object(pipeline, 'missing_inputs', return_value=['grid.txt']), \
                 patch('module.distributions.distribute') as distribution:
                self.assertEqual(pipeline.run('distribute', ('phi',), config), 2)
                distribution.assert_not_called()


if __name__ == '__main__':
    unittest.main()
