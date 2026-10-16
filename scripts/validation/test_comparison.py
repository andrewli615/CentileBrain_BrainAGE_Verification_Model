import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.shared.data import age_mask, join_exports, read_export, catalog, METHODS
from scripts.corrections.training_buckets import assign_buckets, fit_training, apply_training
from scripts.corrections.compare_methods import correct_all, fit_references
from scripts.statistics.summary_tables import statistics, summarize


class ComparisonTests(unittest.TestCase):
    def test_five_and_ten_year_boundaries(self):
        ages=pd.Series([5,9.999,10,14.999,15,35,40,40.001,45,50,90])
        for width,expected in [(5,[0,0,1,1,2,6,6]),(10,[0,0,0,0,1,3,3])]:
            self.assertEqual(assign_buckets(ages,'5-40',width).dropna().tolist(),expected)
        self.assertEqual(assign_buckets(ages,'40-90',5).dropna().tolist(),[0,1,2,9])
        self.assertFalse(age_mask(pd.Series([40.]),'40-90').iloc[0])

    def test_training_subtraction_both_widths_and_extrapolation(self):
        ref=pd.DataFrame({'chronological_age':[6.,7.,8.],'Raw_BrainAGE':[3.,5.,7.]})
        obs=pd.DataFrame({'chronological_age':[5.,7.,9.,16.],'Raw_BrainAGE':[10.,10.,10.,10.]})
        for width in [5,10]:
            fit=fit_training(ref,'5-40',width)
            out=apply_training(obs,fit,'5-40',width)
            np.testing.assert_allclose(out.BrainAGE_TrainingBucketAdjusted[:3],[9,5,1])
            self.assertTrue(out.Outside_Training_Age_Support.iloc[0])
            self.assertTrue(out.Outside_Training_Age_Support.iloc[2])
            self.assertTrue(pd.isna(out.BrainAGE_TrainingBucketAdjusted.iloc[3]))

    def test_exact_identity_join_preserves_leading_zeroes(self):
        raw=pd.DataFrame({'SITE':['01','02'],'SubjectID':['001','002'],'chronological_age':[6.,8.],'MR_predicted_age':[8.,9.],'Raw_BrainAGE':[2.,1.]})
        adjusted=pd.DataFrame({'SITE':['02','01'],'SubjectID':['002','001'],'chronological_age':[8.,6.],'CentileBrain_Adjusted':[3.,4.]})
        merged=join_exports(raw,adjusted)
        self.assertEqual(merged.SubjectID.tolist(),['001','002'])
        self.assertEqual(merged.CentileBrain_Adjusted.tolist(),[4.,3.])
        adjusted.loc[1,'chronological_age']=9
        with self.assertRaisesRegex(ValueError,'age mismatch'):
            join_exports(raw,adjusted)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            join_exports(pd.concat([raw,raw]),adjusted)

    def test_missing_export_retains_rows(self):
        raw=pd.DataFrame({'SITE':['01'],'SubjectID':['001'],'chronological_age':[6.],'Raw_BrainAGE':[2.]})
        adjusted=pd.DataFrame(columns=['SITE','SubjectID','chronological_age','CentileBrain_Adjusted'])
        result=join_exports(raw,adjusted)
        self.assertEqual(len(result),1)
        self.assertTrue(result.CentileBrain_Adjusted.isna().all())

    def test_methods_keep_populations_before_intersection(self):
        ref=pd.DataFrame({'SITE':['x']*4,'SubjectID':['1','2','3','4'],'chronological_age':[6.,7.,8.,9.],
                          'Raw_BrainAGE':[1.,2.,3.,4.],'CentileBrain_Adjusted':[1.,1.,2.,1.]})
        fits=fit_references({('Female','5-40'):ref})
        obs=ref.copy()
        obs.loc[3,'Raw_BrainAGE']=np.nan
        result,_=correct_all(obs,'Female','5-40',fits)
        self.assertEqual(result.Matched_All_Methods.tolist(),[True,True,True,False])
        self.assertEqual(result.BrainAGE_S.notna().sum(),4)
        self.assertAlmostEqual(result.BrainAGE_S.mean(),obs.CentileBrain_Adjusted.mean())
        grouped={('self_chinese','Chinese','Female','5-40'):result}
        tables=summarize(grouped)
        self.assertEqual(tables['overview'].N.iloc[0],3)
        self.assertEqual(tables['matched_statistics'].N.tolist(),[3]*5)
        self.assertEqual(tables['available_statistics'].N.tolist(),[3,4,3,3,4])

    def test_mae_is_participant_weighted_signed_mean_separate(self):
        d=pd.DataFrame({'chronological_age':[6,7,8,16],'value':[-1,-1,1,9]})
        result=statistics(d,'value')
        self.assertEqual(result['MAE'],3)
        self.assertEqual(result['BrainAGE_Mean'],2)
        self.assertNotEqual(result['MAE'],(1+9)/2)
        empty=statistics(d.iloc[:0],'value')
        self.assertEqual(empty['N'],0)
        self.assertIsNone(empty['MAE'])

    def test_unfit_reference_missing_age_and_nonfinite_values(self):
        ref=pd.DataFrame({'chronological_age':[6.,7.],'Raw_BrainAGE':[1.,2.]})
        fits=fit_training(ref,'5-40',5)
        obs=pd.DataFrame({'chronological_age':[6.,np.nan,91.],'Raw_BrainAGE':[1.,2.,np.inf]})
        result=apply_training(obs,fits,'5-40',5)
        self.assertEqual(result.Correction_Status.tolist(),['skipped','excluded','excluded'])
        self.assertTrue(result.BrainAGE_TrainingBucketAdjusted.isna().all())

    def test_catalog_refuses_changed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'source.csv').write_text('age\n5\n')
            from scripts.shared.data import digest
            entries=[dict(Path='source.csv',SHA256=digest(root/'source.csv'))]
            (root/'source_catalog.json').write_text(json.dumps(entries))
            with patch('scripts.shared.data.ROOT',root),patch('scripts.shared.data.PROVENANCE',root):
                self.assertEqual(catalog(),entries)
                (root/'source.csv').write_text('age\n6\n')
                with self.assertRaisesRegex(ValueError,'Pinned source changed'):
                    catalog()


if __name__=='__main__':
    unittest.main()
