"""ASCII uses all 95 characters, with native call/cache accounting and quality."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from kilix_bench import ascii, decomposition, fonts
from kilix_bench.report import assert_no_local_paths


class QualityTests(unittest.TestCase):
    def test_character_confusion_and_firm_errors_are_reported(self):
        target=np.zeros((3,4,2),dtype=np.float32)
        target[1,0,0]=1;target[2,2,1]=1
        actual=target[[0,2,1]]
        result=decomposition.quality(actual,target,[32,65,66])
        self.assertEqual(result['nearest_ok'],1)
        self.assertEqual(result['firm_wrong_pixels'],4)
        self.assertEqual(result['per_glyph'][1]['nearest_scalar'],66)
        self.assertEqual(result['exact_glyphs'],1)
        for broken in (np.full_like(actual,np.nan),actual+1):
            with self.assertRaises(ValueError):decomposition.quality(broken,target,[32,65,66])


class NativeAsciiTests(unittest.TestCase):
    def test_cli_uses_complete_ascii_batches_and_rejects_changed_assets(self):
        home=os.environ.get('NEURONTE_HOME')
        if not home or not shutil.which('cc') or not shutil.which('pkg-config'):
            self.skipTest('NEURONTE_HOME, compiler and FreeType required')
        if subprocess.run(['pkg-config','--exists','freetype2']).returncode:
            self.skipTest('FreeType headers required')
        with tempfile.TemporaryDirectory(prefix='ascii-cli-') as temp:
            root=Path(temp);weights=root/'control.nrt'
            shutil.copyfile(Path(home)/'weights/neuronte-split-v5.nrt',weights)
            truth=root/'truth.npy';np.save(truth,np.ones((95,4,2),dtype=np.float32))
            manifest={'schema':'kilix-bench.ascii-experiment/v1','corpus':list(range(32,127)),
                      'models':{'control':{'file':weights.name,'sha256':fonts.digest(weights)}},
                      'targets':{'2x4':{'file':truth.name,'sha256':fonts.digest(truth)}},
                      'teacher':{'fixture':'deliberately wrong'},'runtime_revision':'fixture',
                      'cases':[{'name':'control','kind':'bundle','members':['control'],'routes':[0]*95}]}
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            env=dict(os.environ,KILIX_BENCH_HOME=str(root/'bench'),KILIX_HOME='',KILIX_STATE_LIBRARY='',
                     NEURONTE_LIBRARY=str(Path(home)/'build/libneuronte.so'),NEURONTE_WEIGHTS=str(weights))
            command=[sys.executable,'-m','kilix_bench','--suite','neuronte','--ascii-experiment',str(path),
                     '--cells','2x4','--repeats','1','--warmup','0','--expert-passes','2']
            subprocess.run(command,env=env,check=True,capture_output=True,text=True,timeout=60)
            result=json.loads((root/'bench/results/kilix-ascii-latest.json').read_text())
            self.assertEqual(len(result['corpus']),95)
            self.assertEqual(len(result['trials']),5)
            self.assertLessEqual(result['reference'][0]['max_abs_difference'],1e-3)
            self.assertFalse(result['quality'][0]['all_binary_exact'])
            for row in result['trials']:
                mode=row['case']
                if mode in ('coverage','draw-warm','draw-cold'):
                    calls=95 if mode=='draw-cold' else 190
                    self.assertEqual(row['calls'],calls)
                    self.assertEqual(row['coverage_calls'],calls if mode=='coverage' else 0)
                    self.assertEqual(row['draw_calls'],0 if mode=='coverage' else calls)
                    self.assertEqual(row['priming_calls'],0 if mode=='draw-cold' else 95)
            assert_no_local_paths(result,(str(Path.home()),temp))
            # Disjoint Unicode ranges use manifest positions, not scalar-minus-first.
            codes=[32,65,0x391,0x410,0x2588]
            routes=[0,0,3,3,4]
            np.save(truth,np.ones((len(codes),4,2),dtype=np.float32))
            general={**manifest,'schema':'kilix-bench.expert-experiment/v1','corpus':codes,
                     'targets':{'2x4':{'file':truth.name,'sha256':fonts.digest(truth)}},
                     'cases':[{**manifest['cases'][0],'routes':routes}]}
            path.write_text(json.dumps(general))
            general_command=[arg if arg!='--ascii-experiment' else '--expert-experiment' for arg in command]
            subprocess.run(general_command,env=env,check=True,capture_output=True,text=True,timeout=60)
            general_result=json.loads((root/'bench/results/kilix-expert-latest.json').read_text())
            self.assertEqual(set(general_result['corpus']),set(codes))
            self.assertEqual(general_result['quality'][0]['glyphs'],5)
            self.assertEqual(general_result['reference'][0]['scalars'],codes)
            timed_bytes=(root/'bench/results/kilix-expert-latest.json').read_bytes()
            subprocess.run(general_command+['--experiment-quality-only'],env=env,check=True,
                           capture_output=True,text=True,timeout=60)
            q=json.loads((root/'bench/results/kilix-expert-quality-latest.json').read_text())
            self.assertTrue(q['quality_only'])
            self.assertEqual(q['trials'],[])
            self.assertEqual(q['scenarios'],[])
            self.assertEqual(q['repeats'],0)
            self.assertEqual(q['quality'],general_result['quality'])
            self.assertEqual(timed_bytes,(root/'bench/results/kilix-expert-latest.json').read_bytes())
            for broken in ([32,32],[65,32],[True],[0xd800]):
                path.write_text(json.dumps({**general,'corpus':broken}))
                with self.assertRaisesRegex(ValueError,'ordered, distinct'):decomposition.read_experiment(path,family='expert')
            np.save(truth,np.ones((95,4,2),dtype=np.float32))
            for changed in ({**manifest,'corpus':list(range(32,126))},
                            {**manifest,'schema':'kilix-bench.braille-experiment/v1'}):
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'complete ascii'):ascii.read_experiment(path)
            path.write_text(json.dumps(manifest))
            weights.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'SHA256'):ascii.read_experiment(path)


if __name__=='__main__':unittest.main()
