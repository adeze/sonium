"""Hardware selection, full target coverage, and failure reporting contracts."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
import bench_machine as bench


class MachineBenchTests(unittest.TestCase):
    def test_workspace_plan_covers_cpu_and_cuda_without_platform_stubs(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources = {}
            for name, source in [('cpu', 'criterion_main!(benches);'),
                                 ('gpu', '#[cfg(feature = "cuda")] fn gpu() {}\ncriterion_main!(benches);'),
                                 ('fft_vs_accelerate', 'criterion_main!(benches);')]:
                path = Path(tmp) / (name + '.rs')
                path.write_text(source)
                sources[name] = dict(name=name, src_path=str(path), kind=['bench'])
            metadata = dict(workspace_members=['member'], packages=[dict(
                id='member', name='package', features={'cuda': [], 'bench-openblas': []},
                targets=list(sources.values()))])
            hardware = dict(system='Linux', machine='x86_64', cuda=True, openblas=True)
            runs, skipped = bench.plan(metadata, hardware, 'cargo', quick=True)
            self.assertEqual([(r['backend'], r['bench']) for r in runs],
                             [('cpu', 'cpu'), ('cpu', 'gpu'), ('cuda', 'gpu')])
            self.assertEqual(len(skipped), 1)
            self.assertIn('bench-openblas,cuda', runs[-1]['command'])
            self.assertIn('--quick', runs[0]['command'])
            hardware['cuda'] = False
            runs, _ = bench.plan(metadata, hardware, 'cargo')
            self.assertEqual(len(runs), 2)
            self.assertTrue(all(r['backend'] == 'cpu' for r in runs))

    def test_cuda_probe_requires_live_device_and_vendor_libraries(self):
        driver = Mock()
        driver.cuInit.return_value = 0
        def count(pointer):
            pointer._obj.value = 1
            return 0
        driver.cuDeviceGetCount.side_effect = count
        with patch.object(bench, 'library', return_value=driver):
            self.assertTrue(bench.capabilities()['cuda'])
        driver.cuInit.return_value = 100
        with patch.object(bench, 'library', return_value=driver):
            capabilities = bench.capabilities()
            self.assertFalse(capabilities['cuda'])
            self.assertIn('100', capabilities['cuda_reason'])
        driver.cuInit.return_value = 0
        def load(name):
            if name == 'cusolver':
                raise OSError('missing cuSOLVER')
            return driver
        with patch.object(bench, 'library', side_effect=load):
            self.assertFalse(bench.capabilities()['cuda'])

    def test_failure_does_not_prevent_remaining_benchmarks(self):
        runs = [dict(backend=backend, package='package', bench='bench', command=['cargo', 'bench'])
                for backend in ['cpu', 'cuda']]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(bench.subprocess, 'run', side_effect=[
                    subprocess.CompletedProcess([], 1), subprocess.CompletedProcess([], 0)]) as execute:
            status = bench.execute(runs, Path(tmp), {}, {}, [])
            self.assertEqual(status, 1)
            self.assertEqual(execute.call_count, 2)
            summary = json.loads((Path(tmp) / 'summary.json').read_text())
            self.assertEqual([r['returncode'] for r in summary['runs']], [1, 0])
            self.assertTrue(all(Path(r['log']).exists() for r in summary['runs']))


if __name__ == '__main__':
    unittest.main()
