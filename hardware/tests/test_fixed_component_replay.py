import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from hardware.baselines.common import ROOT


class FixedComponentReplayTest(unittest.TestCase):
    def test_replays_profile_update_without_reference_tables(self):
        archive = ROOT.parent / "results/component-profile-update/evaluation.json"
        expected = json.loads(archive.read_text())
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            shutil.copytree(ROOT, work / "hardware",
                            ignore=shutil.ignore_patterns("reference", "results", "tests", "__pycache__"))
            result = subprocess.run(
                [sys.executable, "-E", "-B", "-m", "hardware.component_study",
                 "--fixed", "--out", "evaluation.json"],
                cwd=work, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((work / "evaluation.json").read_text()), expected)
