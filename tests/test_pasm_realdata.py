import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import pasm_realdata
from pasm_physionet import PhysioNetRecord


def fake_record(record_id, has_truth=True):
    truth = pd.DataFrame(
        [{"start_s": 1.0, "end_s": 5.0, "type": "af_like"}] if has_truth else []
    )
    return PhysioNetRecord(
        record_id=record_id,
        fs=250.0,
        signal=np.zeros(2500),
        rpeaks=np.arange(100, 2400, 250),
        truth_episodes=truth,
    )


class PASMRealDataTest(unittest.TestCase):
    def test_run_realdata_preset_and_report_with_mocked_loaders(self):
        metrics = pd.DataFrame(
            [
                {
                    "record_id": "afdb/04015",
                    "model": "pasm_physionet",
                    "type": "af_like",
                    "tp": 1,
                    "fp": 0,
                    "fn": 0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "mean_iou": 0.8,
                },
                {
                    "record_id": "afdb/04015",
                    "model": "pasm_physionet",
                    "type": "macro",
                    "tp": 1,
                    "fp": 0,
                    "fn": 0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "mean_iou": np.nan,
                },
                {
                    "record_id": "afdb/04015",
                    "model": "pasm_physionet",
                    "type": "false_alarms_per_hour",
                    "tp": 1,
                    "fp": 0,
                    "fn": 0,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": 0.0,
                    "mean_iou": np.nan,
                },
            ]
        )

        with mock.patch.object(pasm_realdata, "load_afdb_record", return_value=fake_record("afdb/04015")):
            with mock.patch.object(pasm_realdata, "evaluate_physionet_records", return_value=metrics):
                result = pasm_realdata.run_realdata_preset("afdb-smoke")

        self.assertEqual(result["preset"], "afdb-smoke")
        self.assertEqual(len(result["informative_records"]), 2)
        self.assertEqual(len(result["inventory"]), 2)
        self.assertEqual(result["inventory"]["truth_episodes"].tolist(), [1, 1])
        self.assertIn("pasm_physionet", set(result["summary"]["model"]))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.md"
            html_out = Path(tmp) / "report.html"
            diagnostic_out = Path(tmp) / "diagnostic.html"
            pasm_realdata.write_realdata_report(out, result)
            pasm_realdata.write_realdata_html_report(html_out, result, markdown_report=out)
            pasm_realdata.write_diagnostic_html_report(
                diagnostic_out,
                {
                    "record": fake_record("afdb/04015"),
                    "diagnostics": pd.DataFrame(
                        [
                            {"status": "TP", "start_s": 1.0, "end_s": 5.0, "type": "af_like", "iou": 0.8},
                            {"status": "FP", "start_s": 7.0, "end_s": 9.0, "type": "af_like", "iou": 0.0},
                        ]
                    ),
                    "predicted": pd.DataFrame([{"start_s": 1.0, "end_s": 5.0, "type": "af_like"}]),
                    "truth": pd.DataFrame([{"start_s": 1.0, "end_s": 5.0, "type": "af_like"}]),
                    "iou_threshold": 0.30,
                },
            )
            text = out.read_text(encoding="utf-8")
            html = html_out.read_text(encoding="utf-8")
            diagnostic_html = diagnostic_out.read_text(encoding="utf-8")

        self.assertIn("PASM-Rhythm Real-Data Summary", text)
        self.assertIn("Loaded Record Inventory", text)
        self.assertIn("afdb/04015", text)
        self.assertIn("<!doctype html>", html)
        self.assertIn("Real-Data Report: afdb-smoke", html)
        self.assertIn("Episode F1", html)
        self.assertIn("report.md", html)
        self.assertIn("Ectopy flood strong morph z", html)
        self.assertIn("Record Diagnostic: afdb/04015", diagnostic_html)
        self.assertIn("Timeline", diagnostic_html)
        self.assertIn("TP", diagnostic_html)

    def test_list_presets_includes_mini_presets(self):
        presets = pasm_realdata.list_presets()

        self.assertIn("mini", set(presets["preset"]))
        self.assertIn("afdb-mini", set(presets["preset"]))
        self.assertIn("mitdb-mini", set(presets["preset"]))


if __name__ == "__main__":
    unittest.main()
