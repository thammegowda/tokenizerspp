"""Unit tests for tools/nlcodec_to_tokenizer_json.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "nlcodec_to_tokenizer_json.py"

SAMPLE_NLCODEC = """\
#{"scheme": "bpe", "total": 8}
0\t<pad>\t0\t0
1\t<unk>\t0\t0
2\t<s>\t0\t0
3\t</s>\t0\t0
4\t▁\t1\t100
5\th\t1\t50
6\ti\t1\t40
7\thi▁\t2\t30\t5 6
"""


class NlcodecConverterTest(unittest.TestCase):
    def run_tool(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(TOOL), *args],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_converts_valid_bpe_model_to_file_and_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "vocab.model"
            out_path = Path(tmp) / "tokenizer.json"
            model_path.write_text(SAMPLE_NLCODEC, encoding="utf-8")

            res = self.run_tool(str(model_path), str(out_path))
            self.assertEqual(res.returncode, 0, res.stderr)
            doc = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(doc["model"]["type"], "WordPiece")
            self.assertFalse(doc["model"]["fuse_unk"])
            self.assertEqual(doc["model"]["unk_token"], "<unk>")
            self.assertEqual(doc["model"]["vocab"]["hi▁"], 7)

            res_stdout = self.run_tool(str(model_path), "--pretty")
            self.assertEqual(res_stdout.returncode, 0, res_stdout.stderr)
            self.assertEqual(json.loads(res_stdout.stdout), doc)

    def test_duplicate_token_spellings_emit_decoder_alias_rules(self) -> None:
        dup_model = SAMPLE_NLCODEC.replace('"total": 8', '"total": 9') + "8\thi▁\t2\t10\t5 6\n"
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "dup.model"
            model_path.write_text(dup_model, encoding="utf-8")
            res = self.run_tool(str(model_path))
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertIn("duplicate token spellings", res.stderr)
            doc = json.loads(res.stdout)
            self.assertEqual(len(doc["model"]["vocab"]), 9)
            self.assertEqual(doc["model"]["vocab"]["hi▁"], 7)
            decoders = doc["decoder"]["decoders"]
            self.assertEqual(len(decoders), 2)
            self.assertEqual(decoders[0]["type"], "Replace")
            self.assertEqual(decoders[0]["content"], "hi▁")

    def test_rejects_invalid_models_and_in_place_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            valid = base / "valid.model"
            valid.write_text(SAMPLE_NLCODEC, encoding="utf-8")
            same = self.run_tool(str(valid), str(valid))
            self.assertEqual(same.returncode, 2)

            bad_cases = {
                "no_header": "0\t<unk>\t0\t1\n",
                "bad_scheme": '#{"scheme": "char", "total": 2}\n0\t<unk>\t0\t1\n1\t▁\t1\t1\n',
                "bad_count": '#{"scheme": "bpe", "total": 99}\n0\t<unk>\t0\t1\n1\t▁\t1\t1\n',
                "bad_child": '#{"scheme": "bpe", "total": 2}\n0\t<unk>\t0\t1\n1\t▁\t1\t1\t1\n',
            }
            for name, content in bad_cases.items():
                with self.subTest(case=name):
                    path = base / f"{name}.model"
                    path.write_text(content, encoding="utf-8")
                    res = self.run_tool(str(path))
                    self.assertEqual(res.returncode, 2)


if __name__ == "__main__":
    unittest.main()
