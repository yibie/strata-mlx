"""Portable configuration checks use no MLX, model weights, network or external encoder."""
import argparse
import importlib.util
import json
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


setup = module('configure_vision')
server = module('serve_vision')


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="vision's setup ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        official = self.root / 'official checkout'
        (official / 'serve').mkdir(parents=True)
        (official / 'serve/server.py').write_text('raise RuntimeError("must never start during setup")')
        encoder = official / 'build-vision/bin/strata-vision'
        encoder.parent.mkdir(parents=True)
        encoder.write_text('#!/bin/sh\nexit 99\n'); encoder.chmod(0o755)
        model = self.root / 'text.gguf'; model.touch()
        mmproj = self.root / 'mmproj.gguf'; mmproj.touch()
        tokenizer = self.root / 'tokenizer'; tokenizer.mkdir()
        for name in ('vocab.json', 'merges.txt', 'token_type.json', 'chat_template.jinja'):
            (tokenizer / name).touch()
        self.args = argparse.Namespace(strata_root=official, server_python=Path(sys.executable),
            engine_python=Path(sys.executable), encoder=None, model=model, mmproj=mmproj,
            tokenizer=tokenizer, output=self.root / "new profile's folder", mtp=None,
            max_context=16384, port=8095)
        self.checked = subprocess.CompletedProcess([], 0, '{"schema_validation":"required","model_loaded":false}', '')

    def test_creates_portable_quoted_launchers_without_starting_model(self):
        with patch.object(setup.subprocess, 'run', return_value=self.checked) as checked:
            result = setup.configure(self.args)
        self.assertEqual(checked.call_count, 1)
        self.assertIn('--preflight-only', checked.call_args.args[0])
        self.assertFalse(result['model_loaded'])
        config = json.loads((self.args.output / 'config.json').read_text())
        self.assertIn('--vision', config['args'])
        self.assertNotIn('--resident-read', config['args'])
        self.assertNotIn('--fusion-limit', config['args'])
        self.assertIsNone(config['log'])
        self.assertFalse(config['api_monitor'])
        for name in ('serve.sh', 'engine.sh'):
            subprocess.run(['/bin/sh', '-n', str(self.args.output / name)], check=True)
        launch = (self.args.output / 'serve.sh').read_text()
        self.assertIn('127.0.0.1', launch)
        self.assertIn("'\"'\"'", launch)  # apostrophes are shell-quoted, never interpolated

    def test_preserves_virtual_environment_python_symlink(self):
        python = self.root / 'virtual environment/bin/python'
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        self.args.server_python = self.args.engine_python = python
        with patch.object(setup.subprocess, 'run', return_value=self.checked) as checked:
            setup.configure(self.args)
        self.assertEqual(checked.call_args.args[0][0], str(python))
        self.assertEqual(shlex.split((self.args.output / 'engine.sh').read_text().splitlines()[-1])[2], str(python))

    def test_refuses_overwrite_before_preflight(self):
        self.args.output.mkdir(); (self.args.output / 'keep').write_text('original')
        with patch.object(setup.subprocess, 'run') as checked, self.assertRaisesRegex(ValueError, 'overwrite'):
            setup.configure(self.args)
        checked.assert_not_called()
        self.assertEqual((self.args.output / 'keep').read_text(), 'original')

    def test_missing_encoder_fails_without_partial_output(self):
        (self.args.strata_root / 'build-vision/bin/strata-vision').unlink()
        with self.assertRaisesRegex(ValueError, 'vision encoder'):
            setup.configure(self.args)
        self.assertFalse(self.args.output.exists())

    def test_missing_chat_template_is_rejected(self):
        (self.args.tokenizer / 'chat_template.jinja').unlink()
        with self.assertRaisesRegex(ValueError, 'chat_template'):
            setup.configure(self.args)

    def test_dependency_failure_leaves_no_configuration(self):
        with patch.object(setup.subprocess, 'run', side_effect=subprocess.CalledProcessError(78, [], stderr='missing jsonschema')):
            with self.assertRaises(subprocess.CalledProcessError):
                setup.configure(self.args)
        self.assertFalse(self.args.output.exists())

    def test_write_failure_removes_only_new_output(self):
        original = Path.write_text
        def write(path, *args, **kwargs):
            if path.name == 'serve.sh':
                raise OSError('simulated full disk')
            return original(path, *args, **kwargs)
        with patch.object(setup.subprocess, 'run', return_value=self.checked), patch.object(Path, 'write_text', write):
            with self.assertRaises(OSError):
                setup.configure(self.args)
        self.assertFalse(self.args.output.exists())
        self.assertTrue(self.args.model.exists())

    def test_optional_raw_mtp_manifest_and_custom_port(self):
        self.args.mtp = self.root / 'mtp'; self.args.mtp.mkdir()
        (self.args.mtp / 'mtp-manifest.json').touch(); self.args.port = 18095
        with patch.object(setup.subprocess, 'run', return_value=self.checked):
            result = setup.configure(self.args)
        config = json.loads((self.args.output / 'config.json').read_text())
        self.assertIn('--mtp', config['args'])
        self.assertEqual(result['url'], 'http://127.0.0.1:18095/v1')

    def test_bad_port_is_refused(self):
        self.args.port = 0
        with self.assertRaisesRegex(ValueError, 'port'):
            setup.configure(self.args)

    def test_preflight_rejects_missing_external_checkout(self):
        with self.assertRaisesRegex(ValueError, 'serve/server.py'):
            server.preflight(self.root / 'absent')

    def test_preflight_rejects_object_only_schema_fallback(self):
        fallback = type('_ObjectOnly', (), {})()
        structured = types.ModuleType('serve.structured')
        structured.StructuredOutputError = ValueError
        structured.prepare_format = lambda *_: (None, fallback)
        structured.validated_json = lambda *_: self.fail('fallback must fail before validation')
        fake = {'jinja2': types.SimpleNamespace(), 'regex': types.SimpleNamespace(),
                'PIL': types.SimpleNamespace(Image=object), 'referencing': types.SimpleNamespace(Registry=object),
                'jsonschema': types.SimpleNamespace(validators=object),
                'strata_tokenizer': types.SimpleNamespace(Tokenizer=object), 'serve.structured': structured}
        with patch.dict(sys.modules, fake), patch.object(sys, 'path', sys.path.copy()):
            with self.assertRaisesRegex(ValueError, 'fallback'):
                server.preflight(self.args.strata_root)


if __name__ == '__main__':
    unittest.main()
