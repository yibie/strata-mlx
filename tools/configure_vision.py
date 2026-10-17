"""Create a loopback vision configuration from existing dependencies; never download or load weights."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def file_path(path, label, executable=False):
    # Keep venv/bin/python symlinks: resolving them would discard the virtual environment.
    path = Path(os.path.abspath(Path(path).expanduser()))
    if not path.is_file() or not os.access(path, os.R_OK):
        raise ValueError(f'{label} is missing/unreadable: {path}')
    if executable and not os.access(path, os.X_OK):
        raise ValueError(f'{label} is not executable: {path}')
    return path


def configure(args):
    official = args.strata_root.expanduser().resolve()
    file_path(official / 'serve/server.py', 'official HTTP server')
    engine_python = file_path(args.engine_python or ROOT / '.venv/bin/python', 'MLX Python (run uv sync)', True)
    server_python = file_path(args.server_python, 'server Python', True)
    encoder = file_path(args.encoder or official / 'build-vision/bin/strata-vision', 'standalone vision encoder', True)
    model = file_path(args.model, 'text GGUF shard 1')
    projector = file_path(args.mmproj, 'matching BF16 projector')
    tokenizer = args.tokenizer.expanduser().resolve()
    for name in ('vocab.json', 'merges.txt', 'token_type.json', 'chat_template.jinja'):
        file_path(tokenizer / name, 'exported tokenizer ' + name)
    mtp = args.mtp.expanduser().resolve() if args.mtp else None
    if mtp:
        file_path(mtp / 'mtp-manifest.json', 'optional raw MTP manifest')
    if not 1 <= args.port <= 65535 or args.max_context < 1:
        raise ValueError('port must be 1..65535 and max-context must be positive')
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError('refusing to overwrite existing output: ' + str(output))
    # This imports only CPU-side server/schema/tokenizer modules. No engine, encoder or model starts.
    checked = subprocess.run([str(server_python), str(ROOT / 'tools/serve_vision.py'),
                              '--strata-root', str(official), '--preflight-only'],
                             check=True, text=True, capture_output=True, timeout=30)
    preflight = json.loads(checked.stdout)
    engine_args = ['--native', str(model), '--vision', '--max-context', str(args.max_context)]
    if mtp:
        engine_args += ['--mtp', str(mtp)]
    config = {'exe': str(output / 'engine.sh'), 'cwd': str(ROOT), 'args': engine_args,
              'tokenizer': str(tokenizer), 'model_name': 'qwen3.8-flash-next', 'log': None,
              'api_monitor': False, 'lazy_load': False, 'idle_unload_s': 0,
              'vision': {'exe': str(encoder), 'model': str(model), 'mmproj': str(projector),
                         'gpu': True, 'max_tokens': 1024}}
    q = shlex.quote
    engine = '#!/bin/sh\nset -eu\nPYTHONPATH=' + q(str(ROOT)) + ' exec ' + q(str(engine_python)) + ' -m strata_mlx.serve "$@"\n'
    command = [str(server_python), str(ROOT / 'tools/serve_vision.py'), '--strata-root', str(official),
               '--engine', 'strata', '--config', str(output / 'config.json'), '--host', '127.0.0.1', '--port', str(args.port)]
    launch = '#!/bin/sh\nset -eu\nexec ' + shlex.join(command) + ' "$@"\n'
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        for name, source in [('engine.sh', engine), ('serve.sh', launch)]:
            path = output / name
            path.write_text(source)
            path.chmod(0o755)
        (output / 'preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
    except BaseException:
        shutil.rmtree(output)
        raise
    return {'configuration': str(output / 'config.json'), 'start': str(output / 'serve.sh'),
            'url': f'http://127.0.0.1:{args.port}/v1', 'model_loaded': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('strata-root', 'server-python', 'model', 'mmproj', 'tokenizer', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    for name in ('engine-python', 'encoder', 'mtp'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--max-context', type=int, default=16384)
    parser.add_argument('--port', type=int, default=8095)
    args = parser.parse_args(argv)
    try:
        result = configure(args)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        detail = getattr(error, 'stderr', None) or str(error)
        parser.exit(2, 'Vision setup refused: ' + detail.strip() + '\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
