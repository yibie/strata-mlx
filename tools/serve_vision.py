"""Run the external official HTTP server with full schema validation, without its language-model engine."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys


def preflight(strata_root):
    root = Path(strata_root).resolve()
    if not (root / 'serve/server.py').is_file():
        raise ValueError('missing official serve/server.py: use --strata-root')
    sys.path[:0] = [str(root), str(root / 'tools')]
    # No model or GPU imports: missing server dependencies fail before starting the encoder/engine.
    import jinja2  # noqa: F401
    import regex  # noqa: F401
    from PIL import Image  # noqa: F401
    from referencing import Registry  # noqa: F401
    from jsonschema import validators  # noqa: F401
    from strata_tokenizer import Tokenizer  # noqa: F401
    from serve.structured import StructuredOutputError, prepare_format, validated_json

    schema = {'type': 'object', 'properties': {'count': {'type': 'integer', 'enum': [2]}},
              'required': ['count'], 'additionalProperties': False}
    _, validator = prepare_format({'type': 'json_schema', 'json_schema': {
        'name': 'startup_contract', 'strict': True, 'schema': schema}}, [])
    if type(validator).__name__ == '_ObjectOnly':
        raise ValueError('object-only schema fallback is forbidden; install jsonschema and referencing')
    if json.loads(validated_json('{"count":2}', validator, 'stop')) != {'count': 2}:
        raise ValueError('schema validation self-test failed')
    cases = ['{}', '{"count":true}', '{"count":3}', '{"count":2,"extra":1}',
             '{"count":2,"count":2}', '{"count":NaN}']
    for raw, finish in [(raw, 'stop') for raw in cases] + [('{"count":2}', 'length')]:
        try:
            validated_json(raw, validator, finish)
        except StructuredOutputError:
            continue
        raise ValueError('schema validation self-test accepted an invalid response')
    return {'schema_validation': 'required', 'jsonschema': importlib.metadata.version('jsonschema'),
            'rejected_cases': len(cases) + 1, 'model_loaded': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strata-root', required=True, type=Path)
    parser.add_argument('--preflight-only', action='store_true', help='Check dependencies without loading any model')
    args, forwarded = parser.parse_known_args(argv)
    try:
        info = preflight(args.strata_root)
    except (ImportError, OSError, ValueError, AssertionError) as error:
        print('Vision server refused: ' + str(error), file=sys.stderr)
        return 78
    if args.preflight_only:
        print(json.dumps(info))
        return 0
    sys.argv = ['serve.server', *forwarded]
    runpy.run_module('serve.server', run_name='__main__')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
