#!/usr/bin/env python3
"""Fetch only allowlisted small CPU assets. No weights, models, inference or APIs."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from toolrl_pipeline.io import file_hash, write_json


def git_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def save_verified(path, raw, size, blob):
    if len(raw) != size or git_blob(raw) != blob:
        raise ValueError(f"Asset length/blob mismatch: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"name": path.name, "bytes": len(raw), "git_blob_sha1": blob, "sha256": file_hash(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bfcl', action='store_true')
    parser.add_argument('--qwen-tokenizer', action='store_true')
    parser.add_argument('--hf-endpoint', choices=['https://huggingface.co', 'https://hf-mirror.com'], default='https://huggingface.co')
    args = parser.parse_args()
    if not (args.bfcl or args.qwen_tokenizer):
        parser.error('Select an asset; no implicit downloads')
    if args.bfcl:
        lock = json.loads((ROOT/'configs/bfcl_snapshot.json').read_text())
        dest = ROOT/'data_local/bfcl'/lock['commit']

        def fetch(entry):
            path = dest/Path(entry['path']).name
            if path.exists() and len(path.read_bytes()) == entry['size'] and git_blob(path.read_bytes()) == entry['sha']:
                return save_verified(path, path.read_bytes(), entry['size'], entry['sha'])
            for attempt in range(3):
                try:
                    completed = subprocess.run(['gh', 'api', f"repos/{lock['repository']}/git/blobs/{entry['sha']}"],
                                               check=True, capture_output=True, timeout=55)
                    payload = json.loads(completed.stdout)
                    raw = base64.b64decode(payload['content'])
                    return save_verified(path, raw, entry['size'], entry['sha'])
                except (subprocess.SubprocessError, ValueError, KeyError):
                    if attempt == 2:
                        raise
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(fetch, lock['files']))
        write_json(dest/'assets.json', {'repository': lock['repository'], 'commit': lock['commit'], 'files': results})
        print(f'BFCL: {len(results)} verified files, {sum(r["bytes"] for r in results)} bytes at {dest}')
    if args.qwen_tokenizer:
        lock = json.loads((ROOT/'configs/qwen_tokenizer_snapshot.json').read_text())
        dest = ROOT/'data_local/tokenizers/Qwen2.5-1.5B-Instruct'/lock['revision']
        results = []
        for entry in lock['files']:
            path = dest/entry['rfilename']
            if path.exists() and git_blob(path.read_bytes()) == entry['blobId']:
                raw = path.read_bytes()
            else:
                url = f"{args.hf_endpoint}/{lock['model_id']}/raw/{lock['revision']}/{entry['rfilename']}"
                for attempt in range(3):
                    try:
                        result = subprocess.run(['curl', '-L', '--fail', '--silent', '--show-error', '--max-time', '180', '--connect-timeout', '20',
                                                 '--max-filesize', str(entry['size']+1), url],
                                                capture_output=True, check=True, timeout=185)
                        raw = result.stdout
                        break
                    except (OSError, subprocess.SubprocessError):
                        if attempt == 2:
                            raise
            results.append(save_verified(path, raw, entry['size'], entry['blobId']))
        write_json(dest/'assets.json', {**lock, 'download_endpoint': args.hf_endpoint, 'verified_files': results})
        print(f'Tokenizer: {len(results)} verified files, {sum(r["bytes"] for r in results)} bytes at {dest}')


if __name__ == '__main__':
    main()
