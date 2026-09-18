"""Bundle completed Colab diagnostics and follow-up outputs, without model weights."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

if not Path('/content').is_dir() or 'COLAB_RELEASE_TAG' not in os.environ:
    raise RuntimeError('Export this experiment from Colab only.')

root = Path('/content/covt-pilot')
runs = root / 'runs'
folders = [Path(json.loads((runs / name).read_text())['run_dir'])
           for name in ('latest-diagnostic.json', 'latest-followup.json')]
assert (folders[1] / 'controls/summary.json').is_file(), 'Controls are incomplete.'
files = sorted({p for folder in folders for p in folder.rglob('*') if p.is_file()})
files += [p for pattern in ('scripts/*.py', 'src/covt_pilot/*.py', 'tests/*.py')
          for p in root.glob(pattern)]
files += [p for p in (runs / 'cache-diagnostic-console.log', runs / 'followup-console.log',
                       root / 'pyproject.toml') if p.is_file()]
manifest = []
for kind in ('reports', 'full'):
    target = Path('/content') / f'covt-followup-{kind}-20260918.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in files:
            if kind == 'reports' and path.suffix in ('.npz', '.npy'):
                continue
            archive.write(path, path.relative_to(root))
    digest = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
    manifest.append({'kind': kind, 'path': str(target), 'bytes': target.stat().st_size,
                     'sha256': digest})
Path('/content/covt-followup-archives.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2), flush=True)
