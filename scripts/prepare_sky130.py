"""Prepare the bundled AnalogGym SKY130 simulation models without editing upstream."""
import hashlib
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import dump, read, rel

ARCHIVE = ROOT / 'third_party/AnalogGym/RGNN_RL/mosfet_model/sky130_pdk.zip'
EXPECTED_SHA256 = '662a378b3290d6b5d90c5ad0a2c55be56bb8e82b86f5b8f0cee91009bcfb2a09'
DEST = ROOT / 'tools/pdk/sky130_pdk'
MANIFEST = ROOT / 'tools/pdk/sky130_bundle_manifest.json'


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def write_environment():
    dump(ROOT / 'config/simulation_environment.json', {
        'ngspice_executable': 'tools/Spice64/bin/ngspice_con.exe',
        'pdk_root': rel(DEST), 'pdk_manifest': rel(MANIFEST),
        'library': rel(DEST / 'libs.tech/ngspice/sky130.lib.spice'),
        'default_corner': 'tt', 'model_length_input_unit': 'um',
        'model_scale_evidence': rel(DEST / 'libs.tech/ngspice/all.spice'),
        'note': 'No global PATH, PDK_ROOT, or upstream files modified'})


def main():
    from src.audit_sources import verify_source
    verify_source('AnalogGym')
    if sha(ARCHIVE) != EXPECTED_SHA256:
        raise ValueError('Unexpected upstream model archive; review before extracting')
    if DEST.exists():
        manifest = read(MANIFEST)
        assert manifest['archive_sha256'] == EXPECTED_SHA256
        for path, expected in manifest['files_sha256'].items():
            assert sha(ROOT / path) == expected, path
        write_environment()
        print('Existing SKY130 models verified:', len(manifest['files_sha256']), 'files')
        return
    temp_parent = ROOT / '.tmp'; temp_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='sky130_', dir=temp_parent) as temp:
        stage = Path(temp).resolve()
        assert stage.is_relative_to(ROOT)
        files = {}; expanded = 0
        with zipfile.ZipFile(ARCHIVE) as zf:
            infos = zf.infolist()
            if sum(i.file_size for i in infos) > 150_000_000:
                raise ValueError('Archive exceeds reviewed extraction budget')
            for info in infos:
                parts = PurePosixPath(info.filename).parts
                if not parts or parts[0] != 'sky130_pdk' or '..' in parts or ':' in info.filename or '\\' in info.filename:
                    raise ValueError('Unexpected archive path: ' + info.filename)
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError('Archive symlinks are not supported')
                target = stage.joinpath(*parts).resolve()
                if not target.is_relative_to(stage): raise ValueError('Archive path escapes staging')
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True); continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                final_path = DEST.parent.joinpath(*parts)
                files[rel(final_path)] = sha(target)
                expanded += info.file_size
        # Check both absolute paths before moving the complete tree on Windows.
        source = (stage / 'sky130_pdk').resolve()
        destination = DEST.resolve()
        if not source.is_relative_to(ROOT) or not destination.is_relative_to(ROOT):
            raise ValueError('Model move would leave workspace')
        DEST.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
    dump(MANIFEST, {'archive': rel(ARCHIVE), 'archive_sha256': EXPECTED_SHA256,
                    'expanded_bytes': expanded, 'files_sha256': files,
                    'scope': 'Bundled simulation models, not a certified complete layout/signoff PDK'})
    write_environment()
    print('Prepared', len(files), 'model files;', expanded, 'bytes at', rel(DEST))


if __name__ == '__main__': main()
