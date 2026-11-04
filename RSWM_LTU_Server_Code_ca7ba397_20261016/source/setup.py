"""Build provenance for installed CLIs, without modifying the source checkout."""
import hashlib
import json
from pathlib import Path
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


def archive_inputs(root):
    """Hash authored inputs selected by the current MANIFEST.in contract.

    Generated egg-info and _build_info.json are deliberately excluded: neither
    is an original source input, and hashing our own receipt would be recursive.
    Paths remain relative so an extracted sdist has the same input identity.
    """
    paths={root/name for name in ('README.md','NOTICE.md','pyproject.toml','setup.py','MANIFEST.in')}
    for directory,pattern in (('src/rswm_inria','*.py'),('tests','*.py'),('scripts','*.py'),('docs','*.md')):
        paths.update((root/directory).rglob(pattern))
    paths.update((root/'src/rswm_inria/configs').rglob('*.json'))
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file() and not p.name.startswith('.env.') and p.name!='.env'}


def provenance(root):
    root=Path(root)
    package=root/'src/rswm_inria'
    files={p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(package.rglob('*.py'))}
    source_hash=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    inputs=archive_inputs(root)
    try:
        # An extracted source archive can live inside an unrelated checkout.
        # That parent repository cannot attest this archive's source identity.
        git_root=subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=root,
                                        stderr=subprocess.DEVNULL,text=True).strip()
        if Path(git_root).resolve()!=root.resolve():
            raise OSError('Source root is not the Git checkout root')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,stderr=subprocess.DEVNULL,text=True).strip()
        # The sdist also publishes README, manifest, tests, scripts and documentation.
        # Changes there must not claim the committed archive is clean merely
        # because the installable package's Python fingerprint is unchanged.
        dirty=bool(subprocess.check_output(['git','status','--porcelain','--',
                   'src','pyproject.toml','setup.py','MANIFEST.in','README.md','NOTICE.md','tests','scripts','docs'],
                   cwd=root,text=True).strip())
    except (OSError,subprocess.CalledProcessError):
        previous=package/'_build_info.json'
        info=json.loads(previous.read_text()) if previous.exists() else {}
        commit=info.get('source_commit')
        prior_inputs=info.get('source_archive_inputs')
        # A matching package fingerprint alone cannot attest README/docs/tests.
        # Old receipts lacking the archive manifest therefore remain unknown.
        if (info.get('source_dirty') is True or
                (info.get('source_code_sha256') is not None and info['source_code_sha256']!=source_hash) or
                (isinstance(prior_inputs,dict) and prior_inputs!=inputs)):
            dirty=True
        elif (isinstance(prior_inputs,dict) and prior_inputs==inputs and
              info.get('source_code_sha256')==source_hash and info.get('source_dirty') is False):
            dirty=False
        else:
            dirty=None
    return {'schema':'rswm-build-info-v1','source_commit':commit,'source_code_sha256':source_hash,
            'source_archive_inputs':inputs,
            'source_dirty':dirty,'scope':'Build provenance; no experimental acceptance or source authenticity claim.'}


class BuildWithProvenance(build_py):
    def run(self):
        super().run()
        root=Path(__file__).parent
        info=provenance(root)
        dest=Path(self.build_lib)/'rswm_inria/_build_info.json'
        dest.write_text(json.dumps(info,sort_keys=True,indent=2)+'\n')


class SourceWithProvenance(sdist):
    def make_release_tree(self,base_dir,files):
        super().make_release_tree(base_dir,files)
        target=Path(base_dir)/'src/rswm_inria/_build_info.json'
        target.write_text(json.dumps(provenance(Path(__file__).parent),sort_keys=True,indent=2)+'\n')


setup(cmdclass={'build_py':BuildWithProvenance,'sdist':SourceWithProvenance})
