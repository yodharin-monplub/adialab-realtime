"""Build Submission28 from sealed Submission27 with incremental whitening."""
from pathlib import Path
import hashlib, json, shutil, tempfile

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'submissions/submission27'; DEST=ROOT/'submissions/submission28'
HERE=Path(__file__).resolve().parent; OUT=ROOT/'models/model87/deployment_validation'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value): path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    if DEST.exists(): raise FileExistsError('Submission28 already exists')
    seal=json.loads((SOURCE/'bundle_manifest.json').read_text())
    for name,record in seal['files'].items():
        path=SOURCE/name
        if path.is_symlink() or not path.resolve().is_relative_to(SOURCE.resolve()) or sha(path)!=record['sha256']:
            raise ValueError(f'Changed sealed source: {name}')
    stage=Path(tempfile.mkdtemp(prefix='submission28-staging-',dir=ROOT/'submissions'))
    for name in seal['files']:
        if name=='solution73.py': continue
        target=stage/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(SOURCE/name,target)
    shutil.copy2(HERE/'solution.py',stage/'solution73.py')
    shutil.copy2(HERE/'incremental_whitening.py',stage/'incremental_whitening.py')
    manifest=json.loads((stage/'resources/manifest.json').read_text())
    manifest['cloud_candidate']='model87_incremental_model73'
    manifest['runtime_repair']='O(1)-per-row exact incremental whitening; predictions unchanged'
    write(stage/'resources/manifest.json',manifest)
    (stage/'readme.txt').write_text('Submission 28: Model73 with exact incremental whitening. Inference only; forceFirstTrain=false.\n')
    stage.rename(DEST); OUT.mkdir(exist_ok=True)
    write(OUT/'build.json',{'source':'sealed_submission27','bundle':'submissions/submission28',
        'failed_run_id':116613,'failure':'timeout after 35572 seconds','uploaded':False,'cloud_run_started':False})
    print('Built isolated Submission28 incremental runtime candidate.')

if __name__=='__main__': main()
