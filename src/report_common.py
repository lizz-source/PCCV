"""Small reporting runtime: plotting and packaging do not need PyTorch loaded."""
from pathlib import Path
import sys,os,json,csv,hashlib
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'inputs'
sys.path.insert(0,str(ROOT/'vendor'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'runtime_tmp/mpl'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np

def dump(name,obj):
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()),encoding='utf8')

def table(name,rows):
    rows=list(rows)
    if not rows:return
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):value.update(chunk)
    return value.hexdigest()
