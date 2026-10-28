from pathlib import Path
import csv,hashlib
ROOT=Path(__file__).resolve().parents[1]
SKIP={'runs','vendor','.venv','__pycache__','runtime_tmp','logs','.git','figure_previews','tmp'}
MANIFESTS={'DATA_MANIFEST.csv','SOURCE_MANIFEST.csv','RELEASE_MANIFEST.csv'}
def files():
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if not path.is_file() or set(rel.parts)&SKIP or rel.parts[0]=='figures':continue
        if rel.name in MANIFESTS or path.suffix in {'.pyc','.log','.zip'}:continue
        yield path
def data(path):
    rel=path.relative_to(ROOT).as_posix()
    return rel.startswith(('inputs/','results/','study_holdout/results/','replay/results/','publication/figure_data/','publication/case_study/')) or rel=='input_manifest.csv'
def write(name,paths):
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['file','bytes','sha256'])
        for path in paths:writer.writerow([path.relative_to(ROOT).as_posix(),path.stat().st_size,hashlib.sha256(path.read_bytes()).hexdigest()])
all_files=list(files())
write('DATA_MANIFEST.csv',[p for p in all_files if data(p)])
write('SOURCE_MANIFEST.csv',[p for p in all_files if not data(p)])
write('RELEASE_MANIFEST.csv',all_files+[ROOT/'DATA_MANIFEST.csv',ROOT/'SOURCE_MANIFEST.csv'])
print('Release manifests built for',len(all_files),'files.')
