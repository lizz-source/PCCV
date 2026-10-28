from pathlib import Path,PurePosixPath
import argparse,ast,csv,hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def check_manifest(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as stream:records=list(csv.DictReader(stream))
    for row in records:
        rel=PurePosixPath(row['file'])
        assert not rel.is_absolute() and '..' not in rel.parts and ':' not in row['file']
        path=ROOT.joinpath(*rel.parts);assert path.is_file(),row['file']
        assert path.stat().st_size==int(row['bytes']),row['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],row['file']
    return len(records)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-only',action='store_true');args=parser.parse_args()
    names=['DATA_MANIFEST.csv'] if args.data_only else ['DATA_MANIFEST.csv','SOURCE_MANIFEST.csv','RELEASE_MANIFEST.csv']
    counts={name:check_manifest(name) for name in names}
    if not args.data_only:
        for path in sorted((ROOT/'src').glob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+[ROOT/'run_pipeline.py',ROOT/'publication/publication_figures.py',ROOT/'publication/verify_display_values.py']:
            ast.parse(path.read_text(encoding='utf8'),filename=path.name)
        assert len(list((ROOT/'results').glob('controlled_*_seed*.pth')))==40
        assert len(list((ROOT/'results').glob('larger_*_seed*.npz')))==40
    print(json.dumps({'all_pass':True,'manifests':counts,'data_only':args.data_only}))
if __name__=='__main__':main()
