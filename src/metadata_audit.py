from common import *
import pyreadr,collections
df=pyreadr.read_r(DATA/'cmd_sampleMetadata_current.rda')['sampleMetadata']
def clean(v):
    ss=str(v)
    return '' if ss in ['nan','None','<NA>','NA'] else ss
lookup={}
for row in df.to_dict('records'):lookup.setdefault(str(row['sample_id']),[]).append(row)
with (DATA/'species_abundance.csv').open(encoding='utf-8-sig',newline='') as f:sa=next(csv.reader(f))[1:]
with (ROOT/'results/sample_split.csv').open(encoding='utf-8-sig',newline='') as f:split={r['sample_id']:r['split'] for r in csv.DictReader(f)}
rows=[]
for sample in sa:
    hit=lookup.get(sample,[]);keys={(clean(h['study_name']),clean(h['subject_id'])) for h in hit}
    h=hit[0] if hit else {};r={'sample_id':sample,'match_count':len(hit),'unambiguous_study_subject':int(len(keys)==1)}
    for col in ['study_name','subject_id','body_site','study_condition','disease','age','age_category','country','NCBI_accession','PMID']:r[col]=clean(h.get(col,''))
    r['subject_key']=r['study_name']+'::'+r['subject_id'] if r['subject_id'] else r['study_name']+'::unknown::'+sample
    r['random_sample_split']=split[sample];rows.append(r)
table('results/sample_metadata_verified.csv',rows)
studies=collections.Counter(r['study_name'] for r in rows if r['unambiguous_study_subject']);subjects=collections.defaultdict(list)
for r in rows:
    if r['unambiguous_study_subject']:subjects[r['subject_key']].append(r)
overlap=[]
for key,rr in subjects.items():
    labels=set(r['random_sample_split'] for r in rr)
    if len(labels)>1 and rr[0]['subject_id']:overlap.append({'subject_key':key,'n_samples':len(rr),'splits':';'.join(sorted(labels))})
table('results/subject_split_overlap.csv',overlap)
out={'metadata_source':'https://raw.githubusercontent.com/waldronlab/curatedMetagenomicData/devel/data/sampleMetadata.rda','retrieved':'2026-10-27','metadata_sha256':sha(DATA/'cmd_sampleMetadata_current.rda'),'metadata_rows':len(df),'frozen_samples':len(sa),'matched_samples':sum(r['match_count']>0 for r in rows),'unambiguous_study_subject_samples':sum(r['unambiguous_study_subject'] for r in rows),'n_studies':len(studies),'study_counts':dict(studies),'body_sites':dict(collections.Counter(r['body_site'] for r in rows)),'study_conditions':dict(collections.Counter(r['study_condition'] for r in rows)),'subjects_with_multiple_samples':sum(len(v)>1 for v in subjects.values()),'subjects_crossing_random_split':len(overlap),'missing_subject_ids':sum(not r['subject_id'] for r in rows),'note':'Current official metadata used for post hoc ID verification; it does not reconstruct the historical filtering operation.'}
dump('results/metadata_audit.json',out);print(out,flush=True)
