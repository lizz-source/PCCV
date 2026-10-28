"""Run the evaluated pipeline in a new working copy, preserving references."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,importlib.metadata,json,os,platform,shutil,subprocess,sys,time

ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['verify','reproduce','retrain'],default='verify')
    parser.add_argument('--output-dir',type=Path,help='New or empty working directory; defaults to runs/<mode>_<UTC time>.')
    args=parser.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    work=(args.output_dir or ROOT/'runs'/(args.mode+'_'+stamp)).resolve()
    if work==ROOT or ROOT.is_relative_to(work) or (work.is_relative_to(ROOT) and not work.is_relative_to(ROOT/'runs')):
        raise ValueError('Use runs/<new name> or a separate empty directory outside this repository.')
    if work.exists() and any(work.iterdir()):raise FileExistsError('The output directory is not empty: '+str(work))
    subprocess.run([sys.executable,str(ROOT/'scripts/check_release.py'),'--data-only'],cwd=ROOT,check=True)
    work.mkdir(parents=True,exist_ok=True)
    for name in ['src','inputs','results','study_holdout','replay']:
        shutil.copytree(ROOT/name,work/name,ignore=shutil.ignore_patterns('__pycache__','runtime_tmp','logs'),dirs_exist_ok=True)
    shutil.copy2(ROOT/'config.json',work/'config.json')
    tmp=work/'runtime_tmp';tmp.mkdir();logs=work/'logs';logs.mkdir()
    env=os.environ.copy();env.update(TEMP=str(tmp),TMP=str(tmp),PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(tmp/'mpl'),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    stages=[];started=time.perf_counter()
    def run(name,*extra):
        print('RUN',name,*extra,flush=True);before=time.perf_counter()
        log=logs/(str(len(stages)+1).zfill(2)+'_'+Path(name).stem+'.log')
        with log.open('w',encoding='utf8') as stream:
            result=subprocess.run([sys.executable,str(work/'src'/name),*extra],cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT)
        stages.append({'script':name,'arguments':list(extra),'exit_code':result.returncode,'seconds':time.perf_counter()-before})
        if result.returncode:
            print('\n'.join(log.read_text(encoding='utf8',errors='replace').splitlines()[-35:]),file=sys.stderr)
            raise RuntimeError('Stage failed: '+name+'; inspect '+str(log))
    if args.mode=='retrain':
        run('train_models.py','--force');run('metadata_audit.py')
        run('train_models.py','--study-holdout','--force');run('train_models.py','--replay','--force')
    if args.mode in ['reproduce','retrain']:
        for script in ['controlled_benchmark.py','cv_evaluation.py','larger_game_benchmark.py','poly_benchmark.py','quadratic_ablation.py','identifiability_demo.py','verify_core.py']:
            run(script)
        run('real_analysis.py','--stage','discover');run('real_analysis.py','--stage','evaluate','--force')
        run('real_analysis.py','--stage','multiseed');run('sensitivity_checks.py');run('holdout_explanations.py')
    else:run('verify_core.py')
    for script in ['verify_reproduction.py','summarize_and_plot.py','finalize.py']:run(script)
    checks=json.loads((work/'results/final_verification.json').read_text(encoding='utf8'))
    reproduction=json.loads((work/'results/independent_reproduction_summary.json').read_text(encoding='utf8'))
    core=json.loads((work/'results/core_verification.json').read_text(encoding='utf8'))
    versions={name:importlib.metadata.version(name) for name in ['torch','numpy','scipy','matplotlib','pandas','pyreadr','narwhals']}
    report={'all_pass':bool(checks['all_required_computational_checks_pass']),'mode':args.mode,
      'elapsed_seconds':time.perf_counter()-started,'python':platform.python_version(),'platform':platform.system(),
      'dependencies':versions,'real_predictors_retrained':args.mode=='retrain',
      'reference_artifacts_overwritten':False,'model_reload_and_case_checks':reproduction['checks'],
      'core_checks':core,'final_checks':checks,'stages':stages,
      'timing_note':'Fresh hardware-dependent timings are execution checks, not a replacement for the archived publication benchmark.'}
    (work/'run_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    assert report['all_pass']
    print('PASS: '+str(reproduction['checks'])+' model reload / real-case checks. Report: '+str(work/'run_summary.json'),flush=True)
if __name__=='__main__':main()
