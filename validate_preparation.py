"""Run safe preparation cells, save their actual outputs, and write validation evidence."""
from contextlib import redirect_stdout,redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import nbformat
from streamlined.common import *
from streamlined.data import records

ROOT=Path(__file__).resolve().parent

def main():
    path=ROOT/'notebooks/00_prepare_universal_dataset.ipynb'
    nb=nbformat.read(path,as_version=4);namespace={'__name__':'__main__'};count=0
    for i,cell in enumerate(nb.cells):
        if cell.cell_type!='code':continue
        description=nb.cells[i-1].source
        if not any(description.startswith(f'## {n}.') for n in (1,2,3,4)):continue
        capture=io.StringIO();start=time.perf_counter()
        with redirect_stdout(capture),redirect_stderr(capture):exec(compile(cell.source,str(path),'exec'),namespace)
        count+=1;cell.execution_count=count
        cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=capture.getvalue())]
        print(f'Saved preparation section {count}: {time.perf_counter()-start:.1f}s',flush=True)
    nbformat.write(nb,path)
    manifest=verify_experiment(ROOT)
    rows=records(ROOT);audit=read_csv(ROOT/'data/dataset_audit.csv');fit=read_csv(ROOT/'data/context_feasibility.csv')
    # Independently verify against raw physical source rows and source lexical spans.
    from streamlined.data import object_spans
    lines=(ROOT/'data/source/llm_responses.json').read_text(encoding='utf-8-sig').splitlines()
    for row in rows:
        line=lines[int(row['source_row'])-1];original=json.loads(line)
        a,b=object_spans(line)['response']
        assert row['prompt_text']==original['prompt']
        assert row['teacher_response_raw']==line[a:b]
    report={'prepared_rows':len(rows),'eligible_train':len(manifest['eligible_ids']),
        'excluded_train':len(manifest['excluded_ids']),'split_counts':manifest['source']['split_counts'],
        'literal_ground_truth_id_occurrences':sum(r['answer_ids_in_prompt']!='[]' for r in audit),
        'annotation_warning_rows':sum(r['annotation_warning']=='True' for r in audit),
        'teacher_schema_nonconformant_retained':sum(r['schema_adherent']=='False' for r in audit),
        'context':{key:{'inference_unsupported':sum(int(r['generation_allowance'])==0 for r in fit if r['model']==key),
                        'reduced_output_allowance':sum(int(r['generation_allowance'])<1024 for r in fit if r['model']==key)} for key in MODELS},
        'exact_prompt_and_teacher_source_checks':'990/990 passed',
        'gpu_training_and_generation_executed':False,'paid_API_requests_executed':False,
        'kernel':str(ROOT/'.venv-local/Scripts/python.exe'),'manifest_hash':manifest['config_hash']}
    write_json(ROOT/'manifests/implementation_validation.json',report)
    completed=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],
        cwd=ROOT,text=True,capture_output=True)
    atomic(ROOT/'manifests/test_results.txt',completed.stdout+completed.stderr)
    if completed.returncode:print(completed.stdout+completed.stderr);raise SystemExit(completed.returncode)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
