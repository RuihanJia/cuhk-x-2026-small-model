"""Dispatch frozen Selected solutions without fitting or prediction lookup."""
import argparse, subprocess, sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--data-dir',required=True)
p.add_argument('--output-csv',required=True)
p.add_argument('--submission-id',default='primary')
p.add_argument('--config',required=True)
a=p.parse_args()
record=yaml.safe_load((ROOT/'submission/submission_info.yaml').read_text(encoding='utf-8'))
config=yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
if a.submission_id=='primary':
 key=record['primary_solution_key']
else:
 matches=[r for r in record['selected_final_submissions'] if r['submission_id'] is not None and str(r['submission_id'])==a.submission_id]
 if len(matches)!=1: p.error('Unknown or not yet documented Selected numeric submission ID')
 key=matches[0]['model_checkpoint_config_mapping']['solution_key']
if key not in ('dual_mirror','weight_average'):p.error('Invalid frozen solution')
subprocess.run([sys.executable,'-B',str(ROOT/'src/solutions'/key/'code/run_inference.py'),'--data-dir',str(Path(a.data_dir).resolve()),'--output',str(Path(a.output_csv).resolve()),'--workers',str(config['workers']),'--micro',str(config['micro'])],check=True)
