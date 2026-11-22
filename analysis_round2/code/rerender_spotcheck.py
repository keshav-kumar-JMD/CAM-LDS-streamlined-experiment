"""Rerender visual spot-check corrections and merge their QA records."""
from pathlib import Path
import json
import sys
import pandas as pd
from PIL import Image

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
from analysis_round2.code.figure_fixes import Notebook10FigureFixes

output=root/"analysis_round2"
targets={"18F_training_subset_representativeness.png","19A_all_reference_ladder_all_fields.png"}
runner=Notebook10FigureFixes(root,output_root=output)
runner.training_subset();runner.reference_ladder_all()
report_path=output/"qa"/"_qa_report.csv"
existing=pd.read_csv(report_path,low_memory=False)
merged=pd.concat([existing[~existing.figure.isin(targets)],pd.DataFrame(runner.qa)],ignore_index=True,sort=False).sort_values(["figure","check"]).reset_index(drop=True)
merged.to_csv(report_path,index=False,encoding="utf-8-sig")
sizes_path=output/"tables"/"figure_regeneration_sizes.csv";sizes=pd.read_csv(sizes_path);replacement=pd.DataFrame(runner.sizes)
pd.concat([sizes[~sizes.figure.isin(targets)],replacement],ignore_index=True,sort=False).sort_values("figure").to_csv(sizes_path,index=False,encoding="utf-8-sig")
images=[Image.open(path).convert("RGB") for path in sorted((output/"figures").glob("*.png"))]
if images:
    images[0].save(output/"qa"/"_qa_contact_sheet.pdf",save_all=True,append_images=images[1:],resolution=150)
    for image in images:image.close()
failures=int(merged.status.eq("fail").sum());audit_path=output/"manifests"/"isolation_audit.json";audit=json.loads(audit_path.read_text(encoding="utf-8"));audit["renderer_result"]["qa_failures"]=failures;audit["renderer_result"]["qa_rows"]=len(merged);audit["final_targeted_rerender"]=sorted(targets);audit_path.write_text(json.dumps(audit,indent=2),encoding="utf-8")
print({"figures":sorted(targets),"qa_failures":failures,"qa_rows":len(merged)})
if failures:raise RuntimeError(f"Round-two QA still has {failures} failure(s)")
