"""Rerender Figure 18F and merge its QA rows into the round-two report."""
from pathlib import Path
import sys
import json
import pandas as pd
from PIL import Image

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
from analysis_round2.code.figure_fixes import Notebook10FigureFixes

output=root/"analysis_round2"
target="18F_training_subset_representativeness.png"
runner=Notebook10FigureFixes(root,output_root=output)
runner.training_subset()
report_path=output/"qa"/"_qa_report.csv"
existing=pd.read_csv(report_path,low_memory=False)
merged=pd.concat([existing[existing.figure.ne(target)],pd.DataFrame(runner.qa)],ignore_index=True,sort=False)
merged=merged.sort_values(["figure","check"]).reset_index(drop=True)
merged.to_csv(report_path,index=False,encoding="utf-8-sig")
sizes_path=output/"tables"/"figure_regeneration_sizes.csv"
sizes=pd.read_csv(sizes_path)
replacement=pd.DataFrame(runner.sizes)
sizes=pd.concat([sizes[sizes.figure.ne(target)],replacement],ignore_index=True,sort=False).sort_values("figure")
sizes.to_csv(sizes_path,index=False,encoding="utf-8-sig")
images=[]
for path in sorted((output/"figures").glob("*.png")):
    images.append(Image.open(path).convert("RGB"))
if images:
    images[0].save(output/"qa"/"_qa_contact_sheet.pdf",save_all=True,append_images=images[1:],resolution=150)
    for image in images:image.close()
failures=int(merged.status.eq("fail").sum())
audit_path=output/"manifests"/"isolation_audit.json"
audit=json.loads(audit_path.read_text(encoding="utf-8"))
audit["renderer_result"]["qa_failures"]=failures
audit["renderer_result"]["qa_rows"]=len(merged)
audit["final_targeted_rerender"]="18F_training_subset_representativeness.png"
audit_path.write_text(json.dumps(audit,indent=2),encoding="utf-8")
print({"figure":target,"qa_failures":failures,"qa_rows":len(merged)})
if failures:raise RuntimeError(f"Round-two QA still has {failures} failure(s)")
