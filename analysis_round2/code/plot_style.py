"""Shared print-size plotting conventions for Notebook 10 figure repairs."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

MODEL_ORDER=["qwen1b","hermes","bitnet2b","llama3b","mistral7b","qwen7b"]
MODEL_NAMES={
    "qwen1b":"Qwen2.5 1.5B","hermes":"OpenHermes 1.8B","bitnet2b":"BitNet b1.58 2B",
    "llama3b":"Llama 3.2 3B","mistral7b":"Mistral 7B v0.3","qwen7b":"Qwen2.5 7B",
}
MODEL_COLORS={
    "qwen1b":"#0072B2","hermes":"#E69F00","bitnet2b":"#009E73",
    "llama3b":"#D55E00","mistral7b":"#CC79A7","qwen7b":"#56B4E9",
}
CONDITION_ORDER=["untreated","technique","explanation","joint","full"]
CONDITION_NAMES={"untreated":"Untreated","technique":"SFT-technique","explanation":"SFT-explanation","joint":"SFT-joint","full":"SFT-full"}
CONDITION_MARKERS={"untreated":"o","technique":"^","explanation":"v","joint":"D","full":"s"}
FIELD_ORDER=[
    "score_technique_identification","score_prediction_evidence_support","score_explanation_grounding",
    "score_prediction_reason_alignment","score_confidence_evidence_support","score_instruction_adherence",
    "confidence_label_agreement",
]
FIELD_NAMES={
    "score_technique_identification":"Technique identification",
    "score_prediction_evidence_support":"Prediction-evidence support",
    "score_explanation_grounding":"Explanation grounding",
    "score_prediction_reason_alignment":"Prediction-reason alignment",
    "score_confidence_evidence_support":"Confidence-evidence support",
    "score_instruction_adherence":"Instruction adherence",
    "confidence_label_agreement":"Confidence-label agreement",
}
FIELD_SHORT={key:value.replace(" ","_").lower().replace("-","_") for key,value in FIELD_NAMES.items()}
FIELD_ABBR=dict(zip(FIELD_ORDER,["TI","PES","EG","PRA","CES","IA","CLA"]))
ORDINAL_COLORS=["#b2182b","#ef8a62","#fee08b","#67a9cf","#2166ac"]
BF16_COLOR="#0072B2";ONE_BIT_COLOR="#D55E00";TEACHER_COLOR="#7B3294";MODAL_COLOR="#E69F00";NULL_COLOR="#777777"
REFERENCE_NAMES={"gpt55_teacher":"GPT-5.5 teacher","baseline_modal":"Modal baseline (T1105)","baseline_null":"Null response"}

plt.rcParams.update({
    "font.size":8,"axes.titlesize":10,"axes.labelsize":8,"xtick.labelsize":8,"ytick.labelsize":8,
    "legend.fontsize":8,"figure.titlesize":11,"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
})

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def atomic_csv(frame:pd.DataFrame,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".partial")
    frame.to_csv(tmp,index=False,encoding="utf-8-sig");os.replace(tmp,path)

def display_model(value):return MODEL_NAMES.get(str(value),REFERENCE_NAMES.get(str(value),str(value)))
def display_condition(value):return CONDITION_NAMES.get(str(value),str(value))
def display_representation(value,legend=False):
    value=str(value)
    if value=="BF16 HF":return "BF16"
    if value=="one-bit frontier":return "One-bit (TQ1_0; BitNet: I2_S)" if legend else "One-bit"
    return value
def display_field(value):return FIELD_NAMES.get(str(value),str(value).replace("_"," ").title())

def rounded_limit(values,step=.5,minimum=.5):
    x=pd.to_numeric(pd.Series(values),errors="coerce").abs().max()
    return minimum if not np.isfinite(x) or x==0 else max(minimum,math.ceil(float(x)/step)*step)

def exact_zero_text(value):
    if pd.isna(value):return "n.e."
    value=float(value)
    if value==0:return "0"
    if 0<abs(value)<.005:return "<0.01" if value>0 else ">-0.01"
    return f"{value:.2f}"

def annotated_heatmap(ax,matrix,cmap="cividis",vmin=0,vmax=4,diverging=False,cbar=True,cbar_label="",blank_mask=None,cbar_ax=None):
    from matplotlib.patches import Rectangle
    values=np.asarray(matrix,dtype=float)
    blank=np.zeros(values.shape,dtype=bool) if blank_mask is None else np.asarray(blank_mask,dtype=bool)
    masked=np.ma.masked_where(blank|~np.isfinite(values),values)
    norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax) if diverging else None
    image=ax.imshow(masked,cmap=cmap,vmin=None if norm else vmin,vmax=None if norm else vmax,norm=norm,aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value=values[i,j]
            if blank[i,j]:
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor="white",edgecolor="white",lw=0));continue
            if not np.isfinite(value):
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor="white",edgecolor="#d0d0d0",lw=.4));label="n.e."
            elif value==0:
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor="#e0e0e0",edgecolor="#999999",lw=.4,hatch="///"));label="0"
            else:label=exact_zero_text(value)
            ax.text(j,i,label,ha="center",va="center",fontsize=8,color="black")
    if cbar:
        cb=ax.figure.colorbar(image,cax=cbar_ax,ax=None if cbar_ax is not None else ax,fraction=.045,pad=.03);cb.set_label(cbar_label,fontsize=8);cb.ax.tick_params(labelsize=8)
    return image

def figure_id(filename):
    match=re.match(r"^(S?\d+[A-Z]?(?:_\d+|_all)?)",Path(filename).stem)
    return match.group(1) if match else Path(filename).stem

def _content_groups(frame,columns):
    if not isinstance(frame,pd.DataFrame) or frame.empty:return set()
    use=[column for column in columns if column in frame.columns]
    if not use:return set()
    clean=frame[use].fillna("").astype(str).drop_duplicates()
    return {tuple(row) for row in clean.itertuples(index=False,name=None)}

def save_figure(root,filename,fig,data,sources,key,qa_records=None,dpi=300,
                stable_content_columns=None,allow_removed_groups=False,
                output_root=None,baseline_data_path=None):
    """Draw, QA, save figure/data and replace its manifest entry."""
    try:
        from .figure_qa import check
    except ImportError:
        from figure_qa import check
    root=Path(root);analysis=Path(output_root) if output_root else root/"analysis"
    figures=analysis/"figures";figure_data=analysis/"figure_data";manifests=analysis/"manifests"
    figures.mkdir(parents=True,exist_ok=True);figure_data.mkdir(parents=True,exist_ok=True);manifests.mkdir(parents=True,exist_ok=True)
    # Figure-level legends are deliberately below the axes.  Matplotlib's
    # constrained-layout engine does not reserve reliable space for them, so
    # switch to an explicit print-size margin before the QA canvas is drawn.
    if fig.legends:
        try: fig.set_layout_engine(None)
        except Exception: pass
        for legend in fig.legends:
            try: legend.set_bbox_to_anchor((.5,.005),transform=fig.transFigure)
            except Exception: pass
        reserved_bottom=getattr(fig,"_round2_legend_bottom",.27)
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom,reserved_bottom),top=min(fig.subplotpars.top,.94))
    # Grow margins from the measured text extents.  This is deterministic at
    # final print size and prevents long model/field labels from being clipped
    # after disabling constrained layout for a shared bottom legend.
    try:
        from .figure_qa import text_artists
    except ImportError:
        from figure_qa import text_artists
    for _ in range(3):
        fig.canvas.draw();renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height();boxes=[]
        for artist in text_artists(fig):
            try: boxes.append(artist.get_window_extent(renderer))
            except Exception: pass
        if not boxes: break
        minx=min(b.x0 for b in boxes);maxx=max(b.x1 for b in boxes);miny=min(b.y0 for b in boxes);maxy=max(b.y1 for b in boxes)
        left=fig.subplotpars.left+max(0,(-minx+3)/width);right=fig.subplotpars.right-max(0,(maxx-width+3)/width)
        bottom=fig.subplotpars.bottom+max(0,(-miny+3)/height);top=fig.subplotpars.top-max(0,(maxy-height+3)/height)
        if left==fig.subplotpars.left and right==fig.subplotpars.right and bottom==fig.subplotpars.bottom and top==fig.subplotpars.top: break
        if left<right-.15 and bottom<top-.15:
            try: fig.set_layout_engine(None)
            except Exception: pass
            fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top)
    fig.canvas.draw();problems=check(fig,filename)
    if qa_records is not None:
        categories={
            "overlap":"text_overlap","clipped":"clipping","tick label":"panel_tick_intrusion",
            "font <":"minimum_font","internal name":"display_names","suptitle":"duplicate_title",
            "colour bar":"colourbar_overlap","panel title":"panel_title_overlap",
            "empty visible":"empty_axes","duplicate y":"duplicate_ticks",
        }
        grouped={value:[] for value in categories.values()}
        grouped["other_render"]=[]
        for problem in problems:
            check_name=next((value for prefix,value in categories.items() if problem.startswith(prefix)),"other_render")
            grouped[check_name].append(problem)
        for check_name,messages in grouped.items():
            qa_records.append({"figure":filename,"check":check_name,"status":"fail" if messages else "pass","message":"; ".join(messages) if messages else ""})
    fid=figure_id(filename);data_path=figure_data/f"{fid}__{Path(filename).stem}.csv"
    plotted=data.copy() if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
    comparison=Path(baseline_data_path) if baseline_data_path else data_path
    previous=pd.read_csv(comparison,dtype=str,keep_default_na=False) if comparison.exists() else pd.DataFrame()
    columns=stable_content_columns or ["population","model","representation_group","representation","condition_short","condition","dimension","field","metric","k"]
    old_groups=_content_groups(previous,columns);new_groups=_content_groups(plotted,columns);missing=sorted(old_groups-new_groups)
    stable_pass=allow_removed_groups or not missing
    if qa_records is not None:
        qa_records.append({"figure":filename,"check":"stable_content","status":"pass" if stable_pass else "fail",
                           "message":"" if stable_pass else f"groups disappeared: {missing[:20]!r}"})
    atomic_csv(plotted,data_path)
    path=figures/filename;tmp=path.with_name(f".{path.stem}.{os.getpid()}.partial{path.suffix}")
    fig.savefig(tmp,dpi=dpi,bbox_inches=None,facecolor="white");os.replace(tmp,path);plt.close(fig)
    manifest_path=manifests/"figure_manifest.csv"
    manifest=pd.read_csv(manifest_path,dtype=str,keep_default_na=False) if manifest_path.exists() else pd.DataFrame()
    if len(manifest) and "figure_file" in manifest:manifest=manifest[manifest.figure_file.ne(filename)]
    addition=pd.DataFrame([{"figure_id":fid,"figure_file":filename,"figure_data_csv":str(data_path.relative_to(analysis)),"key":key,"source_tables":sources}])
    atomic_csv(pd.concat([manifest,addition],ignore_index=True,sort=False),manifest_path)
    return path,data_path,problems
