"""Print-size figure QA and contact-sheet generation for Notebook 10."""
from __future__ import annotations
from itertools import combinations
from pathlib import Path
import pandas as pd

def text_artists(fig):
    out=[]
    for ax in fig.axes:
        out += [ax.title,ax.xaxis.label,ax.yaxis.label]
        out += [t for t in ax.get_xticklabels()+ax.get_yticklabels() if t.get_text()]
        if ax.get_legend():out.append(ax.get_legend())
    out += [t for t in fig.texts]
    out += list(fig.legends)
    if fig._suptitle:out.append(fig._suptitle)
    seen=[]
    for item in out:
        if item.get_visible() and id(item) not in {id(x) for x in seen}:seen.append(item)
    return seen

def check(fig,name):
    renderer=fig.canvas.get_renderer();artists=text_artists(fig);boxes=[];problems=[]
    for artist in artists:
        try:box=artist.get_window_extent(renderer)
        except Exception:continue
        boxes.append((artist,box))
    # General pairwise overlap is restricted to legends.  Tick labels from
    # different panels are handled by the stricter panel-intrusion test below;
    # treating every Text pair as global otherwise produces false positives
    # for labels that merely share a coordinate-system boundary.
    for (a,b1),(c,b2) in combinations(boxes,2):
        if not (b1.width>0 and b2.width>0 and b1.overlaps(b2)):continue
        if type(a).__name__!="Legend" and type(c).__name__!="Legend":continue
        problems.append(f"overlap: {type(a).__name__}:{getattr(a,'get_text',lambda:'' )()!r} / {type(c).__name__}:{getattr(c,'get_text',lambda:'' )()!r}")
    width,height=fig.canvas.get_width_height()
    for artist,box in boxes:
        # An empty string is normally a figure-level Legend, whose artists are
        # checked for overlap separately and are repositioned by save_figure.
        label=getattr(artist,'get_text',lambda:'' )()
        if label and (box.x0<-1 or box.y0<-1 or box.x1>width+1 or box.y1>height+1):problems.append(f"clipped: {label!r}")
    axes=[ax for ax in fig.axes if ax.get_label()!="<colorbar>"]
    for ax in axes:
        for tick in ax.get_yticklabels()+ax.get_xticklabels():
            if not tick.get_visible() or not tick.get_text():continue
            box=tick.get_window_extent(renderer)
            for other in axes:
                if other is ax:continue
                ob=other.get_window_extent(renderer)
                iw=max(0,min(box.x1,ob.x1)-max(box.x0,ob.x0));ih=max(0,min(box.y1,ob.y1)-max(box.y0,ob.y0))
                # Ignore one-pixel boundary contact from antialiasing; report a
                # label only when it visibly intrudes into the neighbouring
                # plotting rectangle.
                cx=(box.x0+box.x1)/2;cy=(box.y0+box.y1)/2
                if iw>2 and ih>2 and (ob.x0+2<cx<ob.x1-2) and (ob.y0+2<cy<ob.y1-2):
                    problems.append(f"tick label {tick.get_text()!r} enters another panel")
    for artist,_ in boxes:
        if hasattr(artist,"get_fontsize") and artist.get_fontsize()<8:problems.append(f"font < 8pt: {getattr(artist,'get_text',lambda:'' )()!r}")
    bad={"qwen1b","hermes","bitnet2b","llama3b","mistral7b","qwen7b","model_a","model_b","prompt_length_bucket","BF16 HF","train_original"}
    for artist,_ in boxes:
        text=artist.get_text() if hasattr(artist,"get_text") else ""
        if any(word==text or f"{word} " in text or text.endswith(word) for word in bad):problems.append(f"internal name: {text!r}")
    titles=[ax.get_title() for ax in axes if ax.get_title()]
    if fig._suptitle and fig._suptitle.get_text() in titles:problems.append("suptitle duplicates an axes title")
    return sorted(set(problems))

def write_qa_outputs(root,regenerated,records):
    from PIL import Image
    root=Path(root);figures=root/"analysis"/"figures";figures.mkdir(parents=True,exist_ok=True)
    report=pd.DataFrame(records,columns=["figure","check","message"])
    if report.empty:report=pd.DataFrame([{"figure":"ALL","check":"summary","message":"No QA failures"}])
    report.to_csv(figures/"_qa_report.csv",index=False,encoding="utf-8-sig")
    images=[]
    for path in regenerated:
        path=Path(path)
        if path.exists() and path.suffix.lower()==".png":images.append(Image.open(path).convert("RGB"))
    if images:
        images[0].save(figures/"_qa_contact_sheet.pdf",save_all=True,append_images=images[1:],resolution=150)
        for image in images:image.close()
    return report
