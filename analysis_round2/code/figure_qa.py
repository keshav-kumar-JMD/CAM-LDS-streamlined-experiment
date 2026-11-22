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
        out += [t for t in ax.texts if t.get_text()]
        if ax.get_legend():out.append(ax.get_legend())
    out += [t for t in fig.texts]
    out += list(fig.legends)
    if fig._suptitle:out.append(fig._suptitle)
    if getattr(fig,"_supxlabel",None):out.append(fig._supxlabel)
    if getattr(fig,"_supylabel",None):out.append(fig._supylabel)
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
    # Colour bars may be ordinary appended axes or Matplotlib colourbar axes.
    colorbars=[ax for ax in fig.axes if ax.get_label()=="<colorbar>" or getattr(ax,"_colorbar",None) is not None]
    heatmaps=[ax for ax in axes if ax.images]
    for cb in colorbars:
        cbbox=cb.get_window_extent(renderer)
        for heatmap in heatmaps:
            hbox=heatmap.get_window_extent(renderer)
            iw=max(0,min(cbbox.x1,hbox.x1)-max(cbbox.x0,hbox.x0));ih=max(0,min(cbbox.y1,hbox.y1)-max(cbbox.y0,hbox.y0))
            if iw>1 and ih>1:problems.append("colour bar overlaps heatmap axes")
    # Titles must not enter another panel's tick-label region.
    for ax in axes:
        title=ax.title
        if not title.get_text():continue
        tbox=title.get_window_extent(renderer)
        for other in axes:
            if other is ax:continue
            for tick in other.get_xticklabels()+other.get_yticklabels():
                if tick.get_visible() and tick.get_text() and tbox.overlaps(tick.get_window_extent(renderer)):
                    problems.append(f"panel title {title.get_text()!r} overlaps tick label {tick.get_text()!r}")
    # A visible plotting axes must carry at least one plotted artist. Text-only
    # not-estimable panels count as content; deliberately removed axes do not.
    for ax in axes:
        if not ax.get_visible():continue
        has_content=bool(ax.lines or ax.patches or ax.collections or ax.images or [t for t in ax.texts if t.get_text()])
        if not has_content:problems.append(f"empty visible axes: {ax.get_title()!r}")
        labels=[t.get_text() for t in ax.get_yticklabels() if t.get_visible() and t.get_text()]
        duplicates=sorted({label for label in labels if labels.count(label)>1})
        if duplicates:problems.append(f"duplicate y tick labels: {duplicates!r}")
        # Free annotations (bar values, k/n labels, callouts) must not collide
        # with tick labels on the same panel.
        ticks=[t for t in ax.get_xticklabels()+ax.get_yticklabels() if t.get_visible() and t.get_text()]
        for annotation in [t for t in ax.texts if t.get_visible() and t.get_text()]:
            abox=annotation.get_window_extent(renderer)
            for tick in ticks:
                if abox.overlaps(tick.get_window_extent(renderer)):
                    problems.append(f"annotation {annotation.get_text()!r} overlaps tick label {tick.get_text()!r}")
    return sorted(set(problems))

def write_qa_outputs(root,regenerated,records,output_root=None):
    from PIL import Image
    root=Path(root);base=Path(output_root) if output_root else root/"analysis";figures=base/"qa";figures.mkdir(parents=True,exist_ok=True)
    report=pd.DataFrame(records,columns=["figure","check","status","message"])
    if report.empty:report=pd.DataFrame([{"figure":"ALL","check":"summary","status":"pass","message":"No QA failures"}])
    report.to_csv(figures/"_qa_report.csv",index=False,encoding="utf-8-sig")
    images=[]
    for path in regenerated:
        path=Path(path)
        if path.exists() and path.suffix.lower()==".png":images.append(Image.open(path).convert("RGB"))
    if images:
        images[0].save(figures/"_qa_contact_sheet.pdf",save_all=True,append_images=images[1:],resolution=150)
        for image in images:image.close()
    return report
