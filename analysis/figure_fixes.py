"""Notebook 10 print-size figure repairs.

This module is deliberately table-driven: it redraws figures from Notebook
10's materialized analysis tables and never recomputes statistical estimates.
Every save also refreshes the companion plotted-data CSV and manifest row.
"""
from __future__ import annotations

from pathlib import Path
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

from .figure_qa import write_qa_outputs
from .plot_style import (
    BF16_COLOR, CONDITION_MARKERS, CONDITION_NAMES, FIELD_ABBR, FIELD_NAMES,
    FIELD_ORDER, MODEL_COLORS, MODEL_NAMES, MODEL_ORDER, MODAL_COLOR,
    ONE_BIT_COLOR, ORDINAL_COLORS, REFERENCE_NAMES, TEACHER_COLOR,
    annotated_heatmap, display_condition, display_field, display_model,
    display_representation, rounded_limit, save_figure,
)


def _num(frame, columns):
    for column in columns:
        if column in frame: frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _sentence(value):
    return str(value).replace("_", " ").strip().capitalize()


class Notebook10FigureFixes:
    def __init__(self, root):
        self.root = Path(root)
        self.tables = self.root / "analysis" / "tables"
        self.figures = self.root / "analysis" / "figures"
        self.qa = []
        self.regenerated = []
        self.sizes = []

    def read(self, name):
        path = self.tables / name
        return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()

    def save(self, filename, fig, data, sources, key):
        old = self.figures / filename
        old_size = ""
        if old.exists():
            try:
                from PIL import Image
                with Image.open(old) as image:
                    old_size = f"{image.width/300:.2f} x {image.height/300:.2f}"
            except Exception: pass
        width, height = fig.get_size_inches()
        path, _, _ = save_figure(self.root, filename, fig, data, sources, key, self.qa)
        self.regenerated.append(path)
        self.sizes.append({"figure": filename, "old_size_inches": old_size,
                           "new_size_inches": f"{width:.2f} x {height:.2f}"})

    @staticmethod
    def _representation(row):
        return "BF16" if str(row.get("representation_group", row.get("representation", ""))) == "BF16 HF" else "One-bit"

    @staticmethod
    def _field_key(value):
        value = str(value)
        if value in FIELD_ORDER: return value
        for key in FIELD_ORDER:
            if value == key.removeprefix("score_"): return key
        return value

    def split_balance(self):
        scenarios = self.read("split_scenario_balance.csv")
        multiplicity = self.read("split_label_multiplicity.csv")
        lengths = self.read("split_prompt_length_comparison.csv")
        if scenarios.empty or multiplicity.empty or lengths.empty: return
        labels = {"train_original":"Training (original, 705)", "train_feasible":"Training (context-feasible, 604)",
                  "validation":"Validation", "test":"Test"}
        colours = dict(zip(labels, ["#4477AA", "#66CCEE", "#228833", "#CC6677"]))
        fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.85), sharey=True, layout="constrained")
        for pop in labels:
            g = scenarios[scenarios.population.eq(pop)].copy()
            if len(g): axes[0].plot(g.scenario, pd.to_numeric(g.proportion, errors="coerce"), marker="o", ms=3, lw=1, color=colours[pop], label=labels[pop])
        axes[0].set(title="A  Scenario mix", xlabel="Scenario", ylabel="Share of rows")
        m = multiplicity.copy(); m["label_count"] = pd.to_numeric(m.label_count, errors="coerce")
        m["label_bin"] = m.label_count.clip(upper=5).map(lambda x: "≥5" if x == 5 else str(int(x)))
        bins = ["1", "2", "3", "4", "≥5"]
        for pop in labels:
            g = m[m.population.eq(pop)].groupby("label_bin").proportion.sum().reindex(bins, fill_value=0)
            axes[1].plot(bins, g, marker="o", ms=3, lw=1, color=colours[pop])
        axes[1].set(title="B  Ground-truth labels per row", xlabel="Number of labels", ylabel="Share of rows")
        char = lengths[lengths.metric.astype(str).eq("characters")].copy()
        for pop in labels:
            row = char[char.population.eq(pop)]
            if len(row):
                x = pd.to_numeric(row[["median", "q90", "maximum"]].iloc[0], errors="coerce")
                axes[2].plot(x, [.5, .9, 1], marker="o", ms=3, lw=1, color=colours[pop])
        axes[2].set_xscale("log"); axes[2].xaxis.set_major_locator(FixedLocator([2000, 5000, 10000, 20000])); axes[2].xaxis.set_minor_formatter(NullFormatter())
        axes[2].set_xticklabels(["2×10³", "5×10³", "10⁴", "2×10⁴"])
        axes[2].set(title="C  Prompt-length quantiles", xlabel="Prompt length (characters)", ylabel="Cumulative share")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, -.03), ncol=2)
        fig.set_layout_engine(None);fig.subplots_adjust(bottom=.25,wspace=.38)
        self.save("18G_split_label_and_prompt_balance.png", fig, pd.concat([scenarios.assign(source_table="scenario"), multiplicity.assign(source_table="multiplicity"), lengths.assign(source_table="length")], ignore_index=True, sort=False),
                  "split_scenario_balance.csv; split_label_multiplicity.csv; split_prompt_length_comparison.csv", "Shared colours identify the four split populations; label counts of five or more are pooled.")

    def training_subset(self):
        length = self.read("training_subset_length_comparison.csv")
        feasibility = self.read("training_subset_model_feasibility.csv")
        scenarios = self.read("training_subset_scenario_comparison.csv")
        if length.empty or feasibility.empty: return
        fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.0), layout="constrained")
        y0 = np.arange(len(length)); h=.38
        axes[0].barh(y0-h/2, pd.to_numeric(length.retained_median, errors="coerce"), h, label="Retained (604)", color="#4477AA")
        axes[0].barh(y0+h/2, pd.to_numeric(length.excluded_median, errors="coerce"), h, label="Excluded (101)", color="#CC6677")
        axes[0].set(title="A  Prompt length", xlabel="Median model tokens", yticks=y0, yticklabels=[display_model(v) if v in MODEL_NAMES else str(v) for v in length.source_model])
        if not scenarios.empty:
            axes[1].plot(scenarios.scenario, pd.to_numeric(scenarios.retained_rate, errors="coerce"), marker="o", ms=3, label="Retained (604)")
            axes[1].plot(scenarios.scenario, pd.to_numeric(scenarios.excluded_rate, errors="coerce"), marker="o", ms=3, label="Excluded (101)")
        axes[1].set(title="B  Scenario composition", xlabel="Scenario", ylabel="Share of rows")
        f = feasibility.copy(); f["model_specific_nonfit_rows"] = pd.to_numeric(f.model_specific_nonfit_rows, errors="coerce"); f=f.sort_values("model_specific_nonfit_rows")
        y=np.arange(len(f)); axes[2].barh(y, f.model_specific_nonfit_rows, color=[MODEL_COLORS.get(m,"#777") for m in f.model])
        axes[2].set(yticks=y, yticklabels=[display_model(m) for m in f.model], title="C  Rows exceeding context", xlabel="Training rows")
        for yi,value in zip(y,f.model_specific_nonfit_rows): axes[2].text(value+.5, yi, f"{int(value)}", va="center", fontsize=8)
        handles=[Line2D([0],[0],color="#4477AA",lw=6,label="Retained (604)"),Line2D([0],[0],color="#CC6677",lw=6,label="Excluded (101)")]
        fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,-.03),ncol=2);fig.set_layout_engine(None);fig.subplots_adjust(bottom=.22,wspace=.72)
        data=pd.concat([length.assign(source_table="length"),feasibility.assign(source_table="feasibility"),scenarios.assign(source_table="scenario")],ignore_index=True,sort=False)
        self.save("18F_training_subset_representativeness.png",fig,data,"training_subset_length_comparison.csv; training_subset_model_feasibility.csv; training_subset_scenario_comparison.csv","Retained and excluded training rows are compared without changing the training subset.")

    def training_loss(self):
        frame=_num(self.read("training_loss_history.csv"),["step","loss"])
        if frame.empty:return
        fig,axes=plt.subplots(2,3,figsize=(6.3,4.4),sharex=True,layout="constrained")
        colours={"technique":"#0072B2","explanation":"#E69F00","joint":"#009E73","full":"#D55E00"}
        for ax,model in zip(axes.flat,MODEL_ORDER):
            for condition in ["technique","explanation","joint","full"]:
                g=frame[(frame.model==model)&(frame.condition==condition)].sort_values("step")
                if len(g):ax.plot(g.step,g.loss,lw=1,color=colours[condition],label=display_condition(condition))
            ax.set_title(display_model(model));ax.grid(alpha=.2)
        for ax in axes[1]:ax.set_xlabel("Logged training step")
        for ax in axes[:,0]:ax.set_ylabel("Training loss")
        handles=[Line2D([0],[0],color=colours[c],label=display_condition(c)) for c in colours]
        fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,-.01),ncol=4);fig.subplots_adjust(bottom=.17)
        self.save("21_training_loss_curves.png",fig,frame,"training_loss_history.csv","Panels identify model family; line colour identifies adapter.")

    def reference_ladder_all(self):
        frame=_num(self.read("reference_ladder.csv"),["mean","low95","high95","eligible_n","display_order"])
        if frame.empty:return
        frame=frame[frame.dimension.isin(FIELD_ORDER)].copy()
        order=(frame[["display_order","stream_display_name","model","condition","representation_group"]].drop_duplicates().sort_values("display_order"))
        ymap={name:i for i,name in enumerate(order.stream_display_name)};height=max(6.8,.245*len(order)+1.6)
        fig,axes=plt.subplots(1,7,figsize=(6.3,height),sharey=True,layout="constrained")
        for ax,field in zip(axes,FIELD_ORDER):
            g=frame[frame.dimension.eq(field)]
            teacher=g[g.model.eq("gpt55_teacher")]["mean"];modal=g[g.model.eq("baseline_modal")]["mean"]
            if len(teacher):ax.axvline(teacher.iloc[0],color=TEACHER_COLOR,ls="--",lw=.8)
            if len(modal):ax.axvline(modal.iloc[0],color=MODAL_COLOR,ls=":",lw=.8)
            for _,r in g.iterrows():
                y=ymap[r.stream_display_name];mean=r["mean"]
                if not np.isfinite(mean):ax.text(.15,y,"–",ha="center",va="center",fontsize=9,fontweight="bold");continue
                rep=str(r.representation_group);condition=str(r.condition);ref=str(r.model) in REFERENCE_NAMES
                colour=REFERENCE_NAMES and ({"gpt55_teacher":TEACHER_COLOR,"baseline_modal":MODAL_COLOR,"baseline_null":"#777"}.get(str(r.model))) if ref else (BF16_COLOR if rep=="BF16 HF" else ONE_BIT_COLOR)
                marker="s" if condition=="full" else "o";face="white" if (rep!="BF16 HF" or mean==0) else colour
                if r.eligible_n<30:colour="#888";face="#bbb" if mean!=0 else "white"
                ax.errorbar(mean,y,xerr=[[mean-r.low95],[r.high95-mean]],fmt=marker,ms=3.8,mec=colour,mfc=face,ecolor=colour,lw=.8,capsize=1.5)
            ax.set_xlim(0,4);ax.set_title(FIELD_ABBR[field]);ax.set_xlabel("Mean score")
        axes[0].set_yticks(range(len(order)))
        def label(r):
            if r.model in REFERENCE_NAMES:return REFERENCE_NAMES[r.model]
            return f"{display_model(r.model)}\n{display_representation(r.representation_group)} · {display_condition(r.condition)}"
        axes[0].set_yticklabels([label(r) for _,r in order.iterrows()]);axes[0].invert_yaxis()
        handles=[Line2D([0],[0],marker="o",color=BF16_COLOR,mfc=BF16_COLOR,lw=0,label="BF16 · Untreated"),Line2D([0],[0],marker="s",color=BF16_COLOR,mfc=BF16_COLOR,lw=0,label="BF16 · SFT-full"),Line2D([0],[0],marker="o",color=ONE_BIT_COLOR,mfc="white",lw=0,label="One-bit · Untreated"),Line2D([0],[0],marker="s",color=ONE_BIT_COLOR,mfc="white",lw=0,label="One-bit · SFT-full")]
        fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,-.005),ncol=2);fig.subplots_adjust(bottom=.08)
        self.save("19A_all_reference_ladder_all_fields.png",fig,frame,"reference_ladder.csv","Shape identifies treatment; fill identifies representation; dashed and dotted vertical lines identify teacher and modal references.")

    def rank_all(self):
        metrics=_num(self.read("judge_mediated_rank_metrics_by_stream.csv"),["k","estimate","low95","high95","n","nonempty_list_rows"])
        if metrics.empty:return
        metrics=metrics[metrics.population.eq("evaluation_285")].copy()
        desired=[("hit_rate",1,"Hit@1"),("hit_rate",5,"Hit@5"),("hit_rate",10,"Hit@10"),("precision_at_k_fixed",5,"P@5"),("recall_at_k_macro",10,"R@10")]
        core=metrics[metrics.condition_short.isin(["untreated","full"])].copy()
        identities=core[["model","condition_short","representation_group","exact_representation","stream_display_name"]].drop_duplicates()
        refs=identities[identities.model.isin(REFERENCE_NAMES)]
        models=identities[identities.model.isin(MODEL_ORDER)].copy()
        models["model_order"]=models.model.map({m:i for i,m in enumerate(MODEL_ORDER)});models["condition_order"]=models.condition_short.map({"untreated":0,"full":1});models["rep_order"]=models.representation_group.map({"BF16 HF":0,"one-bit frontier":1})
        models=models.sort_values(["model_order","rep_order","condition_order"])
        refs=refs.assign(ref_order=refs.model.map({"gpt55_teacher":0,"baseline_modal":1,"baseline_null":2})).sort_values("ref_order")
        order=pd.concat([models,refs],ignore_index=True).drop_duplicates("stream_display_name");ymap={s:i for i,s in enumerate(order.stream_display_name)}
        fig,axes=plt.subplots(1,6,figsize=(6.3,max(7,.25*len(order)+1.5)),sharey=True,layout="constrained")
        plotted=[]
        for ax,(metric,k,title) in zip(axes[:5],desired):
            g=core[(core.metric.eq(metric))&(core.k.eq(k))]
            for _,r in g.iterrows():
                if r.stream_display_name not in ymap:continue
                y=ymap[r.stream_display_name];colour={"gpt55_teacher":TEACHER_COLOR,"baseline_modal":MODAL_COLOR,"baseline_null":"#777"}.get(str(r.model),BF16_COLOR if r.representation_group=="BF16 HF" else ONE_BIT_COLOR)
                marker="s" if r.condition_short=="full" else "o";face="white" if r.representation_group!="BF16 HF" or r.estimate==0 else colour
                ax.errorbar(r.estimate,y,xerr=[[r.estimate-r.low95],[r.high95-r.estimate]],fmt=marker,ms=3.5,mec=colour,mfc=face,ecolor=colour,lw=.8,capsize=1.2);plotted.append(r.to_dict())
            teacher=g[g.model.eq("gpt55_teacher")].estimate;modal=g[g.model.eq("baseline_modal")].estimate
            if len(teacher):ax.axvline(teacher.iloc[0],ls="--",lw=.8,color=TEACHER_COLOR)
            if len(modal):ax.axvline(modal.iloc[0],ls=":",lw=.8,color=MODAL_COLOR)
            ax.set_xlim(0,.2 if metric=="precision_at_k" else 1);ax.set_title(title);ax.set_xlabel("Proportion")
        ax=axes[5]
        extraction=(core.sort_values("k").groupby(["model","condition_short","representation_group","exact_representation","stream_display_name"],as_index=False).first())
        extraction["extraction_share"]=extraction.nonempty_list_rows/extraction.n.replace(0,np.nan)
        for _,r in extraction.iterrows():
            if r.stream_display_name not in ymap:continue
            y=ymap[r.stream_display_name];colour={"gpt55_teacher":TEACHER_COLOR,"baseline_modal":MODAL_COLOR,"baseline_null":"#777"}.get(str(r.model),BF16_COLOR if r.representation_group=="BF16 HF" else ONE_BIT_COLOR)
            marker="s" if r.condition_short=="full" else "o";face="white" if r.representation_group!="BF16 HF" or r.extraction_share==0 else colour
            ax.plot(r.extraction_share,y,marker=marker,mec=colour,mfc=face,color=colour,ms=3.5);ax.text(1.02,y,f"{int(r.nonempty_list_rows)}/{int(r.n)}",va="center",fontsize=8,clip_on=False)
        ax.set_xlim(0,1);ax.set_title("Identifiers extracted");ax.set_xlabel("Share of rows")
        axes[0].set_yticks(range(len(order)))
        def lab(r):
            if r.model in REFERENCE_NAMES:return REFERENCE_NAMES[r.model]
            return f"{display_model(r.model)}\n{display_representation(r.representation_group)} · {display_condition(r.condition_short)}"
        axes[0].set_yticklabels([lab(r) for _,r in order.iterrows()]);axes[0].invert_yaxis()
        handles=[Line2D([0],[0],marker="o",mfc=BF16_COLOR,mec=BF16_COLOR,color="none",label="BF16 · Untreated"),Line2D([0],[0],marker="s",mfc=BF16_COLOR,mec=BF16_COLOR,color="none",label="BF16 · SFT-full"),Line2D([0],[0],marker="o",mfc="white",mec=ONE_BIT_COLOR,color="none",label="One-bit · Untreated"),Line2D([0],[0],marker="s",mfc="white",mec=ONE_BIT_COLOR,color="none",label="One-bit · SFT-full")]
        fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,-.005),ncol=2);fig.subplots_adjust(bottom=.08,right=.96)
        pdata=pd.concat([pd.DataFrame(plotted),extraction.assign(metric="identifier_extraction_share",k=np.nan,estimate=extraction.extraction_share)],ignore_index=True,sort=False)
        self.save("19C_all_ranked_technique_metrics.png",fig,pdata,"judge_mediated_rank_metrics_by_stream.csv","Exact-ID judge-extracted metrics; shape identifies treatment and fill identifies representation. Reference lines mark teacher and modal baseline.")

    def rank_figures(self):
        frame=_num(self.read("judge_mediated_rank_metrics_by_stream.csv"),["k","estimate","low95","high95"])
        if frame.empty:return
        for population,filename,conditions in [("all_990","19C_990_untreated_judge_mediated_hit_at_k.png",["untreated"]),("evaluation_285","19C_judge_mediated_hit_at_k_core_streams.png",["untreated","full"])]:
            g=frame[(frame.population.eq(population))&(frame.metric.eq("hit_rate"))&(frame.condition_short.isin(conditions))&frame.model.isin(MODEL_ORDER)]
            if g.empty:continue
            fig,axes=plt.subplots(2,3,figsize=(6.3,4.8),sharex=True,sharey=True,layout="constrained")
            for ax,model in zip(axes.flat,MODEL_ORDER):
                h=g[g.model.eq(model)]
                for rep in ["BF16 HF","one-bit frontier"]:
                    for condition in conditions:
                        q=h[(h.representation_group.eq(rep))&(h.condition_short.eq(condition))].sort_values("k")
                        if q.empty:continue
                        color=BF16_COLOR if rep=="BF16 HF" else ONE_BIT_COLOR;ls="-" if rep=="BF16 HF" else "--";marker="o" if condition=="untreated" else "s"
                        ax.plot(q.k,q.estimate,color=color,ls=ls,marker=marker,ms=3,label=f"{display_representation(rep)} · {display_condition(condition)}")
                        ax.fill_between(q.k,q.low95,q.high95,color=color,alpha=.2)
                ax.set_title(display_model(model));ax.set_xticks([1,3,5,10]);ax.set_ylim(-.002,max(.02,float(g.high95.max())*1.25));ax.grid(alpha=.2)
            for ax in axes[1]:ax.set_xlabel("k")
            for ax in axes[:,0]:ax.set_ylabel("Hit@k")
            handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,-.005),ncol=2);fig.subplots_adjust(bottom=.17)
            self.save(filename,fig,g,"judge_mediated_rank_metrics_by_stream.csv","Lines are exact-ID Hit@k; bands are scenario-step cluster-bootstrap 95% intervals.")
        # Compact appendix heatmaps for all evaluated streams.
        specs=[("hit_rate","S19C_judge_mediated_hit_at_k_all_streams.png"),("precision_at_k_fixed","S19D_judge_mediated_precision_at_k_all_streams.png"),("recall_at_k_macro","S19E_judge_mediated_recall_at_k_all_streams.png")]
        for metric,filename in specs:
            g=frame[(frame.population.eq("evaluation_285"))&(frame.metric.eq(metric))&frame.model.isin(MODEL_ORDER)]
            if g.empty:continue
            ks=[k for k in [1,3,5,10] if k in set(g.k)];streams=g.stream_display_name.drop_duplicates().tolist();matrix=np.full((len(streams),len(ks)),np.nan)
            for i,s in enumerate(streams):
                for j,k in enumerate(ks):
                    q=g[(g.stream_display_name.eq(s))&(g.k.eq(k))]
                    if len(q):matrix[i,j]=q.estimate.iloc[0]
            vmax=max(.01,math.ceil(np.nanmax(matrix)*100)/100) if np.isfinite(matrix).any() else .01
            identities=g[["stream_display_name","model","condition_short","representation_group"]].drop_duplicates("stream_display_name").set_index("stream_display_name")
            labels=[]
            for stream_name in streams:
                r=identities.loc[stream_name];labels.append(f"{display_model(r.model)} · {display_representation(r.representation_group)} · {display_condition(r.condition_short)}")
            fig,ax=plt.subplots(figsize=(6.3,max(5,.18*len(streams)+1.2)),layout="constrained");annotated_heatmap(ax,matrix,cmap="cividis",vmin=0,vmax=vmax,cbar_label=f"Observed range: 0–{vmax:.2f}")
            ax.set_xticks(range(len(ks)),[f"k={k}" for k in ks]);ax.set_yticks(range(len(streams)),labels);ax.set_title(metric.replace("_"," ").title())
            self.save(filename,fig,g,"judge_mediated_rank_metrics_by_stream.csv","Sequential colour is scaled to the observed metric maximum; cells show exact values.")
        parent=_num(self.read("judge_mediated_parent_rank_metrics_by_stream.csv"),["k","estimate"])
        exact=frame[(frame.population.eq("evaluation_285"))&(frame.metric.eq("hit_rate"))]
        parent=parent[(parent.population.eq("evaluation_285"))&parent.metric.isin(["hit_at_k","hit_rate"])]
        if len(exact) and len(parent):
            keys=["model","condition_short","representation_group","k"];m=exact.merge(parent[keys+["estimate"]],on=keys,suffixes=("_exact","_parent"));fig,ax=plt.subplots(figsize=(6.3,3.8),layout="constrained")
            for (model,rep),q in m.groupby(["model","representation_group"]):ax.plot(q.estimate_exact,q.estimate_parent,"o",ms=3,color=MODEL_COLORS.get(model,"#777"),mfc="white" if rep!="BF16 HF" else MODEL_COLORS.get(model,"#777"))
            lim=max(m.estimate_exact.max(),m.estimate_parent.max(),.01);ax.plot([0,lim],[0,lim],ls="--",lw=.8,color="#777");ax.set(xlabel="Exact-ID Hit@k",ylabel="Parent-normalized Hit@k",title="Parent-normalization sensitivity")
            self.save("S19F_judge_mediated_parent_normalization_sensitivity.png",fig,m,"judge_mediated_rank_metrics_by_stream.csv; judge_mediated_parent_rank_metrics_by_stream.csv","Points above the diagonal improve after sub-techniques are collapsed to parent IDs.")

    def untreated_deployment_990(self):
        frame=_num(self.read("treatment_contrasts.csv"),["mean_difference","low95","high95"])
        for field in FIELD_ORDER:
            g=frame[(frame.population.eq("all_990"))&(frame.contrast.eq("untreated_one_bit_minus_bf16"))&(frame.dimension.eq(field))]
            if g.empty:continue
            g=g.set_index("model").reindex(MODEL_ORDER).reset_index();y=np.arange(len(g));vals=g.mean_difference.to_numpy(float);lo=g.low95.to_numpy(float);hi=g.high95.to_numpy(float);limit=rounded_limit(pd.concat([g.low95,g.high95]),.5)
            fig,ax=plt.subplots(figsize=(6.3,3.4),layout="constrained");ax.axvline(0,color="#777",lw=.8);ax.errorbar(vals,y,xerr=[vals-lo,hi-vals],fmt="o",capsize=2,color="#0072B2");ax.set_yticks(y,[display_model(m) for m in g.model]);ax.invert_yaxis();ax.set_xlim(-limit,limit);ax.set_xlabel("One-bit minus BF16 (judge score)");ax.set_title(f"{display_field(field)}: untreated deployment contrast")
            self.save(f"03C_990_{field.removeprefix('score_')}_untreated_deployment_stack_effects.png",fig,g,"treatment_contrasts.csv","Paired 990-row untreated representation contrast with 95% cluster-bootstrap interval.")

    def qwen_size(self):
        frame=_num(self.read("size_contrasts.csv"),["mean_difference","low95","high95","mde"])
        g=frame[frame.record_type.eq("primary_paired_qwen") & frame.dimension.isin(FIELD_ORDER)].copy()
        if g.empty:return
        rows=[("BF16 HF","untreated"),("BF16 HF","full"),("one-bit frontier","untreated"),("one-bit frontier","full")];limit=rounded_limit(pd.concat([g.low95,g.high95,g.mde]),.5)
        fig,axes=plt.subplots(2,4,figsize=(6.3,4.2),sharex=True,sharey=True,layout="constrained")
        for ax,field in zip(axes.flat,FIELD_ORDER):
            h=g[g.dimension.eq(field)];y=np.arange(4)
            for i,(rep,condition) in enumerate(rows):
                q=h[(h.representation_group.eq(rep))&(h.condition.eq(condition))]
                if q.empty:ax.text(0,i,"no scorable rows",ha="center",va="center",fontsize=8);continue
                r=q.iloc[0];ax.axvspan(-r.mde,r.mde,color="#ddd",alpha=.3,zorder=0);ax.errorbar(r.mean_difference,i,xerr=[[r.mean_difference-r.low95],[r.high95-r.mean_difference]],fmt="o",capsize=2,zorder=2,color="#0072B2")
            ax.axvline(0,color="#777",lw=.8);ax.set_title(FIELD_ABBR[field]);ax.set_xlim(-limit,limit)
        axes.flat[-1].remove();axes[0,0].set_yticks(range(4),[f"{display_representation(r)} · {display_condition(c)}" for r,c in rows]);axes[0,0].invert_yaxis()
        for ax in axes[1,:-1]:ax.set_xlabel("7B minus 1.5B score")
        self.save("16C_qwen_family_size_contrasts.png",fig,g,"size_contrasts.csv","Points compare Qwen2.5 7B with Qwen2.5 1.5B; shaded bands are MDEs.")


    def score_distributions(self):
        frame=_num(self.read("model_score_distributions.csv"),["score","proportion","denominator"])
        if frame.empty:return
        for rep,prefix in [("BF16 HF","02A"),("one-bit frontier","02B")]:
            for field in FIELD_ORDER:
                short=field.removeprefix("score_")
                g=frame[(frame.population.eq("evaluation_285"))&(frame.representation_group.eq(rep))&(frame.dimension.eq(field))&(frame.condition_short.isin(["untreated","full"]))]
                if g.empty:continue
                models=[m for m in MODEL_ORDER if m in set(g.model)];rows=[]
                for model in models:
                    for condition in ["untreated","full"]:rows.append((model,condition))
                fig,ax=plt.subplots(figsize=(6.3,max(3,.23*len(rows)+1.35)),layout="constrained");left=np.zeros(len(rows))
                for score,color in enumerate(ORDINAL_COLORS):
                    vals=[]
                    for model,condition in rows:
                        r=g[(g.model.eq(model))&(g.condition_short.eq(condition))&(g.score.eq(score))]
                        vals.append(float(r.proportion.iloc[0]) if len(r) else 0)
                    ax.barh(range(len(rows)),vals,left=left,color=color,label=str(score));left+=np.asarray(vals)
                ax.set_yticks(range(len(rows)));ax.set_yticklabels([f"{display_model(m)} · {display_condition(c)}" for m,c in rows]);ax.invert_yaxis();ax.set_xlim(0,1.13)
                ax.set_xlabel("Share of scorable answers");ax.set_title(f"{display_field(field)}: {display_representation(rep)} score distribution")
                for i,(model,condition) in enumerate(rows):
                    r=g[(g.model.eq(model))&(g.condition_short.eq(condition))];n=int(pd.to_numeric(r.denominator,errors="coerce").max()) if len(r) else 0;ax.text(1.01,i,f"n = {n}",va="center",fontsize=8)
                fig.legend(title="Score",loc="lower center",bbox_to_anchor=(.5,-.01),ncol=5);fig.subplots_adjust(bottom=.18)
                self.save(f"{prefix}_{short}_{'bf16' if rep=='BF16 HF' else 'one_bit'}_untreated_full.png",fig,g,"model_score_distributions.csv","Stacked colours show ordinal judge-score categories; row-end labels give scorable n.")

    def score_overviews(self):
        source=self.read("model_scores_by_dimension.csv")
        if source.empty:return
        source=_num(source,["mean"])
        for rep,filename in [("BF16 HF","01A_score_overview_bf16.png"),("one-bit frontier","01B_score_overview_one_bit_frontier.png")]:
            g=source[(source.population.eq("evaluation_285"))&(source.representation_group.eq(rep))&(source.condition_short.isin(["untreated","full"]))&(source.dimension.isin(FIELD_ORDER))]
            rows=[]
            for model in MODEL_ORDER:
                for condition in ["untreated","full"]:
                    if len(g[(g.model.eq(model))&(g.condition_short.eq(condition))]):rows.append((model,condition))
            matrix=np.full((len(rows),len(FIELD_ORDER)),np.nan)
            for i,(model,condition) in enumerate(rows):
                for j,field in enumerate(FIELD_ORDER):
                    r=g[(g.model.eq(model))&(g.condition_short.eq(condition))&(g.dimension.eq(field))]
                    if len(r):matrix[i,j]=r["mean"].iloc[0]
            fig,ax=plt.subplots(figsize=(6.3,max(3,.25*len(rows)+1.25)),layout="constrained");annotated_heatmap(ax,matrix,cmap="cividis",vmin=0,vmax=4,cbar_label="Mean judge score (0–4)")
            ax.set_xticks(range(7),[FIELD_ABBR[f] for f in FIELD_ORDER]);ax.set_yticks(range(len(rows)),[f"{display_model(m)} · {display_condition(c)}" for m,c in rows]);ax.set_title(f"Judge-score overview: {display_representation(rep)}");ax.set_xlabel("Judge dimension (abbreviations defined in the figure key)")
            self.save(filename,fig,g,"model_scores_by_dimension.csv","Rows identify model and treatment; columns follow the canonical seven-field order.")

    def concordance(self):
        frame=_num(self.read("model_item_difficulty_concordance.csv"),["spearman","low_score_jaccard"])
        if frame.empty:return
        for population in ["all_990","evaluation_285"]:
            for field in FIELD_ORDER:
                subsets=[]
                if population=="all_990":subsets=[("untreated",None,f"17_990_{field.removeprefix('score_')}_untreated_item_difficulty_concordance.png")]
                else:
                    for condition in ["untreated","full"]:
                        for rep in ["BF16 HF","one-bit frontier"]:
                            rep_file="bf16_hf" if rep=="BF16 HF" else "one-bit-frontier"
                            subsets.append((condition,rep,f"17_{field.removeprefix('score_')}_{condition}_{rep_file}_item_difficulty_concordance.png"))
                for condition,rep,filename in subsets:
                    g=frame[(frame.population.eq(population))&(frame.dimension.eq(field))&(frame.condition.eq(condition))]
                    if rep:g=g[g.representation.eq(rep)]
                    if g.empty:continue
                    panels=[("BF16 HF","BF16"),("one-bit frontier","One-bit")] if rep is None else [(rep,display_representation(rep))]
                    fig,axes=plt.subplots(1,len(panels),figsize=(6.3,4.2),squeeze=False,sharey=True,layout="constrained")
                    for ax,(representation,title) in zip(axes.flat,panels):
                        h=g[g.representation.eq(representation)];matrix=np.full((6,6),np.nan)
                        for _,r in h.iterrows():
                            if r.model_a in MODEL_ORDER and r.model_b in MODEL_ORDER:
                                i=MODEL_ORDER.index(r.model_a);j=MODEL_ORDER.index(r.model_b);matrix[max(i,j),min(i,j)]=r.spearman
                        if np.isfinite(matrix).any():annotated_heatmap(ax,matrix,cmap="cividis",vmin=-1,vmax=1,cbar=True,cbar_label="Spearman correlation")
                        else:ax.text(.5,.5,"Not estimable:\none-bit scores are constant",ha="center",va="center",transform=ax.transAxes);ax.set_facecolor("white")
                        ax.set_xticks(range(6),[display_model(m) for m in MODEL_ORDER],rotation=90,ha="center");ax.set_yticks(range(6),[display_model(m) for m in MODEL_ORDER]);ax.set_title(title)
                    if len(panels)>1:axes.flat[1].tick_params(labelleft=False)
                    if len(panels)>1:fig.suptitle(f"{display_field(field)}: untreated item-difficulty concordance")
                    self.save(filename,fig,g,"model_item_difficulty_concordance.csv","Only lower-triangle model pairs are shown; n.e. indicates constant scores or insufficient paired rows.")

    def attack_heatmaps(self):
        frame=_num(self.read("attack_class_scores.csv"),["mean","n"])
        if frame.empty:return
        for field in FIELD_ORDER:
            short=field.removeprefix("score_")
            g=frame[(frame.population.eq("all_990"))&(frame.condition.eq("untreated"))&(frame.dimension.eq(field))]
            if g.empty:continue
            techniques=(g.groupby("technique").n.sum().sort_values(ascending=False).head(20).index.tolist())
            matrix=np.full((len(techniques),len(MODEL_ORDER)*2),np.nan);columns=[]
            for model in MODEL_ORDER:
                for rep in ["BF16 HF","one-bit frontier"]:columns.append((model,rep))
            for i,t in enumerate(techniques):
                for j,(model,rep) in enumerate(columns):
                    r=g[(g.technique.eq(t))&(g.model.eq(model))&(g.representation.eq(rep))]
                    if len(r):matrix[i,j]=r["mean"].iloc[0]
            vmax=max(.5,math.ceil(np.nanmax(matrix)*2)/2) if np.isfinite(matrix).any() else .5
            fig,ax=plt.subplots(figsize=(6.3,max(4,.27*len(techniques)+1.3)),layout="constrained");annotated_heatmap(ax,matrix,cmap="cividis",vmin=0,vmax=vmax,cbar_label=f"Mean score (0–{vmax:g})")
            ax.set_xticks(range(len(columns)),[f"{display_model(m)}\n{display_representation(r)}" for m,r in columns],rotation=45,ha="right");ax.set_yticks(range(len(techniques)),techniques);ax.set_ylabel("ATT&CK technique");ax.set_title(f"{display_field(field)} by ATT&CK technique: untreated 990-row universe")
            self.save(f"06_990_{short}_untreated_attack_class_scores.png",fig,g,"attack_class_scores.csv","Sequential colour encodes the observed mean range; exact zeros are hatched grey.")

    def failures(self):
        rates=_num(self.read("failure_rates.csv"),["rate_per_100"]);sim=_num(self.read("failure_profile_similarity.csv"),["cosine_similarity"])
        if not rates.empty:
            for population,filename,conditions in [("all_990","09_990_untreated_failure_rates.png",["untreated"]),("evaluation_285","09_failure_rate_bars.png",["untreated","full"])]:
                g=rates[(rates.population.eq(population))&(rates.condition_short.isin(conditions))&rates.model.isin(MODEL_ORDER)].copy()
                if g.empty:continue
                categories=g.groupby("failure_category").rate_per_100.sum().sort_values().index.tolist();fig,axes=plt.subplots(2,3,figsize=(6.3,7.5),sharey=True,layout="constrained")
                for ax,model in zip(axes.flat,MODEL_ORDER):
                    h=g[g.model.eq(model)];y=np.arange(len(categories));width=.18;series=[]
                    for rep in ["BF16 HF","one-bit frontier"]:
                        for condition in conditions:series.append((rep,condition))
                    offsets=np.linspace(-width*(len(series)-1)/2,width*(len(series)-1)/2,len(series))
                    for off,(rep,condition) in zip(offsets,series):
                        vals=h[(h.representation_group.eq(rep))&(h.condition_short.eq(condition))].set_index("failure_category").rate_per_100.reindex(categories,fill_value=0)
                        colour=BF16_COLOR if rep=="BF16 HF" else ONE_BIT_COLOR;alpha=1 if condition=="untreated" else .55
                        ax.barh(y+off,vals,height=width,color=colour,alpha=alpha,label=f"{display_representation(rep)} · {display_condition(condition)}")
                    ax.set_title(display_model(model));ax.set_xlabel("Labels per 100 judged rows");ax.set_yticks(y,[_sentence(c) for c in categories]);ax.grid(axis="x",alpha=.2)
                handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,-.005),ncol=2);fig.subplots_adjust(bottom=.1)
                self.save(filename,fig,g,"failure_rates.csv","All panels use the same broad failure-category order and include zero-rate categories.")
        if not sim.empty:
            for population,filename in [("all_990","10_990_untreated_failure_profile_similarity.png"),("evaluation_285","10_failure_profile_similarity.png")]:
                g=sim[(sim.population.eq(population))&(sim.condition.eq("untreated"))]
                if g.empty:continue
                panels=[]
                for rep in ["BF16 HF","one-bit frontier"]:
                    h=g[g.representation_group.eq(rep)];matrix=np.full((6,6),np.nan)
                    for _,r in h.iterrows():
                        if r.model_a in MODEL_ORDER and r.model_b in MODEL_ORDER:matrix[max(MODEL_ORDER.index(r.model_a),MODEL_ORDER.index(r.model_b)),min(MODEL_ORDER.index(r.model_a),MODEL_ORDER.index(r.model_b))]=r.cosine_similarity
                    panels.append((rep,matrix))
                fig,axes=plt.subplots(1,2,figsize=(6.3,4.2),sharey=True,layout="constrained")
                for ax,(rep,matrix) in zip(axes,panels):
                    annotated_heatmap(ax,matrix,cmap="cividis",vmin=0,vmax=1,cbar_label="Cosine similarity");ax.set_xticks(range(6),[display_model(m) for m in MODEL_ORDER],rotation=90,ha="center");ax.set_yticks(range(6),[display_model(m) for m in MODEL_ORDER]);ax.set_title(display_representation(rep))
                axes[1].tick_params(labelleft=False)
                fig.suptitle("Similarity of untreated failure profiles")
                self.save(filename,fig,g,"failure_profile_similarity.csv","Lower triangles compare broad-category failure-rate vectors between model pairs.")

    def stopping(self):
        frame=_num(self.read("token_ceiling_and_stop_reasons.csv"),["natural_eos_rate","at_ceiling_rate","completed"])
        if frame.empty:return
        for population,filename in [("all_990","14_990_untreated_token_ceiling_rates.png"),("evaluation_285","14_token_ceiling_rates.png")]:
            conditions=["untreated"] if population=="all_990" else ["untreated","full"]
            g=frame[(frame.population.eq(population))&(frame.condition.isin(conditions))&frame.model.isin(MODEL_ORDER)]
            if g.empty:continue
            fig,axes=plt.subplots(2,3,figsize=(6.3,5.3),sharex=True,sharey=True,layout="constrained")
            for ax,model in zip(axes.flat,MODEL_ORDER):
                h=g[g.model.eq(model)].copy();rows=[]
                for condition in conditions:
                    for rep in ["BF16 HF","one-bit frontier"]:
                        r=h[(h.condition.eq(condition))&(h.representation.eq(rep))]
                        if len(r):rows.append(r.iloc[0])
                y=np.arange(len(rows));eos=np.array([r.natural_eos_rate for r in rows]);ceiling=np.array([r.at_ceiling_rate for r in rows])
                ax.barh(y,eos,color="#009E73",label="Ended naturally (EOS)");ax.barh(y,ceiling,left=eos,color="#D55E00",label="Reached 1,024-token ceiling")
                ax.set_yticks(y,[f"{display_condition(r.condition)} · {display_representation(r.representation)}" for r in rows]);ax.invert_yaxis();fmt="I2_S" if model=="bitnet2b" else "TQ1_0";ax.set_title(f"{display_model(model)} ({fmt})");ax.set_xlim(0,1)
            for ax in axes[1]:ax.set_xlabel("Share of completed answers")
            fig.legend(*axes.flat[0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.5,-.005),ncol=2);fig.subplots_adjust(bottom=.12)
            self.save(filename,fig,g,"token_ceiling_and_stop_reasons.csv","Green is natural EOS; orange is the fixed 1,024-token ceiling.")

    def contrast_figures(self):
        frame=_num(self.read("treatment_contrasts.csv"),["mean_difference","low95","high95"])
        if frame.empty:return
        for rep,prefix,label in [("BF16","03A","BF16"),("one-bit","03B","one-bit")]:
            for field in FIELD_ORDER:
                key=field.removeprefix("score_")
                pattern="bf16" if rep=="BF16" else "one_bit"
                contrast_name="bf16_sft_minus_untreated" if rep=="BF16" else "one_bit_sft_minus_untreated"
                g=frame[(frame.population.eq("evaluation_285"))&(frame.dimension.eq(field))&frame.contrast.eq(contrast_name)&frame.adapter.isin(["technique","explanation","joint","full"])]
                if g.empty:continue
                limit=rounded_limit(pd.concat([g.low95,g.high95]),.5)
                fig,axes=plt.subplots(2,3,figsize=(6.3,4.4),sharex=True,sharey=True,layout="constrained")
                for ax,model in zip(axes.flat,MODEL_ORDER):
                    h=g[g.model.eq(model)].set_index("adapter").reindex(["technique","explanation","joint","full"]);y=np.arange(4)
                    vals=h.mean_difference.to_numpy(float);lo=h.low95.to_numpy(float);hi=h.high95.to_numpy(float);ax.axvline(0,color="#777",lw=.8);ax.errorbar(vals,y,xerr=[vals-lo,hi-vals],fmt="o",capsize=2,color=MODEL_COLORS[model]);ax.set_yticks(y,[display_condition(a) for a in h.index]);ax.set_title(display_model(model));ax.set_xlim(-limit,limit)
                for ax in axes[1]:ax.set_xlabel("Adapter minus untreated (judge score)")
                fig.suptitle(f"{display_field(field)}: fine-tuning effect in {label}")
                self.save(f"{prefix}_{key}_{'bf16' if rep=='BF16' else 'one_bit'}_sft_effects.png",fig,g,"treatment_contrasts.csv","Points are paired mean differences; capped bars are scenario-step cluster-bootstrap 95% intervals.")
        for field in FIELD_ORDER:
            key=field.removeprefix("score_");g=frame[(frame.population.eq("evaluation_285"))&(frame.dimension.eq(field))&frame.contrast.isin(["untreated_one_bit_minus_bf16","sft_one_bit_minus_bf16"])]
            if g.empty:continue
            limit=rounded_limit(pd.concat([g.low95,g.high95]),.5);fig,axes=plt.subplots(2,3,figsize=(6.3,4.2),sharex=True,sharey=True,layout="constrained")
            for ax,model in zip(axes.flat,MODEL_ORDER):
                h=g[g.model.eq(model)].sort_values("adapter");y=np.arange(len(h));vals=h.mean_difference.to_numpy(float);lo=h.low95.to_numpy(float);hi=h.high95.to_numpy(float);ax.axvline(0,color="#777",lw=.8)
                if len(h):ax.errorbar(vals,y,xerr=[vals-lo,hi-vals],fmt="o",capsize=2,color=MODEL_COLORS[model]);ax.set_yticks(y,[display_condition(a) for a in h.adapter]);ax.set_title(display_model(model));ax.set_xlim(-limit,limit);ax.tick_params(labelbottom=True)
            for ax in axes[1]:ax.set_xlabel("One-bit minus BF16 (judge score)")
            fig.suptitle(f"{display_field(field)}: deployment-stack contrasts")
            self.save(f"03C_{key}_deployment_stack_effects.png",fig,g,"treatment_contrasts.csv","Points are paired representation contrasts with 95% cluster-bootstrap intervals.")

    def specificity(self):
        frame=_num(self.read("adapter_specificity_profiles.csv"),["mean_difference","low95","high95"])
        if frame.empty:return
        for model in MODEL_ORDER:
            g=frame[(frame.model.eq(model))&(frame.population.eq("evaluation_285"))&(frame.dimension.isin(FIELD_ORDER))]
            if g.empty:continue
            limit=rounded_limit(pd.concat([g.low95,g.high95]),.5);fig,ax=plt.subplots(figsize=(6.3,4.1),layout="constrained");y=np.arange(7);colors=dict(zip(["technique","explanation","joint","full"],["#0072B2","#E69F00","#009E73","#D55E00"]))
            series=[(a,c) for a in ["technique","explanation","joint","full"] for c in ["bf16_sft_minus_untreated","one_bit_sft_minus_untreated"]]
            offsets=np.linspace(-.28,.28,len(series))
            for off,(adapter,contrast) in zip(offsets,series):
                h=g[(g.adapter.eq(adapter))&(g.contrast.eq(contrast))].drop_duplicates("dimension").set_index("dimension").reindex(FIELD_ORDER);vals=h.mean_difference.to_numpy(float);lo=h.low95.to_numpy(float);hi=h.high95.to_numpy(float);marker="o" if contrast.startswith("bf16") else "s";face=colors[adapter] if marker=="o" else "white";ax.errorbar(vals,y+off,xerr=[vals-lo,hi-vals],fmt=marker,ms=3.5,mfc=face,mec=colors[adapter],capsize=1.5,color=colors[adapter],label=f"{display_condition(adapter)} · {'BF16' if marker=='o' else 'One-bit'}")
            ax.axvline(0,color="#777",lw=.8);ax.set_yticks(y,[FIELD_NAMES[f] for f in FIELD_ORDER]);ax.set_xlim(-limit,limit);ax.set_xlabel("Adapter minus untreated (judge score)");ax.set_title(f"{display_model(model)}: adapter-specificity profile")
            fig.legend(loc="lower center",bbox_to_anchor=(.5,-.01),ncol=4);fig.subplots_adjust(bottom=.2)
            self.save(f"04C_{model}_adapter_specificity_profiles.png",fig,g,"adapter_specificity_profiles.csv","Offset points show the four adapter contrasts across all seven judge fields; capped bars are 95% intervals.")

    def retention(self):
        frame=_num(self.read("retention_index.csv"),["retention_index","retention_low95","retention_high95","deployment_loss","deployment_loss_low95","deployment_loss_high95","retained_gap","retained_gap_low95","retained_gap_high95","bf16_untreated_minus_modal"])
        if frame.empty:return
        for field in FIELD_ORDER:
            g=frame[frame.dimension.eq(field)].copy()
            if g.empty:continue
            fig,axes=plt.subplots(1,4,figsize=(6.3,3.8),sharey=True,layout="constrained");y=np.arange(len(g));models=[display_model(m) for m in g.model]
            specs=[("retention_index","retention_low95","retention_high95","Retention index","R"),("deployment_loss","deployment_loss_low95","deployment_loss_high95","Deployment loss","S − Q (score)"),("retained_gap","retained_gap_low95","retained_gap_high95","Retained gap","Q − M (score)"),("bf16_untreated_minus_modal",None,None,"Untreated context","U − M (score)")]
            for ax,(col,lo,hi,title,xlabel) in zip(axes,specs):
                vals=pd.to_numeric(g[col],errors="coerce").to_numpy(float);ax.axvline(1 if col=="retention_index" else 0,color="#777",ls="--",lw=.8)
                if lo:
                    lows=pd.to_numeric(g[lo],errors="coerce").to_numpy(float);highs=pd.to_numeric(g[hi],errors="coerce").to_numpy(float);ax.errorbar(vals,y,xerr=[vals-lows,highs-vals],fmt="o",capsize=2,color="#0072B2")
                else:ax.plot(vals,y,"o",color="#0072B2")
                for i,v in enumerate(vals):
                    if not np.isfinite(v):ax.plot(0,i,marker="x",color="#777")
                ax.set_title(title);ax.set_xlabel(xlabel)
            axes[0].set_yticks(y,models);axes[0].invert_yaxis();handles=[Line2D([0],[0],marker="x",color="#777",lw=0,label="× not estimable"),Line2D([0],[0],ls="--",color="#777",label="Dashed line: R = 1 (full retention)")]
            fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,-.01),ncol=2);fig.subplots_adjust(bottom=.2)
            self.save(f"19B_{field.removeprefix('score_')}_retention_and_gap_decomposition.png",fig,g,"retention_index.csv","Four panels decompose retention relative to the modal reference; crosses are not estimable.")

    def prompt_length(self):
        frame=_num(self.read("prompt_length_and_long_tail.csv"),["n"]+["mean_"+f.removeprefix("score_") for f in FIELD_ORDER])
        if frame.empty:return
        g=frame[(frame.population.eq("all_990"))&(frame.condition.eq("untreated"))&frame.model.isin(MODEL_ORDER)].copy()
        if g.empty:return
        buckets=["<=2048","2049-4096","4097-8192",">8192"];labels=["≤2,048","2,049–4,096","4,097–8,192",">8,192"]
        fig,axes=plt.subplots(2,3,figsize=(6.3,4.8),sharex=True,layout="constrained")
        field="mean_explanation_grounding"
        for ax,model in zip(axes.flat,MODEL_ORDER):
            for rep,colour,ls in [("BF16 HF",BF16_COLOR,"-"),("one-bit frontier",ONE_BIT_COLOR,"--")]:
                h=g[(g.model.eq(model))&(g.representation.eq(rep))].set_index("prompt_length_bucket").reindex(buckets);vals=pd.to_numeric(h[field],errors="coerce");ax.plot(range(4),vals,marker="o",ms=3,color=colour,ls=ls,label=display_representation(rep))
            counts=g[g.model.eq(model)].groupby("prompt_length_bucket").n.max().reindex(buckets,fill_value=0);ax.set_xticks(range(4),[f"{lab}\nn={int(n)}" for lab,n in zip(labels,counts)]);ax.set_title(display_model(model));ax.set_ylim(-.02,4)
        for ax in axes[:,0]:ax.set_ylabel("Mean explanation-grounding score")
        for ax in axes[1]:ax.set_xlabel("Prompt length (model tokens)")
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.5,-.01),ncol=2);fig.subplots_adjust(bottom=.18)
        self.save("18C_990_untreated_scores_by_prompt_length.png",fig,g,"prompt_length_and_long_tail.csv","Lines compare representations within model; tick labels give prompt counts in each frozen length bucket.")

    def self_evaluation(self):
        frame=_num(self.read("haiku_self_evaluation_summary.csv"),["mean","score_0_n","score_1_n","score_2_n","score_3_n","score_4_n"])
        if frame.empty:return
        frame=frame.set_index("dimension").reindex(FIELD_ORDER).reset_index();fig,axes=plt.subplots(1,2,figsize=(6.3,3.4),sharey=True,layout="constrained")
        axes[0].barh(range(7),frame["mean"],color="#7B3294");axes[0].set_yticks(range(7),[FIELD_NAMES[f] for f in FIELD_ORDER]);axes[0].invert_yaxis();axes[0].set_xlim(0,4);axes[0].set_title("Mean self-evaluation score");axes[0].set_xlabel("Mean score")
        left=np.zeros(7)
        total=frame[[f"score_{i}_n" for i in range(5)]].sum(axis=1).replace(0,np.nan)
        for score,color in enumerate(ORDINAL_COLORS):vals=frame[f"score_{score}_n"]/total;axes[1].barh(range(7),vals,left=left,color=color,label=str(score));left+=vals.fillna(0).to_numpy()
        axes[1].set_xlim(0,1);axes[1].set_title("Ordinal score distribution");axes[1].set_xlabel("Share of scorable rows");axes[1].tick_params(labelleft=False)
        fig.legend(title="Score",loc="lower center",bbox_to_anchor=(.5,-.01),ncol=5);fig.subplots_adjust(bottom=.2)
        self.save("12D_haiku_self_evaluation_diagnostic.png",fig,frame,"haiku_self_evaluation_summary.csv","Left: field means. Right: the corresponding ordinal distributions.")

    def compute_figures(self):
        training=_num(self.read("compute_sft_training_report.csv"),["gpu_runtime_session_hours"])
        if not training.empty:
            g=training.sort_values(["model","condition_short"]);labels=[f"{display_model(m)} · {display_condition(c)}" for m,c in zip(g.model,g.condition_short)]
            fig,ax=plt.subplots(figsize=(6.3,max(3,.2*len(g)+1.1)),layout="constrained");ax.barh(range(len(g)),g.gpu_runtime_session_hours,color=[MODEL_COLORS.get(m,"#777") for m in g.model]);ax.set_yticks(range(len(g)),labels);ax.invert_yaxis();ax.set_xlabel("GPU session hours");ax.set_title("SFT resource time by model and adapter")
            self.save("20_training_resource_hours.png",fig,g,"compute_sft_training_report.csv","Bar length is captured GPU-runtime session time; colours identify model family.")
        quant=_num(self.read("compute_quantization_report.csv"),["resource_wall_seconds"]);art=_num(self.read("quantized_artifact_inventory.csv"),["artifact_gib"])
        if not quant.empty or not art.empty:
            keys=["model","condition_short"] if "condition_short" in art else ["model"]
            q=quant.copy()
            if not art.empty:
                if "condition" in art and "condition_short" not in art:art["condition_short"]=art.condition
                q=q.merge(art[[c for c in ["model","condition_short","artifact_gib"] if c in art]],on=[c for c in keys if c in q and c in art],how="outer")
            q["label"]=[f"{display_model(m)} · {display_condition(c)}" for m,c in zip(q.model,q.get("condition_short",pd.Series("",index=q.index)))]
            q=q.sort_values("label");fig,axes=plt.subplots(1,2,figsize=(6.3,max(3,.2*len(q)+1.2)),sharey=True,layout="constrained")
            axes[0].barh(range(len(q)),pd.to_numeric(q.resource_wall_seconds,errors="coerce")/3600,color=[MODEL_COLORS.get(m,"#777") for m in q.model]);axes[1].barh(range(len(q)),pd.to_numeric(q.artifact_gib,errors="coerce"),color=[MODEL_COLORS.get(m,"#777") for m in q.model]);axes[0].set_yticks(range(len(q)),q.label);axes[0].invert_yaxis();axes[1].tick_params(labelleft=False);axes[0].set(title="Preparation time",xlabel="Session hours");axes[1].set(title="Saved artifact size",xlabel="GGUF GiB");fig.suptitle("One-bit artifact preparation")
            self.save("22_quantization_time_and_artifact_size.png",fig,q,"compute_quantization_report.csv; quantized_artifact_inventory.csv","Panels report preparation-session time and final artifact size.")
        coverage=self.read("compute_stream_resource_summary.csv")
        if coverage.empty: coverage=self.read("stream_coverage.csv")
        if not coverage.empty:
            coverage=_num(coverage,["completed","context_unsupported","retryable_error","other_status","not_reached","intended_rows"]);g=coverage.copy();g=g.sort_values(["model","condition_short","representation_group"]);g["label"]=[f"{display_condition(c)} · {display_representation(r)}" for c,r in zip(g.condition_short,g.representation_group)]
            fig,ax=plt.subplots(figsize=(6.3,max(4,.19*len(g)+1.2)),layout="constrained");left=np.zeros(len(g));series=[("completed","Completed","#009E73"),("context_unsupported","Context unsupported","#E69F00"),("retryable_error","Retryable error","#D55E00"),("other_status","Other status","#888"),("not_reached","Not reached","#ddd")]
            den=g.intended_rows.replace(0,np.nan)
            for col,label,color in series:
                vals=g[col].fillna(0)/den;ax.barh(range(len(g)),vals,left=left,color=color,label=label);left+=vals.to_numpy()
            ax.set_yticks(range(len(g)),g.label);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel("Share of intended rows");ax.set_title("Response-stream coverage")
            for i in range(1,len(g)):
                if g.model.iloc[i]!=g.model.iloc[i-1]:ax.axhline(i-.5,color="#555",lw=.5)
            fig.legend(loc="lower center",bbox_to_anchor=(.5,-.005),ncol=3);fig.subplots_adjust(bottom=.14)
            self.save("26A_stream_coverage.png",fig,g,"compute_stream_resource_summary.csv","Every segment uses the intended stream denominator; rows are grouped by model and treatment.")
        validation=_num(self.read("runtime_reconstruction_validation.csv"),["observed_inference_session_hours","slot_adjusted_success_hours"])
        if not validation.empty:
            fig,ax=plt.subplots(figsize=(6.3,4),layout="constrained");markers={"untreated":"o","technique":"^","explanation":"v","joint":"D","full":"s"}
            for _,r in validation.iterrows():ax.scatter(r.observed_inference_session_hours,r.slot_adjusted_success_hours,color=MODEL_COLORS.get(r.model,"#777"),marker=markers.get(r.condition_short,"o"),s=28)
            lim=np.nanmax([validation.observed_inference_session_hours.max(),validation.slot_adjusted_success_hours.max()]);ax.plot([0,lim],[0,lim],color="#777",ls="--",lw=.8);ax.set(xlabel="Observed inference session hours",ylabel="Slot-adjusted request hours",title="Runtime reconstruction validation")
            out=validation.assign(abs_gap=(validation.observed_inference_session_hours-validation.slot_adjusted_success_hours).abs()).nlargest(2,"abs_gap")
            for _,r in out.iterrows():ax.annotate(f"{display_model(r.model)} · {display_condition(r.condition_short)}",(r.observed_inference_session_hours,r.slot_adjusted_success_hours),xytext=(4,4),textcoords="offset points",fontsize=8)
            handles=[Line2D([0],[0],marker=m,color="#555",lw=0,label=display_condition(c)) for c,m in markers.items()];fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,-.01),ncol=3);fig.subplots_adjust(bottom=.2)
            self.save("37A_runtime_reconstruction_validation.png",fig,validation,"runtime_reconstruction_validation.csv","The diagonal is equality; marker shape identifies condition and colour identifies model.")
        recon=_num(self.read("runtime_reconciliation_by_stream.csv"),["selected_inference_low_hours","selected_inference_high_hours","observed_inference_session_hours"])
        if not recon.empty:
            recon=recon.sort_values("selected_inference_high_hours");fig,ax=plt.subplots(figsize=(6.3,max(4,.2*len(recon)+1.2)),layout="constrained");y=np.arange(len(recon));ax.hlines(y,recon.selected_inference_low_hours,recon.selected_inference_high_hours,lw=3,color="#0072B2");ax.plot(recon.selected_inference_low_hours,y,"|",ms=7,color="#0072B2");ax.plot(recon.selected_inference_high_hours,y,"|",ms=7,color="#0072B2");ax.plot(recon.observed_inference_session_hours,y,"o",ms=3,color="#D55E00",label="Observed session time");ax.set_yticks(y,[f"{display_model(m)} · {display_condition(c)} · {display_representation(r)}" for m,c,r in zip(recon.model,recon.condition_short,recon.representation_group)]);ax.set_xlabel("Inference hours");ax.set_title("Reconciled inference time by stream");fig.legend(loc="lower center",bbox_to_anchor=(.5,-.005));fig.subplots_adjust(bottom=.12)
            self.save("37B_runtime_reconciliation_by_stream.png",fig,recon,"runtime_reconciliation_by_stream.csv","Thick bars are reconstructed ranges; orange points are observed session hours when available.")

    def compute_redesigns(self):
        frontier=_num(self.read("compression_performance_frontier.csv"),["efficiency_seconds_per_completed_row_point","efficiency_seconds_per_completed_row_low","efficiency_seconds_per_completed_row_high"])
        scores=_num(self.read("model_scores_by_dimension.csv"),["mean"])
        if len(frontier) and len(scores):
            if "efficiency_seconds_per_completed_row_point" not in frontier:
                efficiency=_num(self.read("deployment_stack_comparisons.csv"),["efficiency_seconds_per_completed_row_point","efficiency_seconds_per_completed_row_low","efficiency_seconds_per_completed_row_high"])
                keep=["population","model","condition_short","representation_group","efficiency_seconds_per_completed_row_point","efficiency_seconds_per_completed_row_low","efficiency_seconds_per_completed_row_high"]
                frontier=frontier.merge(efficiency[[c for c in keep if c in efficiency]].drop_duplicates(keep[:4]),on=keep[:4],how="left")
            f=frontier[(frontier.population.eq("evaluation_285"))&(frontier.condition_short.isin(["untreated","full"]))].merge(scores[["population","model","condition_short","representation_group","dimension","mean"]],on=["population","model","condition_short","representation_group"],how="left")
            fields=["score_technique_identification","score_explanation_grounding","score_instruction_adherence"];fig,axes=plt.subplots(1,3,figsize=(6.3,2.9),layout="constrained")
            for ax,field in zip(axes,fields):
                g=f[f.dimension.eq(field)]
                for _,r in g.iterrows():
                    x=r.efficiency_seconds_per_completed_row_point;color=MODEL_COLORS.get(r.model,"#777");marker="o" if r.condition_short=="untreated" else "s";face=color if r.representation_group=="BF16 HF" else "white"
                    ax.errorbar(x,r["mean"],xerr=[[max(0,x-r.efficiency_seconds_per_completed_row_low)],[max(0,r.efficiency_seconds_per_completed_row_high-x)]],fmt=marker,mfc=face,mec=color,ecolor=color,ms=4,capsize=1.5)
                ax.set_xscale("log");ax.set_title(FIELD_ABBR[field]);ax.set_xlabel("Seconds per completed answer");ax.set_ylim(-.02,4)
            axes[0].set_ylabel("Mean judge score")
            self.save("35_compression_performance_frontier.png",fig,f,"compression_performance_frontier.csv; model_scores_by_dimension.csv","Colour identifies model; fill identifies representation; shape identifies untreated or SFT-full.")
        stack=_num(self.read("deployment_stack_comparisons.csv"),["efficiency_seconds_per_completed_row_point","selected_inference_high_seconds","completed"])
        if len(stack):
            g=stack[(stack.population.eq("evaluation_285"))&(stack.condition_short.isin(["untreated","full"]))];fig,axes=plt.subplots(1,2,figsize=(6.3,3.2),layout="constrained")
            for model in MODEL_ORDER:
                for condition in ["untreated","full"]:
                    h=g[(g.model.eq(model))&(g.condition_short.eq(condition))]
                    if len(h)<2:continue
                    h=h.sort_values("representation_group");same=len(set(h.gpu_types.astype(str)))==1;color="#0072B2" if same else "#999";axes[0].plot(range(2),h.efficiency_seconds_per_completed_row_point,"-o",color=color,ms=3);axes[1].plot(range(2),h.selected_inference_high_seconds/3600,"-o",color=color,ms=3)
            for ax,title,ylabel in zip(axes,["Per-answer latency","Reconciled inference time"],["Seconds per completed answer","Hours"]):ax.set_xticks([0,1],["BF16","One-bit"]);ax.set_title(title);ax.set_ylabel(ylabel)
            fig.suptitle("Deployment-stack resource changes")
            self.save("31_deployment_stack_resource_changes.png",fig,g,"deployment_stack_comparisons.csv","Blue lines are same-GPU comparisons; grey lines have mixed GPU types. Whole-device memory is omitted.")
        eff=_num(self.read("sft_quantization_efficiency.csv"),["comparison_reconciled_inference_hours","mean_difference"])
        if len(eff):
            contrasts=eff.contrast.drop_duplicates().tolist();fig,axes=plt.subplots(1,len(contrasts),figsize=(6.3,max(2.8,2.6)),sharey=True,squeeze=False,layout="constrained")
            for ax,contrast in zip(axes.flat,contrasts):
                g=eff[eff.contrast.eq(contrast)]
                for field,marker in zip(FIELD_ORDER,["o","s","^","v","D","P","X"]):
                    h=g[g.dimension.eq(field)];ax.scatter(h.comparison_reconciled_inference_hours,h.mean_difference,color=[MODEL_COLORS.get(m,"#777") for m in h.model],marker=marker,s=18,label=FIELD_ABBR[field])
                ax.axhline(0,color="#777",lw=.8);ax.set_title(_sentence(contrast));ax.set_xlabel("Reconciled inference hours")
            axes.flat[0].set_ylabel("Paired judge-score difference");handles=[Line2D([0],[0],marker=m,color="#555",lw=0,label=FIELD_ABBR[f]) for f,m in zip(FIELD_ORDER,["o","s","^","v","D","P","X"])];fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,-.005),ncol=4);fig.subplots_adjust(bottom=.22)
            self.save("32_sft_quantization_resource_outcomes.png",fig,eff,"sft_quantization_efficiency.csv","Panels separate contrast types; marker shape identifies judge field and colour identifies model.")
        amort=_num(self.read("preparation_amortization.csv"),["break_even_rows"])
        if len(amort):
            fig,ax=plt.subplots(figsize=(6.3,3.6),layout="constrained");eligible=amort[np.isfinite(amort.break_even_rows)];excluded=amort[~np.isfinite(amort.break_even_rows)]
            if len(eligible):ax.barh(range(len(eligible)),eligible.break_even_rows,color=[MODEL_COLORS.get(m,"#777") for m in eligible.model]);ax.set_yticks(range(len(eligible)),[f"{display_model(m)} · {display_condition(c)}" for m,c in zip(eligible.model,eligible.condition)]);ax.invert_yaxis()
            ax.set_xlabel("Completed rows required to amortize preparation");ax.set_title("One-bit preparation amortization")
            if len(excluded):ax.text(.99,.02,"Excluded: "+"; ".join(f"{display_model(m)} ({s})" for m,s in zip(excluded.model,excluded.status)),transform=ax.transAxes,ha="right",va="bottom",fontsize=8,wrap=True)
            self.save("36_preparation_amortization.png",fig,amort,"preparation_amortization.csv","Only same-GPU streams with robust per-row speed savings receive a break-even bar; exclusions are named.")


    def run_all(self):
        run_start=pd.Timestamp.now(tz="UTC")
        for method in [self.split_balance,self.training_subset,self.training_loss,self.reference_ladder_all,self.rank_all,self.rank_figures,
                       self.score_distributions,self.score_overviews,self.concordance,self.attack_heatmaps,self.failures,
                       self.stopping,self.untreated_deployment_990,self.contrast_figures,self.specificity,self.retention,self.prompt_length,
                       self.self_evaluation,self.qwen_size,self.compute_figures,self.compute_redesigns]:
            method()
        sizes=pd.DataFrame(self.sizes);sizes.to_csv(self.tables/"figure_regeneration_sizes.csv",index=False,encoding="utf-8-sig")
        report=write_qa_outputs(self.root,self.regenerated,self.qa)
        return {"run_started_utc":str(run_start),"figures_regenerated":len(self.regenerated),"qa_rows":len(report),
                "qa_failures":int((report.message!="No QA failures").sum()),"sizes_csv":str(self.tables/"figure_regeneration_sizes.csv")}


def regenerate_notebook10_figures(root):
    return Notebook10FigureFixes(root).run_all()




