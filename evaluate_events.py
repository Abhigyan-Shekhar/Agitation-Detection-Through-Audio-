"""Evaluate predicted CMAI events against labelled reference events.

Matching is deterministic one-to-one maximum-IoU: all same-behaviour candidate
pairs at or above the configured IoU are sorted by descending IoU, then by
prediction/reference index, and greedily accepted if neither event is already
matched.
"""
from __future__ import annotations
import argparse, csv, json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

@dataclass(frozen=True)
class EvalEvent:
    behaviour: str; start: float; end: float; speaker_id: str | None = None; severity: str | None = None; confidence: float | None = None

def load_events(path: str | Path) -> list[EvalEvent]:
    p=Path(path); rows=json.loads(p.read_text()) if p.suffix.lower()=='.json' else list(csv.DictReader(p.open()))
    return [EvalEvent(str(r['behaviour']), float(r['start']), float(r['end']), None if r.get('speaker_id') in (None,'') else str(r.get('speaker_id')), r.get('severity'), None if r.get('confidence') in (None,'') else float(r.get('confidence'))) for r in rows]

def temporal_iou(a: EvalEvent,b: EvalEvent)->float:
    inter=max(0.0,min(a.end,b.end)-max(a.start,b.start)); union=max(a.end,b.end)-min(a.start,b.start); return inter/union if union>0 else 0.0

def evaluate(predictions:list[EvalEvent], references:list[EvalEvent], *, iou_threshold:float=0.3)->dict[str,Any]:
    pairs=[]
    for pi,p in enumerate(predictions):
        for ri,r in enumerate(references):
            if p.behaviour==r.behaviour:
                iou=temporal_iou(p,r)
                if iou>=iou_threshold: pairs.append((-iou,pi,ri))
    matched_p=set(); matched_r=set(); matches=[]
    for neg,pi,ri in sorted(pairs):
        if pi not in matched_p and ri not in matched_r:
            matched_p.add(pi); matched_r.add(ri); matches.append((pi,ri,-neg))
    tp=len(matches); fp=len(predictions)-tp; fn=len(references)-tp
    def prf(tp,fp,fn):
        prec=tp/(tp+fp) if tp+fp else 0.0; rec=tp/(tp+fn) if tp+fn else 0.0; f1=2*prec*rec/(prec+rec) if prec+rec else 0.0; return prec,rec,f1
    precision,recall,f1=prf(tp,fp,fn); ious=[m[2] for m in matches]
    behaviours=sorted({e.behaviour for e in predictions+references}); per={}
    for b in behaviours:
        mt=sum(1 for pi,ri,_ in matches if predictions[pi].behaviour==b); pp=sum(1 for e in predictions if e.behaviour==b); rr=sum(1 for e in references if e.behaviour==b); p,r,f=prf(mt,pp-mt,rr-mt); per[b]={"TP":mt,"FP":pp-mt,"FN":rr-mt,"precision":p,"recall":r,"f1":f}
    details=[]
    for pi,ri,iou in matches:
        p=predictions[pi]; r=references[ri]; details.append({"prediction":asdict(p),"matched_reference":asdict(r),"match_status":"TP","iou":iou,"onset_error":p.start-r.start,"offset_error":p.end-r.end})
    for pi,p in enumerate(predictions):
        if pi not in matched_p: details.append({"prediction":asdict(p),"matched_reference":None,"match_status":"FP","iou":0.0,"onset_error":None,"offset_error":None})
    for ri,r in enumerate(references):
        if ri not in matched_r: details.append({"prediction":None,"matched_reference":asdict(r),"match_status":"FN","iou":0.0,"onset_error":None,"offset_error":None})
    macro_f1=mean([v['f1'] for v in per.values()]) if per else 0.0
    return {"TP":tp,"FP":fp,"FN":fn,"precision":precision,"recall":recall,"f1":f1,"micro":{"precision":precision,"recall":recall,"f1":f1},"macro":{"f1":macro_f1},"per_behaviour":per,"mean_matched_iou":mean(ious) if ious else 0.0,"median_matched_iou":median(ious) if ious else 0.0,"details":details}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--predictions',required=True); ap.add_argument('--ground-truth',required=True); ap.add_argument('--iou-threshold',type=float,default=0.3); ap.add_argument('--output')
    args=ap.parse_args(); report=evaluate(load_events(args.predictions),load_events(args.ground_truth),iou_threshold=args.iou_threshold); text=json.dumps(report,indent=2)
    Path(args.output).write_text(text) if args.output else print(text)
if __name__=='__main__': main()
