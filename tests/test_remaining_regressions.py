import math
from dataclasses import replace
import numpy as np

import config
from batch_transcription import _acoustic_evidence, _acoustic_only_events, _local_baseline_rms
from confidence_calibration import CalibrationExample, LogisticConfidenceCalibrator, apply_calibration
from evaluate_events import EvalEvent, evaluate
from person2_module import Person2Config, analyze_person1_transcript
from qwen_person3 import FinalBehaviourResult, deduplicate_final_results
from acoustic_vocalization_detector import detect_acoustic_vocalization


def _final(support, score, severity='Moderate', explanation='supported reason', ids=None):
    return FinalBehaviourResult('Negativism',0,1,support=='supported',severity,score,'evidence',explanation,'Negativism',0.8,'p2','tx',evidence_segment_ids=ids or ['s1'],support=support,model_support_score=score)


def test_qwen_support_precedence_and_evidence_dedupe():
    merged = deduplicate_final_results([_final('unsupported', .95, 'Insufficient','unsupported'), _final('supported', .80, ids=['s1','s2'])])[0]
    assert merged.support == 'supported' and merged.validated is True and merged.severity == 'Moderate'
    assert merged.model_support_score == .80
    merged = deduplicate_final_results([_final('unsupported', .95, 'Insufficient'), _final('insufficient', .90, 'Insufficient','missing')])[0]
    assert merged.support == 'insufficient' and merged.validated is False
    merged = deduplicate_final_results([_final('supported', .80, ids=['s1','s1']), _final('supported', .90, 'High','better', ids=['s1','s2'])])[0]
    assert merged.support == 'supported' and merged.model_support_score == .90 and merged.explanation == 'better'
    assert merged.evidence_segment_ids == ['s1','s2']


def test_linguistic_urgency_does_not_leak_to_acoustic_segment():
    result = analyze_person1_transcript([
        {'id':'a','start':0,'end':1,'text':'Help me now!','confidence':.9},
        {'id':'b','start':2,'end':3,'text':'The weather is lovely today.','confidence':.9,'acoustic':{'available':True,'agitation_score':.55,'scream_score':.1,'relative_energy':2,'burst_score':.1}},
    ], settings=Person2Config(embedding_backend='hashing', target_speaker_id=None))
    assert not any(b.behaviour == 'Distressed/urgent verbalization' and b.source_segment_ids == ['b'] for b in result.behaviours)


def test_speaker_filtering_target_and_inverse():
    segs=[{'id':'s1','start':0,'end':1,'text':"I won't take my medicine.",'speaker_id':1,'speaker_label':'Speaker 1'}, {'id':'s2','start':2,'end':3,'text':'The weather is lovely today.','speaker_id':2,'speaker_label':'Speaker 2'}]
    r1=analyze_person1_transcript(segs, settings=Person2Config(embedding_backend='hashing', target_speaker_id='1'))
    assert any(b.behaviour=='Negativism' and b.speaker_id==1 for b in r1.behaviours)
    assert all(b.speaker_id != 2 for b in r1.behaviours if b.score_type.startswith('heuristic_linguistic'))
    r2=analyze_person1_transcript(segs, settings=Person2Config(embedding_backend='hashing', target_speaker_id='2'))
    assert not any(b.behaviour=='Negativism' for b in r2.behaviours)


def test_isolated_request_not_constant_but_repeated_requests_are():
    one=analyze_person1_transcript([{'id':'h1','start':0,'end':1,'text':'Please help me.'}], settings=Person2Config(embedding_backend='hashing'))
    assert not any(b.behaviour=='Constant unwarranted requests for attention/help' for b in one.behaviours)
    many=analyze_person1_transcript([{'id':'h1','start':0,'end':1,'text':'Please help me.'},{'id':'x','start':5,'end':6,'text':'The light is on.'},{'id':'h2','start':10,'end':11,'text':'Can someone help me?'},{'id':'h3','start':20,'end':21,'text':'Please come help me.'}], settings=Person2Config(embedding_backend='hashing', semantic_similarity_threshold=.0))
    events=[b for b in many.behaviours if b.behaviour=='Constant unwarranted requests for attention/help']
    assert events and len(events[0].source_segment_ids) >= 2 and 'x' not in events[0].source_segment_ids


def test_negativism_paraphrase_and_negative_sentiment_not_refusal():
    yes=analyze_person1_transcript([{'id':'n','start':0,'end':1,'text':"You're not making me do that."}], settings=Person2Config(embedding_backend='hashing'))
    assert any(b.behaviour=='Negativism' for b in yes.behaviours)
    no=analyze_person1_transcript([{'id':'c','start':0,'end':1,'text':'This food tastes awful.'}], settings=Person2Config(embedding_backend='hashing'))
    assert not any(b.behaviour=='Negativism' for b in no.behaviours)


def test_preceding_baseline_gain_transition_and_long_episode():
    sr=16000
    normal=np.ones(sr*20,dtype=np.float32)*0.03
    gain=np.ones(sr*30,dtype=np.float32)*0.18
    audio=np.concatenate([normal,gain])
    base=_local_baseline_rms(audio,sr,25*sr,26*sr,fallback=.03,context_seconds=20)
    ev=_acoustic_evidence(audio[25*sr:26*sr],sr,base)
    assert ev['available'] and ev['rms_mean'] >= .17
    startbase=_local_baseline_rms(audio,sr,0,sr,fallback=.03,context_seconds=20)
    assert startbase >= config.BATCH_LOCAL_BASELINE_MIN_RMS
    shout=np.concatenate([np.ones(sr*20,dtype=np.float32)*.03, np.sin(2*np.pi*180*np.arange(sr*8)/sr).astype(np.float32)*.5])
    base2=_local_baseline_rms(shout,sr,22*sr,28*sr,fallback=.03,context_seconds=20)
    assert base2 < .1


def test_scream_aggregation_retains_strongest():
    sr=16000
    audio=np.sin(2*np.pi*180*np.arange(sr*3)/sr).astype(np.float32)*.95
    events=_acoustic_only_events(audio,sr,baseline_rms=.03)
    assert events and events[0].acoustic['max_scream_score'] >= events[0].acoustic['scream_score']


def test_strange_noise_detector_silence_speech_impact_moan():
    sr=16000; t=np.arange(sr)/sr
    assert detect_acoustic_vocalization(np.zeros(sr,dtype=np.float32),sr).score == 0
    speech_like=(np.sin(2*np.pi*180*t)+0.45*np.sin(2*np.pi*360*t)+0.25*np.sin(2*np.pi*540*t)).astype(np.float32)*.12
    impact=np.zeros(sr,dtype=np.float32); impact[100:120]=1
    moan=np.sin(2*np.pi*120*t).astype(np.float32)*.2
    assert detect_acoustic_vocalization(impact,sr).score == 0
    assert detect_acoustic_vocalization(moan,sr).score > 0


def test_calibration_none_fit_serialize_and_unsupported_guard(tmp_path):
    f=_final('supported', .91)
    assert f.calibrated_confidence is None
    cal=LogisticConfidenceCalibrator().fit([CalibrationExample({'model_support_score':.1,'verifier_supported':0},0), CalibrationExample({'model_support_score':.9,'verifier_supported':1},1)], epochs=10)
    out=apply_calibration(f, cal)
    assert 0 <= out.calibrated_confidence <= 1 and out.model_support_score == .91
    path=tmp_path/'cal.json'; cal.save(path); loaded=LogisticConfidenceCalibrator.load(path)
    unsupported=apply_calibration(_final('unsupported', .99, 'Insufficient', 'no'), loaded)
    assert unsupported.calibrated_confidence == 0.0


def test_evaluation_harness_cases():
    p=[EvalEvent('A',0,1),EvalEvent('A',0,1),EvalEvent('B',0,1)]
    r=[EvalEvent('A',0,1),EvalEvent('A',.4,1.4)]
    rep=evaluate(p,r,iou_threshold=.3)
    assert rep['TP']==2 and rep['FP']==1 and rep['FN']==0
    assert evaluate([],[],iou_threshold=.3)['f1']==0
    assert evaluate([EvalEvent('A',0,1)],[EvalEvent('A',1,2)],iou_threshold=.0)['TP']==1
    assert evaluate([EvalEvent('A',0,1)],[EvalEvent('B',0,1)],iou_threshold=.3)['TP']==0
