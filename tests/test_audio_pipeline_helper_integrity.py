import ast
from pathlib import Path


AUDIO_PIPELINE_PATH = Path(__file__).resolve().parents[1] / "audio_pipeline.py"


def test_audio_pipeline_private_helper_calls_are_implemented():
    module = ast.parse(AUDIO_PIPELINE_PATH.read_text())
    pipeline_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "AudioPipeline"
    )
    implemented = {
        node.name
        for node in pipeline_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    called = set()
    for node in ast.walk(pipeline_class):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != "self":
            continue
        if node.func.attr.startswith("_"):
            called.add(node.func.attr)

    assert called <= implemented
