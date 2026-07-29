import ast
from pathlib import Path


CLIENT_PATH = Path(__file__).resolve().parents[1] / "whisperlivekit_client.py"


def _module():
    return ast.parse(CLIENT_PATH.read_text())


def test_wlk_chunk_duration_matches_reference_client_default():
    module = _module()
    assignments = {
        target.id: node.value
        for node in module.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        for target in [node.target]
    }
    chunk_duration = assignments["_SEND_CHUNK_DURATION_SEC"]
    assert isinstance(chunk_duration, ast.Constant)
    assert chunk_duration.value == 0.5


def test_wlk_send_loop_waits_for_server_config_before_audio():
    module = _module()
    client_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "WhisperLiveKitClient"
    )
    send_loop = next(
        node
        for node in client_class.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_send_loop"
    )
    awaited_calls = [
        node.value.value.func.attr
        for node in ast.walk(send_loop)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Await)
        and isinstance(node.value.value, ast.Call)
        and isinstance(node.value.value.func, ast.Attribute)
        and isinstance(node.value.value.func.value, ast.Name)
        and node.value.value.func.value.id == "self"
    ]
    assert awaited_calls[0] == "_wait_for_server_config_before_audio"
