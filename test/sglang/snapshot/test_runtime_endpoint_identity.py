from types import SimpleNamespace
from unittest.mock import patch

from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.lang.interpreter import ProgramState


class FakeResponse:
    status_code = 200
    text = "{}"

    def __init__(self, body=None):
        self.body = body or {"text": "ok", "meta_info": {}}

    def json(self):
        return self.body


class FakeSamplingParams:
    dtype = None
    stop = ()
    regex = None

    def to_srt_kwargs(self):
        return {"max_new_tokens": 1}


class FakeStreamExecutor:
    def __init__(self, sid="runtime-sid"):
        self.sid = sid
        self.text_ = "hello"
        self.images_ = []
        self.backend = None

    def sync(self):
        pass

    def end(self):
        pass


def make_endpoint():
    endpoint = RuntimeEndpoint.__new__(RuntimeEndpoint)
    endpoint.base_url = "http://runtime.test"
    endpoint.api_key = None
    endpoint.verify = None
    return endpoint


def capture_generate_payloads(endpoint_call):
    payloads = []

    def fake_http_request(url, **kwargs):
        if url.endswith("/generate"):
            payloads.append(kwargs["json"])
        return FakeResponse()

    with patch("sglang.lang.backend.runtime_endpoint.http_request", fake_http_request):
        endpoint_call()

    return payloads


def test_generate_payload_includes_runtime_identity():
    endpoint = make_endpoint()
    stream_executor = FakeStreamExecutor("state-123")

    payloads = capture_generate_payloads(
        lambda: endpoint.generate(stream_executor, FakeSamplingParams())
    )

    assert payloads[0]["rid"] == "state-123"
    assert payloads[0]["conversation_id"] == "state-123"


def test_generate_stream_payload_includes_runtime_identity():
    endpoint = make_endpoint()
    stream_executor = FakeStreamExecutor("state-456")

    def fake_http_request(url, **kwargs):
        class StreamingResponse(FakeResponse):
            def iter_lines(self, decode_unicode=False):
                yield b'data: {"text": "x", "meta_info": {}}'
                yield b"data: [DONE]"

        assert kwargs["json"]["rid"] == "state-456"
        assert kwargs["json"]["conversation_id"] == "state-456"
        assert kwargs["json"]["stream"] is True
        return StreamingResponse()

    with patch("sglang.lang.backend.runtime_endpoint.http_request", fake_http_request):
        assert list(
            endpoint.generate_stream(stream_executor, FakeSamplingParams())
        ) == [("x", {})]


def test_lazy_generate_paths_include_runtime_identity():
    endpoint = make_endpoint()
    stream_executor = FakeStreamExecutor("state-789")

    commit_payloads = capture_generate_payloads(
        lambda: endpoint.commit_lazy_operations(stream_executor)
    )
    fill_payloads = capture_generate_payloads(
        lambda: endpoint.fill_image(stream_executor)
    )
    helper_payloads = capture_generate_payloads(
        lambda: endpoint._generate_http_request(
            stream_executor,
            {"text": stream_executor.text_, "sampling_params": {"max_new_tokens": 0}},
        )
    )

    for payload in commit_payloads + fill_payloads + helper_payloads:
        assert payload["rid"] == "state-789"
        assert payload["conversation_id"] == "state-789"


def test_cache_prefix_uses_explicit_runtime_identity():
    endpoint = make_endpoint()

    payloads = capture_generate_payloads(lambda: endpoint.cache_prefix("prefix"))

    assert payloads[0]["rid"].startswith("runtime-cache-prefix-")
    assert payloads[0]["conversation_id"] == payloads[0]["rid"]


def test_program_state_snapshot_uses_stream_executor_identity():
    stream_executor = FakeStreamExecutor("state-snapshot")
    backend = SimpleNamespace()
    backend.save_snapshot = lambda **kwargs: kwargs
    stream_executor.backend = backend
    state = ProgramState(stream_executor)

    result = state.save_snapshot(snapshot_id="snapshot-1")

    assert result["rid"] == "state-snapshot"
    assert result["conversation_id"] == "state-snapshot"
