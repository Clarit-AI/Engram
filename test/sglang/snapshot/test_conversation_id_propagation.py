"""
Regression test for conversation_id propagation through Req construction.

This test is a CANARY: it verifies that the Req class accepts and stores
a conversation_id parameter. The real protection against upstream merges
clobbering the conversation_id= kwarg in handle_generate_request comes
from the protected-paths policy flag on scheduler.py
(.engram/policy/protected-paths.json).

Background: The upstream sync has broken conversation_id propagation
twice (PR #24 and upstream-sync/2026-04-07). Each time, the Req()
constructor call in handle_generate_request was rewritten by upstream
without the conversation_id= kwarg, causing snapshots to be keyed by
request UUID instead of the user-provided conversation_id.
"""

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams


class TestConversationIdPropagation:
    """Verify conversation_id flows through Req construction."""

    def test_req_accepts_conversation_id(self):
        """Req constructor should accept and store conversation_id."""
        req = Req(
            rid="test-rid-001",
            origin_input_text="hello",
            origin_input_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
            conversation_id="conv-abc-123",
        )
        assert req.conversation_id == "conv-abc-123"

    def test_req_conversation_id_defaults_to_none(self):
        """Req.conversation_id should default to None when not provided."""
        req = Req(
            rid="test-rid-002",
            origin_input_text="hello",
            origin_input_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
        )
        assert req.conversation_id is None

    def test_getattr_pattern_with_conversation_id(self):
        """The getattr(recv_req, 'conversation_id', None) pattern used in
        handle_generate_request should correctly extract conversation_id."""

        class MockTokenizedReq:
            """Mimics TokenizedGenerateReqInput with conversation_id."""

            def __init__(self, conversation_id=None):
                self.conversation_id = conversation_id

        # With conversation_id set
        mock_with = MockTokenizedReq(conversation_id="user-conv-456")
        extracted = getattr(mock_with, "conversation_id", None)
        assert extracted == "user-conv-456"

        req = Req(
            rid="test-rid-003",
            origin_input_text="hello",
            origin_input_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
            conversation_id=extracted,
        )
        assert req.conversation_id == "user-conv-456"

    def test_getattr_pattern_without_conversation_id(self):
        """getattr fallback should return None for objects without
        conversation_id attribute."""

        class MockTokenizedReqNoConvId:
            """Mimics a TokenizedGenerateReqInput without conversation_id."""

            pass

        mock_without = MockTokenizedReqNoConvId()
        extracted = getattr(mock_without, "conversation_id", None)
        assert extracted is None

        req = Req(
            rid="test-rid-004",
            origin_input_text="hello",
            origin_input_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
            conversation_id=extracted,
        )
        assert req.conversation_id is None
