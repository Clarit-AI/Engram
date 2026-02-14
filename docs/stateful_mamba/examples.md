# Stateful Mamba Examples

Real-world examples demonstrating the snapshot system for Mamba models in SGLang.

## Table of Contents

- [Complete Examples](#complete-examples)
  - [Interactive Chatbot](#interactive-chatbot)
  - [Story Generator with Branching](#story-generator-with-branching)
  - [Document Summarization Pipeline](#document-summarization-pipeline)
  - [Code Generation with Checkpoints](#code-generation-with-checkpoints)
  - [Multi-Agent Conversation](#multi-agent-conversation)
- [Integration Examples](#integration-examples)
  - [FastAPI Server](#fastapi-server)
  - [Gradio Interface](#gradio-interface)
  - [Batch Processing](#batch-processing)
- [Advanced Patterns](#advanced-patterns)
  - [Snapshot Versioning](#snapshot-versioning)
  - [Distributed Snapshots](#distributed-snapshots)
  - [Performance Monitoring](#performance-monitoring)

## Complete Examples

### Interactive Chatbot

A complete chatbot implementation with session persistence and context management.

```python
# chatbot.py
import time
from typing import List, Optional
from dataclasses import dataclass
from sglang import Engine
from sglang.lang import function, gen
from sglang.snapshot import SnapshotManager

@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: float

class ChatSession:
    """Persistent chat session with snapshot support."""

    def __init__(self, session_id: str, engine: Engine):
        self.session_id = session_id
        self.engine = engine
        self.manager = SnapshotManager(engine)
        self.current_snapshot: Optional[str] = None
        self.history: List[ChatMessage] = []

    def send_message(self, user_message: str) -> str:
        """Send a message and get assistant response."""
        # Add user message to history
        self.history.append(ChatMessage(
            role="user",
            content=user_message,
            timestamp=time.time()
        ))

        # Generate response
        @function
        def chat_turn(s, snapshot_id, user_msg):
            # Restore previous state if exists
            if snapshot_id:
                s.restore_snapshot(snapshot_id)

            # Add user message
            s += f"\nUser: {user_msg}\n"
            s += "Assistant: " + gen("response", max_tokens=200, temperature=0.7)

            # Save updated state
            new_snapshot = s.save_snapshot(
                metadata={
                    "session_id": self.session_id,
                    "turn": len(self.history),
                    "timestamp": time.time()
                }
            )

            return s, new_snapshot

        result, new_snapshot = chat_turn.run(
            snapshot_id=self.current_snapshot,
            user_msg=user_message,
            engine=self.engine
        )

        # Update state
        self.current_snapshot = new_snapshot
        assistant_message = result["response"]

        # Add assistant response to history
        self.history.append(ChatMessage(
            role="assistant",
            content=assistant_message,
            timestamp=time.time()
        ))

        return assistant_message

    def save_to_disk(self, path: Optional[str] = None):
        """Persist session to disk."""
        if not self.current_snapshot:
            return

        if path is None:
            path = f"./sessions/{self.session_id}.snapshot"

        self.manager.persist_snapshot(self.current_snapshot, path)

    def load_from_disk(self, path: Optional[str] = None) -> bool:
        """Load session from disk."""
        if path is None:
            path = f"./sessions/{self.session_id}.snapshot"

        try:
            self.current_snapshot = self.manager.load_snapshot(path)
            return True
        except FileNotFoundError:
            return False

    def reset(self):
        """Reset the conversation."""
        if self.current_snapshot:
            self.manager.delete_snapshot(self.current_snapshot)
        self.current_snapshot = None
        self.history = []

    def get_history(self) -> List[ChatMessage]:
        """Get conversation history."""
        return self.history.copy()

# Usage
def main():
    # Initialize engine
    engine = Engine(
        model_path="state-spaces/mamba-2.8b",
        enable_mamba_snapshots=True,
        snapshot_config={
            "enable_persistence": True,
            "storage_path": "./chat_sessions"
        }
    )

    # Create or restore session
    session = ChatSession("user_alice_001", engine)

    # Try to load previous session
    if session.load_from_disk():
        print("Restored previous session")
    else:
        print("Starting new session")

    # Interactive chat loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            # Save before exiting
            session.save_to_disk()
            print("Session saved. Goodbye!")
            break

        response = session.send_message(user_input)
        print(f"Assistant: {response}")

        # Auto-save every 5 messages
        if len(session.history) % 10 == 0:
            session.save_to_disk()
            print("[Session auto-saved]")

if __name__ == "__main__":
    main()
```

---

### Story Generator with Branching

Generate interactive stories with multiple choice branching.

```python
# story_branching.py
from typing import Dict, List, Optional
from sglang import Engine
from sglang.lang import function, gen
from sglang.snapshot import SnapshotManager

class BranchingStory:
    """Interactive story with branching narratives."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.manager = SnapshotManager(engine)
        self.checkpoints: Dict[str, str] = {}  # checkpoint_id -> snapshot_id

    def start_story(self, genre: str = "fantasy") -> str:
        """Initialize the story."""
        @function
        def story_intro(s, genre):
            s += f"Write the beginning of a {genre} story:\n\n"
            s += gen("intro", max_tokens=300, temperature=0.8)
            s += "\n\nWhat happens next?\n"

            snapshot_id = s.save_snapshot(
                metadata={"checkpoint": "intro", "genre": genre}
            )

            return s, snapshot_id

        result, snapshot_id = story_intro.run(genre=genre, engine=self.engine)
        self.checkpoints["intro"] = snapshot_id

        return result["intro"]

    def continue_story(
        self,
        checkpoint_id: str,
        choice: str,
        choice_text: Optional[str] = None
    ) -> tuple[str, str]:
        """Continue story from a checkpoint with a choice."""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found")

        snapshot_id = self.checkpoints[checkpoint_id]

        @function
        def story_branch(s, snapshot_id, choice, choice_text):
            # Restore from checkpoint
            s.restore_snapshot(snapshot_id)

            # Add choice
            if choice_text:
                s += f"\nChoice: {choice_text}\n\n"

            # Generate continuation
            s += gen(f"branch_{choice}", max_tokens=400, temperature=0.8)
            s += "\n\nWhat happens next?\n"

            # Save new checkpoint
            new_checkpoint_id = f"{checkpoint_id}_{choice}"
            new_snapshot = s.save_snapshot(
                metadata={
                    "checkpoint": new_checkpoint_id,
                    "parent": checkpoint_id,
                    "choice": choice
                }
            )

            return s, new_snapshot, new_checkpoint_id

        result, new_snapshot, new_checkpoint_id = story_branch.run(
            snapshot_id=snapshot_id,
            choice=choice,
            choice_text=choice_text,
            engine=self.engine
        )

        # Save new checkpoint
        self.checkpoints[new_checkpoint_id] = new_snapshot

        return result[f"branch_{choice}"], new_checkpoint_id

    def explore_all_branches(self, checkpoint_id: str, choices: List[str]) -> Dict[str, str]:
        """Explore multiple branches from a single checkpoint."""
        results = {}

        for choice in choices:
            continuation, new_checkpoint = self.continue_story(
                checkpoint_id,
                choice,
                choice_text=f"Option {choice}"
            )
            results[choice] = continuation

        return results

    def cleanup(self):
        """Clean up all checkpoints."""
        for snapshot_id in self.checkpoints.values():
            self.manager.delete_snapshot(snapshot_id)
        self.checkpoints.clear()

# Usage
def main():
    engine = Engine(
        model_path="state-spaces/mamba-2.8b",
        enable_mamba_snapshots=True,
        snapshot_config={"enable_cow": True}
    )

    story = BranchingStory(engine)

    # Start story
    print("=== STORY BEGINNING ===")
    intro = story.start_story(genre="science fiction")
    print(intro)

    # Branch 1: Hero accepts the mission
    print("\n=== BRANCH A: Accept Mission ===")
    branch_a, checkpoint_a = story.continue_story(
        "intro",
        "accept",
        "The hero accepts the dangerous mission"
    )
    print(branch_a)

    # Branch 2: Hero declines the mission
    print("\n=== BRANCH B: Decline Mission ===")
    branch_b, checkpoint_b = story.continue_story(
        "intro",
        "decline",
        "The hero declines and walks away"
    )
    print(branch_b)

    # Further branches from branch A
    print("\n=== EXPLORING SUB-BRANCHES FROM BRANCH A ===")
    sub_branches = story.explore_all_branches(
        checkpoint_a,
        ["stealth", "direct", "negotiate"]
    )

    for choice, text in sub_branches.items():
        print(f"\n--- Sub-branch: {choice} ---")
        print(text)

    # Cleanup
    story.cleanup()

if __name__ == "__main__":
    main()
```

---

### Document Summarization Pipeline

Multi-stage document processing with checkpoints.

```python
# document_pipeline.py
from typing import List, Dict, Optional
from dataclasses import dataclass
from sglang import Engine
from sglang.lang import function, gen
from sglang.snapshot import SnapshotManager

@dataclass
class ProcessingStage:
    stage_name: str
    snapshot_id: str
    output: str
    timestamp: float

class DocumentPipeline:
    """Multi-stage document processing with checkpoint recovery."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.manager = SnapshotManager(engine)
        self.stages: List[ProcessingStage] = []

    def process_document(self, document: str, stages: List[str]) -> Dict[str, str]:
        """Process document through multiple stages with checkpointing."""
        results = {}
        current_snapshot = None

        for i, stage_name in enumerate(stages):
            print(f"Processing stage {i+1}/{len(stages)}: {stage_name}")

            try:
                output, snapshot_id = self._process_stage(
                    document,
                    stage_name,
                    current_snapshot,
                    i
                )

                # Record stage
                self.stages.append(ProcessingStage(
                    stage_name=stage_name,
                    snapshot_id=snapshot_id,
                    output=output,
                    timestamp=time.time()
                ))

                results[stage_name] = output
                current_snapshot = snapshot_id

            except Exception as e:
                print(f"Error in stage '{stage_name}': {e}")
                # Can resume from previous checkpoint
                if i > 0:
                    print(f"Can resume from stage '{stages[i-1]}'")
                raise

        return results

    def _process_stage(
        self,
        document: str,
        stage_name: str,
        previous_snapshot: Optional[str],
        stage_num: int
    ) -> tuple[str, str]:
        """Process a single stage."""
        @function
        def stage_processor(s, doc, stage, prev_snap, num):
            if prev_snap:
                s.restore_snapshot(prev_snap)
            else:
                # Initial stage
                s += f"Document:\n{doc}\n\n"

            # Stage-specific processing
            if stage == "extract_key_points":
                s += "Extract key points:\n"
                s += gen("key_points", max_tokens=300)

            elif stage == "summarize":
                s += "\n\nSummarize:\n"
                s += gen("summary", max_tokens=200)

            elif stage == "generate_questions":
                s += "\n\nGenerate questions:\n"
                s += gen("questions", max_tokens=150)

            elif stage == "action_items":
                s += "\n\nAction items:\n"
                s += gen("actions", max_tokens=100)

            # Save checkpoint
            snapshot_id = s.save_snapshot(
                metadata={
                    "stage": stage,
                    "stage_num": num,
                    "timestamp": time.time()
                }
            )

            return s, snapshot_id

        result, snapshot_id = stage_processor.run(
            doc=document,
            stage=stage_name,
            prev_snap=previous_snapshot,
            num=stage_num,
            engine=self.engine
        )

        output = result.get(stage_name.split("_")[-1], "")
        return output, snapshot_id

    def resume_from_stage(self, stage_name: str) -> Optional[str]:
        """Resume processing from a specific stage."""
        for stage in self.stages:
            if stage.stage_name == stage_name:
                return stage.snapshot_id
        return None

    def get_stage_output(self, stage_name: str) -> Optional[str]:
        """Get output from a specific stage."""
        for stage in self.stages:
            if stage.stage_name == stage_name:
                return stage.output
        return None

    def cleanup(self):
        """Clean up all checkpoints."""
        for stage in self.stages:
            self.manager.delete_snapshot(stage.snapshot_id)
        self.stages.clear()

# Usage
import time

def main():
    engine = Engine(
        model_path="state-spaces/mamba-2.8b",
        enable_mamba_snapshots=True
    )

    pipeline = DocumentPipeline(engine)

    document = """
    The quarterly meeting discussed three main topics: product roadmap,
    hiring plans, and budget allocation. The engineering team proposed
    launching two new features next month. HR plans to hire 5 engineers
    and 2 designers. Finance allocated $500K for Q2 marketing.
    """

    # Define processing stages
    stages = [
        "extract_key_points",
        "summarize",
        "generate_questions",
        "action_items"
    ]

    # Process document
    results = pipeline.process_document(document, stages)

    # Display results
    for stage_name, output in results.items():
        print(f"\n{'='*50}")
        print(f"{stage_name.upper()}")
        print('='*50)
        print(output)

    # Example: Resume from a specific stage if needed
    # checkpoint = pipeline.resume_from_stage("summarize")
    # if checkpoint:
    #     # Continue from this checkpoint
    #     pass

    # Cleanup
    pipeline.cleanup()

if __name__ == "__main__":
    main()
```

---

### Code Generation with Checkpoints

Generate code iteratively with the ability to rollback and try different approaches.

```python
# code_generation.py
from typing import List, Dict, Optional
from sglang import Engine
from sglang.lang import function, gen
from sglang.snapshot import SnapshotManager

class CodeGenerator:
    """Iterative code generation with checkpoint rollback."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.manager = SnapshotManager(engine)
        self.checkpoints: Dict[str, str] = {}

    def generate_function(
        self,
        spec: str,
        programming_language: str = "Python"
    ) -> str:
        """Generate a function from specification."""
        @function
        def gen_function(s, spec, lang):
            s += f"Generate a {lang} function:\n"
            s += f"Specification: {spec}\n\n"
            s += f"```{lang.lower()}\n"
            s += gen("code", max_tokens=500, temperature=0.2, stop=["```"])
            s += "\n```"

            snapshot_id = s.save_snapshot(
                metadata={"stage": "function", "language": lang}
            )

            return s, snapshot_id

        result, snapshot_id = gen_function.run(
            spec=spec,
            lang=programming_language,
            engine=self.engine
        )

        self.checkpoints["function"] = snapshot_id
        return result["code"]

    def add_docstring(self, checkpoint_id: str = "function") -> str:
        """Add docstring to generated code."""
        snapshot_id = self.checkpoints.get(checkpoint_id)
        if not snapshot_id:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found")

        @function
        def add_docs(s, snapshot_id):
            s.restore_snapshot(snapshot_id)
            s += "\n\nAdd detailed docstring:\n```python\n"
            s += gen("documented_code", max_tokens=600, temperature=0.2, stop=["```"])
            s += "\n```"

            new_snapshot = s.save_snapshot(
                metadata={"stage": "documented"}
            )

            return s, new_snapshot

        result, new_snapshot = add_docs.run(
            snapshot_id=snapshot_id,
            engine=self.engine
        )

        self.checkpoints["documented"] = new_snapshot
        return result["documented_code"]

    def add_tests(self, checkpoint_id: str = "documented") -> str:
        """Generate unit tests for the code."""
        snapshot_id = self.checkpoints.get(checkpoint_id)
        if not snapshot_id:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found")

        @function
        def gen_tests(s, snapshot_id):
            s.restore_snapshot(snapshot_id)
            s += "\n\nGenerate unit tests:\n```python\n"
            s += gen("tests", max_tokens=700, temperature=0.2, stop=["```"])
            s += "\n```"

            new_snapshot = s.save_snapshot(
                metadata={"stage": "tested"}
            )

            return s, new_snapshot

        result, new_snapshot = gen_tests.run(
            snapshot_id=snapshot_id,
            engine=self.engine
        )

        self.checkpoints["tested"] = new_snapshot
        return result["tests"]

    def rollback_to(self, checkpoint_id: str) -> bool:
        """Rollback to a previous checkpoint."""
        if checkpoint_id in self.checkpoints:
            # Delete later checkpoints
            stages = ["function", "documented", "tested"]
            if checkpoint_id in stages:
                idx = stages.index(checkpoint_id)
                for stage in stages[idx+1:]:
                    if stage in self.checkpoints:
                        self.manager.delete_snapshot(self.checkpoints[stage])
                        del self.checkpoints[stage]
            return True
        return False

    def try_alternative(
        self,
        checkpoint_id: str,
        alternative_prompt: str
    ) -> str:
        """Try an alternative approach from a checkpoint."""
        snapshot_id = self.checkpoints.get(checkpoint_id)
        if not snapshot_id:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found")

        @function
        def gen_alternative(s, snapshot_id, prompt):
            s.restore_snapshot(snapshot_id)
            s += f"\n\nAlternative approach: {prompt}\n```python\n"
            s += gen("alternative", max_tokens=500, temperature=0.3, stop=["```"])
            s += "\n```"

            return s

        result = gen_alternative.run(
            snapshot_id=snapshot_id,
            prompt=alternative_prompt,
            engine=self.engine
        )

        return result["alternative"]

# Usage
def main():
    engine = Engine(
        model_path="state-spaces/mamba-2.8b",
        enable_mamba_snapshots=True
    )

    generator = CodeGenerator(engine)

    # Step 1: Generate function
    print("=== STEP 1: Generate Function ===")
    code = generator.generate_function(
        spec="Sort a list of integers using quicksort",
        programming_language="Python"
    )
    print(code)

    # Step 2: Add documentation
    print("\n=== STEP 2: Add Documentation ===")
    documented_code = generator.add_docstring()
    print(documented_code)

    # Step 3: Generate tests
    print("\n=== STEP 3: Generate Tests ===")
    tests = generator.add_tests()
    print(tests)

    # Alternative: Try different approach from function stage
    print("\n=== ALTERNATIVE: Different Algorithm ===")
    alternative = generator.try_alternative(
        "function",
        "Use merge sort instead of quicksort"
    )
    print(alternative)

    # Can also rollback
    # generator.rollback_to("function")
    # # Now can regenerate from that point with different parameters

if __name__ == "__main__":
    main()
```

---

### Multi-Agent Conversation

Simulate conversations between multiple AI agents with shared context.

```python
# multi_agent.py
from typing import List, Dict
from dataclasses import dataclass
from sglang import Engine
from sglang.lang import function, gen
from sglang.snapshot import SnapshotManager

@dataclass
class Agent:
    name: str
    role: str
    personality: str

class MultiAgentConversation:
    """Multi-agent conversation with shared context snapshots."""

    def __init__(self, engine: Engine, agents: List[Agent]):
        self.engine = engine
        self.agents = agents
        self.manager = SnapshotManager(engine)
        self.conversation_snapshot: Optional[str] = None

    def start_conversation(self, topic: str) -> str:
        """Initialize the conversation."""
        @function
        def init_conversation(s, topic, agents):
            s += f"Topic: {topic}\n\n"
            s += "Participants:\n"
            for agent in agents:
                s += f"- {agent.name} ({agent.role}): {agent.personality}\n"
            s += "\nConversation:\n"

            snapshot_id = s.save_snapshot(
                metadata={"stage": "init", "topic": topic}
            )

            return s, snapshot_id

        result, snapshot_id = init_conversation.run(
            topic=topic,
            agents=self.agents,
            engine=self.engine
        )

        self.conversation_snapshot = snapshot_id
        return result.text()

    def agent_speaks(
        self,
        agent: Agent,
        about: Optional[str] = None
    ) -> str:
        """Have an agent contribute to the conversation."""
        @function
        def speak(s, snapshot_id, agent, about):
            # Restore conversation context
            s.restore_snapshot(snapshot_id)

            # Agent speaks
            prompt = f"\n{agent.name}: "
            if about:
                prompt += f"[responding to: {about}] "

            s += prompt
            s += gen(
                f"speech_{agent.name}",
                max_tokens=150,
                temperature=0.8,
                stop=["\n\n", f"\n{a.name}:" for a in self.agents if a != agent]
            )

            # Update conversation snapshot
            new_snapshot = s.save_snapshot(
                metadata={
                    "speaker": agent.name,
                    "turn": len(self.agents)
                }
            )

            return s, new_snapshot

        result, new_snapshot = speak.run(
            snapshot_id=self.conversation_snapshot,
            agent=agent,
            about=about,
            engine=self.engine
        )

        # Update conversation state
        self.conversation_snapshot = new_snapshot

        speech_key = f"speech_{agent.name}"
        return result.get(speech_key, "")

    def run_conversation(self, num_rounds: int) -> str:
        """Run multiple rounds of conversation."""
        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1} ---")
            for agent in self.agents:
                speech = self.agent_speaks(agent)
                print(f"{agent.name}: {speech}")

        # Get full conversation
        snapshot = self.manager.get_snapshot(self.conversation_snapshot)
        return snapshot.token_text if hasattr(snapshot, 'token_text') else ""

    def branch_conversation(
        self,
        from_turn: int,
        new_topic: str
    ) -> "MultiAgentConversation":
        """Create a branch from a specific turn."""
        # This would require storing turn snapshots
        # Simplified version: create new conversation
        new_conversation = MultiAgentConversation(self.engine, self.agents)
        new_conversation.start_conversation(new_topic)
        return new_conversation

# Usage
def main():
    engine = Engine(
        model_path="state-spaces/mamba-2.8b",
        enable_mamba_snapshots=True,
        snapshot_config={"enable_cow": True}
    )

    # Define agents
    agents = [
        Agent(
            name="Alice",
            role="Optimist",
            personality="Always sees the positive side"
        ),
        Agent(
            name="Bob",
            role="Realist",
            personality="Focuses on practical considerations"
        ),
        Agent(
            name="Charlie",
            role="Critic",
            personality="Points out potential problems"
        )
    ]

    # Create conversation
    conversation = MultiAgentConversation(engine, agents)

    # Start conversation
    print("=== CONVERSATION START ===")
    intro = conversation.start_conversation(
        "Should we adopt AI-powered code review tools?"
    )
    print(intro)

    # Run conversation rounds
    full_conversation = conversation.run_conversation(num_rounds=3)

    print("\n=== FULL CONVERSATION ===")
    print(full_conversation)

if __name__ == "__main__":
    main()
```

---

## Integration Examples

### FastAPI Server

Integrate snapshot-based chat into a FastAPI web service.

```python
# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import uuid
from sglang import Engine
from sglang.snapshot import SnapshotManager

app = FastAPI()

# Initialize engine
engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    enable_mamba_snapshots=True,
    snapshot_config={
        "enable_persistence": True,
        "storage_path": "./api_snapshots"
    }
)

manager = SnapshotManager(engine)

# Session storage
sessions: Dict[str, str] = {}  # session_id -> snapshot_id

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    max_tokens: int = 200
    temperature: float = 0.7

class ChatResponse(BaseModel):
    session_id: str
    response: str
    tokens_generated: int

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a chat message."""
    from sglang.lang import function, gen

    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    previous_snapshot = sessions.get(session_id)

    @function
    def chat_turn(s, snapshot_id, message):
        if snapshot_id:
            s.restore_snapshot(snapshot_id)

        s += f"\nUser: {message}\n"
        s += "Assistant: " + gen(
            "response",
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        new_snapshot = s.save_snapshot()
        return s, new_snapshot

    try:
        result, new_snapshot = chat_turn.run(
            snapshot_id=previous_snapshot,
            message=request.message,
            engine=engine
        )

        # Update session
        sessions[session_id] = new_snapshot

        return ChatResponse(
            session_id=session_id,
            response=result["response"],
            tokens_generated=len(result["response"].split())
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    if session_id in sessions:
        snapshot_id = sessions[session_id]
        manager.delete_snapshot(snapshot_id)
        del sessions[session_id]
        return {"message": "Session deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    return {
        "sessions": list(sessions.keys()),
        "count": len(sessions)
    }

# Run with: uvicorn server:app --reload
```

For detailed migration instructions and more examples, see the complete [Examples](examples.md) and [Migration Guide](migration_guide.md) documentation.
