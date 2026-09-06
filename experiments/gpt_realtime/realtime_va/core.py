"""
Core WebSocket communication with OpenAI Realtime API
"""

import os
import json
import base64
import logging
from pathlib import Path
from typing import Optional, Callable, Awaitable

import websockets

logger = logging.getLogger(__name__)


class RealtimeVACore:
    """Core WebSocket handler for OpenAI Realtime API"""

    def __init__(
        self,
        model: str = "gpt-realtime-mini",
        prompt_file: Optional[str] = None,
        mode: str = "live",
    ):
        """
        Initialize Realtime VA Core

        Args:
            model: OpenAI model to use
            prompt_file: Path to system prompt file (defaults to prompts/sigma_wakeup.txt)
            mode: Operation mode ('live' or 'batch') for VAD configuration
        """
        self.model = model
        self.mode = mode
        self.ws = None
        self.is_running = False

        # Load system prompt from file
        if prompt_file:
            self.prompt_file = Path(prompt_file)
        else:
            # Default to sigma_wakeup.txt in prompts/ directory
            script_dir = Path(__file__).parent.parent
            self.prompt_file = script_dir / "prompts" / "sigma_wakeup.txt"

        self.system_prompt = self._load_prompt()

        # Get API key
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Event callbacks (set by interface)
        self.on_audio_delta: Optional[Callable[[bytes], Awaitable[None]]] = None
        self.on_audio_done: Optional[Callable[[], Awaitable[None]]] = None
        self.on_transcript_delta: Optional[Callable[[str], Awaitable[None]]] = None
        self.on_transcript_done: Optional[Callable[[str], Awaitable[None]]] = None
        self.on_speech_started: Optional[Callable[[], Awaitable[None]]] = None
        self.on_speech_stopped: Optional[Callable[[], Awaitable[None]]] = None
        self.on_user_transcript: Optional[Callable[[str], Awaitable[None]]] = None
        self.on_response_done: Optional[Callable[[], Awaitable[None]]] = None

        # Internal state
        self.suppress_current_response = False

    def _load_prompt(self) -> str:
        """Load system prompt from file"""
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            return prompt
        except FileNotFoundError as e:
            logger.error("Prompt file not found: %s", self.prompt_file)
            logger.error(
                "Please create the prompt file or use --prompt to specify a different file"
            )
            raise FileNotFoundError(
                f"System prompt file not found: {self.prompt_file}. "
                f"Create this file or use --prompt to specify a different prompt file."
            ) from e
        except (OSError, IOError) as e:
            logger.error("Error reading prompt file: %s", e)
            raise

    async def connect(self):
        """Establish WebSocket connection to OpenAI Realtime API"""
        uri = f"wss://api.openai.com/v1/realtime?model={self.model}"

        try:
            self.ws = await websockets.connect(
                uri,
                additional_headers={
                    "Authorization": f"Bearer {self.api_key}",
                },
            )

            # Configure session
            await self.configure_session()

        except Exception as e:
            logger.error("Failed to connect: %s", e)
            raise

    async def configure_session(self):
        """Configure the OpenAI Realtime API session"""
        config = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self.model,
                "instructions": self.system_prompt,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": 24000,
                        },
                        "turn_detection": (
                            {"type": "semantic_vad"} if self.mode == "live" else None
                        ),
                    },
                    "output": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": 24000,
                        },
                        "voice": "marin",
                    },
                },
            },
        }

        await self.ws.send(json.dumps(config))

    async def send_audio_chunk(self, audio_data: bytes):
        """Send audio chunk to OpenAI"""
        audio_b64 = base64.b64encode(audio_data).decode()
        message = {"type": "input_audio_buffer.append", "audio": audio_b64}
        await self.ws.send(json.dumps(message))

    async def commit_audio(self):
        """Commit audio buffer and request response"""
        await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await self.ws.send(json.dumps({"type": "response.create"}))
        logger.debug("Audio committed, waiting for response...")

    async def inject_assistant_message(self, text: str):
        """Inject a text message as an assistant turn into the conversation history.

        Used in persistent-session evaluation to feed the corpus Sigma response
        back into context after each evaluated turn, so the model knows what the
        VA "said" when deciding how to handle follow-up turns.
        """
        message = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            },
        }
        await self.ws.send(json.dumps(message))
        logger.debug("Injected assistant message: %s", text)

    async def _handle_speech_started(self):
        """Handle speech started event"""
        logger.debug("[SPEECH] Detected")
        if self.on_speech_started:
            await self.on_speech_started()

    async def _handle_speech_stopped(self):
        """Handle speech stopped event"""
        logger.debug("[SPEECH] Ended")
        if self.on_speech_stopped:
            await self.on_speech_stopped()

    async def _handle_user_transcript(self, event: dict):
        """Handle user transcript completion"""
        transcript = event.get("transcript", "")
        logger.info("[USER] You: %s", transcript)
        if self.on_user_transcript:
            await self.on_user_transcript(transcript)

    async def _handle_transcript_delta(self, event: dict):
        """Handle transcript delta"""
        delta = event.get("delta", "")
        # Check if this is a [SILENCE] response
        if "[SILENCE]" in delta.upper():
            self.suppress_current_response = True
            logger.debug("Detected [SILENCE] response - suppressing audio")
        if self.on_transcript_delta is not None and not self.suppress_current_response:
            await self.on_transcript_delta(delta)

    async def _handle_transcript_done(self, event: dict):
        """Handle transcript done"""
        transcript = event.get("transcript", "")
        # Check if this is a [SILENCE] response
        if "[SILENCE]" in transcript.upper():
            self.suppress_current_response = True
            logger.debug("[ASSISTANT] Wakeup word not detected - staying silent")
        else:
            logger.debug("[ASSISTANT] Finished speaking")

        if self.on_transcript_done:
            await self.on_transcript_done(transcript)

    async def _handle_audio_delta(self, event: dict):
        """Handle audio delta"""
        # Only queue audio if not suppressing
        if not self.suppress_current_response:
            audio_b64 = event.get("delta", "")
            if audio_b64:
                audio_data = base64.b64decode(audio_b64)
                if self.on_audio_delta:
                    await self.on_audio_delta(audio_data)

    async def _handle_audio_done(self):
        """Handle audio done"""
        if self.on_audio_done:
            await self.on_audio_done()
        # Reset suppression flag for next response
        self.suppress_current_response = False

    async def _dispatch_event(self, event: dict):
        """Dispatch event to appropriate handler"""
        event_type = event.get("type")

        # Map event types to handlers that need event data
        handlers_with_event = {
            "conversation.item.input_audio_transcription.completed": self._handle_user_transcript,
            "response.output_audio_transcript.delta": self._handle_transcript_delta,
            "response.output_audio_transcript.done": self._handle_transcript_done,
            "response.output_audio.delta": self._handle_audio_delta,
        }

        # Map event types to handlers that don't need event data
        handlers_no_event = {
            "input_audio_buffer.speech_started": self._handle_speech_started,
            "input_audio_buffer.speech_stopped": self._handle_speech_stopped,
            "response.output_audio.done": self._handle_audio_done,
        }

        # Simple events that just need logging
        if event_type == "response.done":
            if self.on_response_done:
                await self.on_response_done()
        elif event_type == "error":
            error_info = event.get("error", {})
            logger.error("Error from API: %s", error_info)
        elif event_type in handlers_with_event:
            await handlers_with_event[event_type](event)
        elif event_type in handlers_no_event:
            await handlers_no_event[event_type]()

    async def handle_server_messages(self):
        """Handle incoming messages from OpenAI"""
        self.is_running = True
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type")

                await self._dispatch_event(event)

        except websockets.exceptions.ConnectionClosed:
            logger.debug("Connection closed")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error handling server messages: %s", e)
        finally:
            self.is_running = False

    async def close(self):
        """Close WebSocket connection"""
        if self.ws:
            await self.ws.close()
