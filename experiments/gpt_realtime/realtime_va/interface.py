"""
User interface implementations for live and batch modes
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from realtime_va.core import RealtimeVACore
    from realtime_va.audio_io import AudioInput, AudioOutput

logger = logging.getLogger(__name__)


class LiveInterface:
    """Interface for live mode - real-time microphone conversation"""

    def __init__(
        self,
        va_core: "RealtimeVACore",
        audio_input: "AudioInput",
        audio_output: "AudioOutput",
    ):
        """
        Initialize Live Interface

        Args:
            va_core: VA Core instance
            audio_input: Audio input instance
            audio_output: Audio output instance
        """
        self.va_core = va_core
        self.audio_input = audio_input
        self.audio_output = audio_output
        self.chunk_count = 0

    async def run(self):
        """Run live mode interface"""
        logger.info("\n%s", "=" * 60)
        logger.info("LIVE MODE - Real-time conversation")
        logger.info("%s", "=" * 60)
        logger.info("Press Ctrl+C to stop")
        logger.info("%s\n", "=" * 60)

        # Wire up callbacks
        self.va_core.on_audio_delta = self._handle_audio_delta
        self.va_core.on_audio_done = self._handle_audio_done
        self.va_core.on_transcript_delta = self._handle_transcript_delta

        # Connect and configure
        await self.va_core.connect()

        # Start recording
        await self.audio_input.start_recording()

        # Create tasks
        tasks = [
            asyncio.create_task(self.va_core.handle_server_messages()),
            asyncio.create_task(self._send_audio_loop()),
            asyncio.create_task(self.audio_output.playback_loop()),
        ]

        try:
            # Wait for any task to complete
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # Check if any tasks failed
            for task in done:
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass
                except (
                    Exception
                ) as task_error:  # pylint: disable=broad-exception-caught
                    logger.error("Task failed: %s", task_error)

            # Cancel remaining tasks
            for task in pending:
                task.cancel()

            # Wait for all tasks to finish
            await asyncio.gather(*pending, return_exceptions=True)

        except KeyboardInterrupt:
            logger.info("\nShutting down...")
        finally:
            self.audio_input.stop()
            self.audio_output.stop()
            await self.va_core.close()

    async def _send_audio_loop(self):
        """Continuously read and send audio from microphone"""
        try:
            while self.va_core.is_running:
                audio_data = await self.audio_input.read_chunk()
                if audio_data:
                    await self.va_core.send_audio_chunk(audio_data)

                    self.chunk_count += 1
                    if self.chunk_count % 50 == 0:  # Every ~10 seconds
                        logger.info(
                            "[MIC] Sent %d audio chunks (~%.1fs)",
                            self.chunk_count,
                            self.chunk_count
                            * self.audio_input.CHUNK_SIZE
                            / self.audio_input.SAMPLE_RATE,
                        )

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error in audio send loop: %s", e)

    async def _handle_audio_delta(self, audio_data: bytes):
        """Handle audio chunk from VA core"""
        await self.audio_output.play_chunk(audio_data)

    async def _handle_audio_done(self):
        """Handle audio response completion"""
        await self.audio_output.play_complete()

    async def _handle_transcript_delta(self, delta: str):
        """Handle transcript delta from VA core"""
        print(delta, end="", flush=True)


class BatchInterface:
    """Interface for batch mode - turn-based WAV file testing"""

    def __init__(
        self,
        va_core: "RealtimeVACore",
        audio_input: "AudioInput",
        audio_output: "AudioOutput",
        audio_dir: Path,
    ):
        """
        Initialize Batch Interface

        Args:
            va_core: VA Core instance
            audio_input: Audio input instance
            audio_output: Audio output instance
            audio_dir: Directory containing WAV files
        """
        self.va_core = va_core
        self.audio_input = audio_input
        self.audio_output = audio_output
        self.audio_dir = audio_dir
        self.response_complete_event = asyncio.Event()

    async def run(self):
        """Run batch mode interface"""
        logger.info("\n%s", "=" * 60)
        logger.info("BATCH MODE")
        logger.info("%s", "=" * 60)
        logger.info("Commands:")
        logger.info("  - Enter WAV filename (e.g., '1.wav') to send audio")
        logger.info("  - Type 'list' to see available files")
        logger.info("  - Type 'EOD' or 'quit' to exit")
        logger.info("%s\n", "=" * 60)

        # Initialize audio output
        await self.audio_output.initialize()

        # Wire up callbacks
        self.va_core.on_audio_delta = self._handle_audio_delta
        self.va_core.on_audio_done = self._handle_audio_done
        self.va_core.on_transcript_delta = self._handle_transcript_delta

        # Connect and configure
        await self.va_core.connect()

        # List available files
        self._list_wav_files()
        print()

        # Create tasks
        tasks = [
            asyncio.create_task(self.va_core.handle_server_messages()),
            asyncio.create_task(self.audio_output.playback_loop()),
            asyncio.create_task(self._batch_loop()),
        ]

        try:
            # Wait for any task to complete
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # Check if any tasks failed
            for task in done:
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass
                except (
                    Exception
                ) as task_error:  # pylint: disable=broad-exception-caught
                    logger.error("Task failed: %s", task_error)

            # Cancel remaining tasks
            for task in pending:
                task.cancel()

            # Wait for all tasks to finish
            await asyncio.gather(*pending, return_exceptions=True)

        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
        finally:
            self.audio_output.stop()
            await self.va_core.close()

    def _list_wav_files(self):
        """List available WAV files in the directory"""
        wav_files = sorted(self.audio_dir.glob("*.wav"))
        if wav_files:
            logger.info("Available WAV files:")
            for f in wav_files:
                logger.info("  - %s", f.name)
        else:
            logger.warning("No WAV files found in %s", self.audio_dir)

    async def _batch_loop(self):
        """Main batch mode loop - prompt for files and process"""
        try:
            while self.va_core.is_running:
                # Get user input
                print("Enter filename (or 'EOD' to quit): ", end="", flush=True)
                user_input = await asyncio.get_event_loop().run_in_executor(None, input)
                user_input = user_input.strip()

                if user_input.upper() in ["EOD", "QUIT", "EXIT"]:
                    logger.info("Exiting batch mode...")
                    break

                if user_input.lower() == "list":
                    self._list_wav_files()
                    continue

                # Try to load the file
                file_path = self.audio_dir / user_input

                if not file_path.exists():
                    logger.error("File not found: %s", file_path)
                    continue

                # Reset suppression flag before processing each file
                self.va_core.suppress_current_response = False

                # Load and process file
                try:
                    logger.info("[FILE] Loading: %s", file_path.name)

                    # Load WAV file
                    audio_data, sample_rate = await self.audio_input.load_wav_file(
                        file_path
                    )

                    # Play the input audio file
                    await self.audio_output.play_raw_audio(audio_data, sample_rate)

                    # Send to OpenAI
                    logger.info("[SEND] Sending audio...")
                    offset = 0
                    chunk_size = (
                        self.audio_input.CHUNK_SIZE * self.audio_input.SAMPLE_WIDTH
                    )
                    while offset < len(audio_data):
                        chunk = audio_data[offset : offset + chunk_size]
                        await self.va_core.send_audio_chunk(chunk)
                        offset += chunk_size

                    # Commit and request response
                    await self.va_core.commit_audio()

                    # Wait for response
                    print("[ASSISTANT] ", end="", flush=True)
                    self.response_complete_event.clear()
                    await self.response_complete_event.wait()

                    # Small delay before next prompt
                    await asyncio.sleep(0.2)
                    print("\n")

                except (OSError, ValueError) as e:
                    logger.error("Error processing file: %s", e)

        finally:
            self.va_core.is_running = False

    async def _handle_audio_delta(self, audio_data: bytes):
        """Handle audio chunk from VA core"""
        await self.audio_output.play_chunk(audio_data)

    async def _handle_audio_done(self):
        """Handle audio response completion"""
        await self.audio_output.play_complete()
        self.response_complete_event.set()

    async def _handle_transcript_delta(self, delta: str):
        """Handle transcript delta from VA core"""
        print(delta, end="", flush=True)
