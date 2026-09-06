"""
Audio input/output handling with PyAudio
"""

import asyncio
import logging
import wave
import array
from pathlib import Path
from typing import Optional

import pyaudio

logger = logging.getLogger(__name__)


class AudioInput:
    """Handle audio input from microphone or WAV files"""

    SAMPLE_RATE = 24000
    CHANNELS = 1
    SAMPLE_WIDTH = 2  # 16-bit PCM
    CHUNK_SIZE = 4800  # 200ms at 24kHz

    def __init__(self, mode: str = "live"):
        """
        Initialize Audio Input

        Args:
            mode: 'live' for microphone, 'batch' for WAV files
        """
        self.mode = mode
        self.audio_interface = None
        self.input_stream = None
        self.is_running = False

    async def start_recording(self):
        """Start recording from microphone (live mode only)"""
        if self.mode != "live":
            raise ValueError("start_recording() only available in live mode")

        # Initialize PyAudio
        self.audio_interface = pyaudio.PyAudio()

        # Get default input device info
        try:
            default_device = self.audio_interface.get_default_input_device_info()
            logger.info("Using audio device: %s", default_device["name"])
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Could not get default device info: %s", e)

        # Open input stream
        self.input_stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=self.CHANNELS,
            rate=self.SAMPLE_RATE,
            input=True,
            input_device_index=None,  # Use default
            frames_per_buffer=self.CHUNK_SIZE,
        )

        self.is_running = True
        logger.info("[MIC] Microphone active - start speaking!")

    async def read_chunk(self) -> Optional[bytes]:
        """
        Read audio chunk from microphone

        Returns:
            Audio chunk as bytes, or None if not recording
        """
        if not self.is_running or not self.input_stream:
            return None

        # Read in thread executor to avoid blocking
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            None,
            lambda: self.input_stream.read(
                self.CHUNK_SIZE, exception_on_overflow=False
            ),
        )
        return audio_data

    async def load_wav_file(self, file_path: Path) -> tuple[bytes, int]:
        """
        Load and prepare WAV file for sending

        Args:
            file_path: Path to WAV file

        Returns:
            Tuple of (audio_data as bytes, sample_rate)
        """
        try:
            with wave.open(str(file_path), "rb") as wav_file:
                # Verify format
                if wav_file.getnchannels() != self.CHANNELS:
                    raise ValueError(
                        f"Expected mono audio, got {wav_file.getnchannels()} channels"
                    )

                sample_rate = wav_file.getframerate()
                audio_data = wav_file.readframes(wav_file.getnframes())

                # Resample if needed
                if sample_rate != self.SAMPLE_RATE:
                    logger.info(
                        "Resampling audio from %dHz to %dHz...",
                        sample_rate,
                        self.SAMPLE_RATE,
                    )
                    # Simple linear interpolation resampling
                    samples = array.array("h", audio_data)
                    ratio = sample_rate / self.SAMPLE_RATE
                    new_length = int(len(samples) / ratio)
                    resampled = array.array("h", [0] * new_length)

                    for i in range(new_length):
                        src_idx = int(i * ratio)
                        if src_idx < len(samples):
                            resampled[i] = samples[src_idx]

                    audio_data = resampled.tobytes()
                    sample_rate = self.SAMPLE_RATE

                return audio_data, sample_rate

        except (OSError, ValueError, wave.Error) as e:
            logger.error("Error loading WAV file %s: %s", file_path, e)
            raise

    def stop(self):
        """Stop recording and cleanup"""
        self.is_running = False
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.audio_interface:
            self.audio_interface.terminate()


class AudioOutput:
    """Handle audio output to speakers"""

    SAMPLE_RATE = 24000
    CHANNELS = 1
    SAMPLE_WIDTH = 2  # 16-bit PCM
    CHUNK_SIZE = 4800  # 200ms at 24kHz

    def __init__(self):
        """Initialize Audio Output"""
        self.audio_interface = None
        self.playback_stream = None
        self.playback_queue = asyncio.Queue()
        self.is_running = False

    async def initialize(self):
        """Initialize PyAudio for playback"""
        if not self.audio_interface:
            self.audio_interface = pyaudio.PyAudio()

        if not self.playback_stream:
            self.playback_stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                output=True,
                frames_per_buffer=self.CHUNK_SIZE,
            )
            logger.info("[PLAY] Audio output initialized")

    async def play_chunk(self, audio_data: bytes):
        """Queue audio chunk for playback"""
        await self.playback_queue.put(audio_data)

    async def play_complete(self):
        """Signal that current response playback is complete"""
        await self.playback_queue.put(None)

    async def play_raw_audio(self, audio_data: bytes, sample_rate: int = None):
        """
        Play raw audio data directly (for input file playback in batch mode)

        Args:
            audio_data: Raw PCM audio data
            sample_rate: Sample rate (defaults to SAMPLE_RATE)
        """
        if sample_rate is None:
            sample_rate = self.SAMPLE_RATE

        if not self.playback_stream:
            await self.initialize()

        logger.info("[PLAY] Playing input audio...")

        # Play in chunks
        offset = 0
        chunk_size_bytes = self.CHUNK_SIZE * self.SAMPLE_WIDTH
        while offset < len(audio_data):
            chunk = audio_data[offset : offset + chunk_size_bytes]
            self.playback_stream.write(chunk)
            offset += chunk_size_bytes

    async def playback_loop(self):
        """Main playback loop - processes queued audio chunks"""
        await self.initialize()
        self.is_running = True

        try:
            while self.is_running:
                audio_data = await self.playback_queue.get()

                if audio_data is None:
                    # End of current response
                    logger.debug("[PLAY] End of response marker received")
                    continue

                # Play the audio chunk
                if self.playback_stream:
                    logger.debug("[PLAY] Playing %d bytes", len(audio_data))
                    self.playback_stream.write(audio_data)

        except (OSError, ValueError, asyncio.CancelledError) as e:
            logger.error("Error in playback loop: %s", e)
        finally:
            self.stop()

    def stop(self):
        """Stop playback and cleanup"""
        self.is_running = False
        if self.playback_stream:
            try:
                self.playback_stream.stop_stream()
                self.playback_stream.close()
                logger.debug("[PLAY] Playback stream closed")
            except OSError as e:
                logger.error("Error closing playback stream: %s", e)
        if self.audio_interface:
            self.audio_interface.terminate()
