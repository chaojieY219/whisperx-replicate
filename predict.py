from cog import BasePredictor, Input, Path, BaseModel
from typing import Any
from whisperx.audio import N_SAMPLES, log_mel_spectrogram

import gc
import math
import os

# Make torch.load behave like pre-2.6 (weights_only=False) for libraries that don't
# yet pass weights_only explicitly, such as pyannote/lightning. This avoids the
# new safe-unpickler errors when loading older checkpoints that store more than
# plain tensor weights. See PyTorch docs on TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import shutil
import whisperx
import tempfile
import time
import torch
import ffmpeg

compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
device = "cuda"
# Use faster-whisper "turbo" model; weights will be downloaded automatically.
whisper_arch = "turbo"


class Output(BaseModel):
    segments: Any
    detected_language: str


class Predictor(BasePredictor):
    def setup(self):
        source_folder = './models/vad'
        destination_folder = '../root/.cache/torch'
        file_name = 'whisperx-vad-segmentation.bin'

        os.makedirs(destination_folder, exist_ok=True)

        source_file_path = os.path.join(source_folder, file_name)
        if os.path.exists(source_file_path):
            destination_file_path = os.path.join(destination_folder, file_name)

            if not os.path.exists(destination_file_path):
                shutil.copy(source_file_path, destination_folder)

    def predict(
        self,
        audio_file: Path = Input(
            description="Audio file to transcribe (any ffmpeg-supported format)",
        ),
        language: str = Input(
            description="ISO language code of the spoken language (set to None to auto-detect)",
            default=None,
        ),
        language_detection_min_prob: float = Input(
            description="Minimum confidence required for language detection; if not reached, detection retries on more segments",
            default=0,
        ),
        language_detection_max_tries: int = Input(
            description="Maximum number of segments to try for language detection before accepting the best guess",
            default=5,
        ),
        initial_prompt: str = Input(
            description="Optional text prompt for the first window to bias transcription",
            default=None,
        ),
        batch_size: int = Input(
            description="Number of audio chunks processed in parallel (higher is faster but uses more GPU memory)",
            default=64,
        ),
        temperature: float = Input(
            description="Sampling temperature for decoding (higher is more diverse but less stable)",
            default=0,
        ),
        vad_onset: float = Input(
            description="VAD threshold to start a speech segment (probability above this value starts speech)",
            default=0.500,
        ),
        vad_offset: float = Input(
            description="VAD threshold to end a speech segment (probability below this value ends speech)",
            default=0.363,
        ),
        align_output: bool = Input(
            description="If true, run alignment to refine word-level timestamps when supported for the detected language",
            default=False,
        ),
        diarization: bool = Input(
            description="If true, run speaker diarization and assign speaker labels to segments",
            default=False,
        ),
        huggingface_access_token: str = Input(
            description="Hugging Face access token (read) required for gated pyannote diarization models",
            default=None,
        ),
        min_speakers: int = Input(
            description="Minimum number of speakers for diarization (leave None if unknown)",
            default=None,
        ),
        max_speakers: int = Input(
            description="Maximum number of speakers for diarization (leave None if unknown)",
            default=None,
        ),
        debug: bool = Input(
            description="If true, print detailed timing and GPU memory usage information",
            default=False,
        )
    ) -> Output:
        with torch.inference_mode():
            asr_options = {
                "temperatures": [temperature],
                "initial_prompt": initial_prompt
            }

            vad_options = {
                "vad_onset": vad_onset,
                "vad_offset": vad_offset
            }

            audio_duration = get_audio_duration(audio_file)

            if language is None and language_detection_min_prob > 0 and audio_duration > 30000:
                segments_duration_ms = 30000

                language_detection_max_tries = min(
                    language_detection_max_tries,
                    math.floor(audio_duration / segments_duration_ms)
                )

                segments_starts = distribute_segments_equally(audio_duration, segments_duration_ms,
                                                              language_detection_max_tries)

                print("Detecting languages on segments starting at " + ', '.join(map(str, segments_starts)))

                detected_language_details = detect_language(audio_file, segments_starts, language_detection_min_prob,
                                                            language_detection_max_tries, asr_options, vad_options)

                detected_language_code = detected_language_details["language"]
                detected_language_prob = detected_language_details["probability"]
                detected_language_iterations = detected_language_details["iterations"]

                print(f"Detected language {detected_language_code} ({detected_language_prob:.2f}) after "
                      f"{detected_language_iterations} iterations.")

                language = detected_language_details["language"]

            start_time = time.time_ns() / 1e6

            model = whisperx.load_model(
                whisper_arch,
                device,
                compute_type=compute_type,
                language=language,
                asr_options=asr_options,
                vad_options=vad_options,
            )

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load model: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            audio = whisperx.load_audio(audio_file)

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to load audio: {elapsed_time:.2f} ms")

            start_time = time.time_ns() / 1e6

            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = result["language"]

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                print(f"Duration to transcribe: {elapsed_time:.2f} ms")

            gc.collect()
            torch.cuda.empty_cache()
            del model

            if align_output:
                if detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH or detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_HF:
                    result = align(audio, result, debug)
                else:
                    print(f"Cannot align output as language {detected_language} is not supported for alignment")

            if diarization:
                result = diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers)

            if debug:
                print(f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")

        return Output(
            segments=result["segments"],
            detected_language=detected_language
        )


def get_audio_duration(file_path):
    probe = ffmpeg.probe(file_path)
    stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
    return float(stream['duration']) * 1000


def detect_language(full_audio_file_path, segments_starts, language_detection_min_prob,
                    language_detection_max_tries, asr_options, vad_options, iteration=1):
    model = whisperx.load_model(
        whisper_arch,
        device,
        compute_type=compute_type,
        asr_options=asr_options,
        vad_options=vad_options,
    )

    start_ms = segments_starts[iteration - 1]

    audio_segment_file_path = extract_audio_segment(full_audio_file_path, start_ms, 30000)

    audio = whisperx.load_audio(audio_segment_file_path)

    model_n_mels = model.model.feat_kwargs.get("feature_size")
    segment = log_mel_spectrogram(audio[: N_SAMPLES],
                                  n_mels=model_n_mels if model_n_mels is not None else 80,
                                  padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
    encoder_output = model.model.encode(segment)
    results = model.model.model.detect_language(encoder_output)
    language_token, language_probability = results[0][0]
    language = language_token[2:-2]

    print(f"Iteration {iteration} - Detected language: {language} ({language_probability:.2f})")

    audio_segment_file_path.unlink()

    gc.collect()
    torch.cuda.empty_cache()
    del model

    detected_language = {
        "language": language,
        "probability": language_probability,
        "iterations": iteration
    }

    if language_probability >= language_detection_min_prob or iteration >= language_detection_max_tries:
        return detected_language

    next_iteration_detected_language = detect_language(full_audio_file_path, segments_starts,
                                                       language_detection_min_prob, language_detection_max_tries,
                                                       asr_options, vad_options, iteration + 1)

    if next_iteration_detected_language["probability"] > detected_language["probability"]:
        return next_iteration_detected_language

    return detected_language


def extract_audio_segment(input_file_path, start_time_ms, duration_ms):
    input_file_path = Path(input_file_path) if not isinstance(input_file_path, Path) else input_file_path
    file_extension = input_file_path.suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file_path = Path(temp_file.name)

        print(f"Extracting from {input_file_path.name} to {temp_file.name}")

        try:
            (
                ffmpeg
                .input(input_file_path, ss=start_time_ms/1000)
                .output(temp_file.name, t=duration_ms/1000)
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            print("ffmpeg error occurred: ", e.stderr.decode('utf-8'))
            raise e

    return temp_file_path


def distribute_segments_equally(total_duration, segments_duration, iterations):
    available_duration = total_duration - segments_duration

    if iterations > 1:
        spacing = available_duration // (iterations - 1)
    else:
        spacing = 0

    start_times = [i * spacing for i in range(iterations)]

    if iterations > 1:
        start_times[-1] = total_duration - segments_duration

    return start_times


def align(audio, result, debug):
    start_time = time.time_ns() / 1e6

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                            return_char_alignments=False)

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to align output: {elapsed_time:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    return result


def diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers):
    start_time = time.time_ns() / 1e6

    diarize_model = whisperx.DiarizationPipeline(use_auth_token=huggingface_access_token, device=device)
    diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to diarize segments: {elapsed_time:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    del diarize_model

    return result
