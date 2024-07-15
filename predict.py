import json
import tempfile
import numpy as np
import torchaudio
from cog import BasePredictor, Input, Path
from pyannote.audio import Audio
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Segment
from lib.diarization import DiarizationPostProcessor, format_ts
from lib.audio import AudioPreProcessor
import openai
import os

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.audio_pre = AudioPreProcessor()

        self.diarization = SpeakerDiarization(
            segmentation="/data/pyannote/segmentation/pytorch_model.bin",
            embedding="/data/speechbrain/spkrec-ecapa-voxceleb",
            clustering="AgglomerativeClustering",
            segmentation_batch_size=32,
            embedding_batch_size=32,
            embedding_exclude_overlap=True,
        )
        self.diarization.instantiate({
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 15,
                "threshold": 0.7153814381597874,
            },
            "segmentation": {
                "min_duration_off": 0.5817029604921046,
                "threshold": 0.4442333667381752,
            },
        })
        self.diarization_post = DiarizationPostProcessor()
        
        # Initialize OpenAI API key
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

    def run_diarization(self):
        closure = {'embeddings': None}

        def hook(name, *args, **kwargs):
            if name == "embeddings" and len(args) > 0:
                closure['embeddings'] = args[0]

        print('diarizing audio file...')
        diarization = self.diarization(self.audio_pre.output_path, hook=hook)
        embeddings = {
            'data': closure['embeddings'],
            'chunk_duration': self.diarization.segmentation_duration,
            'chunk_offset': self.diarization.segmentation_step * self.diarization.segmentation_duration,
        }
        return self.diarization_post.process(diarization, embeddings)

    def run_transcription(self, audio, segments, whisper_prompt):
        print('transcribing segments...')
        if whisper_prompt:
            print('using prompt:', repr(whisper_prompt))
        
        trimmer = Audio(sample_rate=16000, mono=True)
        for seg in segments:
            start = seg['start']
            stop = seg['stop']
            print(f"transcribing segment {format_ts(start)} to {format_ts(stop)}")
            frames, _ = trimmer.crop(audio, Segment(start, stop))
            frames = frames[0]
            seg['transcript'] = self.transcribe_segment(frames, start, whisper_prompt)

    def transcribe_segment(self, audio, ctx_start, whisper_prompt):

        temp_audio_path = tempfile.mktemp(suffix=".wav")
        torchaudio.save(temp_audio_path, audio.unsqueeze(0), 16000)
        
        with open(temp_audio_path, "rb") as audio_file:
            response = openai.Audio.transcribe("whisper-1", audio_file, prompt=whisper_prompt)
        
        result = []
        for s in response.get('segments', []):
            timestamp = ctx_start + s['start']
            result.append({
                'start': format_ts(timestamp),
                'text': s['text']
            })
        return result

    def predict(
        self,
        audio: Path = Input(description="Audio file"),
        prompt: str = Input(
            default=None,
            description="Optional text to provide as a prompt for each Whisper model call.",
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        self.audio_pre.process(audio)

        if self.audio_pre.error:
            print(self.audio_pre.error)
            result = self.diarization_post.empty_result()
        else:
            result = self.run_diarization()

        self.run_transcription(self.audio_pre.output_path, result["segments"], prompt)

        result["segments"] = self.diarization_post.format_segments(result["segments"])

        self.audio_pre.cleanup()
        output = Path(tempfile.mkdtemp()) / "output.json"
        with open(output, "w") as f:
            f.write(json.dumps(result, indent=2))
        return output
