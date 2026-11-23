# Where the stand-in clips come from

Every clip in this folder is real human speech from **LibriSpeech**
(Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur: "LibriSpeech: an ASR corpus
based on public domain audio books", ICASSP 2015), https://www.openslr.org/12 ,
licence **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). LibriSpeech is made
from audiobooks read by LibriVox volunteers. No clip here was made with text-to-speech or any
other synthetic voice.

The clips were fetched on 2026-11-23 through the Hugging Face copy of the corpus
(`openslr/librispeech_asr`). An utterance id reads speaker-chapter-number.

Changes made to the originals (CC BY asks that changes are stated): each clip is a cut-out
of one utterance with 10 ms fades at both ends; clips named `stretched_...` also have the one
stand-in word slowed down with a phase vocoder, to stand in for a word that is held while
singing. They are not singing, and a stretched word sounds processed.

The stand-in "blocked" words are `pineapple` and `banana`. The times in the `.txt` files come
from two offline passes over the whole source utterance (faster-whisper small.en with beam 5,
and Vosk), cross-checked against the dataset's transcript and a loudness check. Nobody
listened to them. Expect the edges to be right to about 50 ms, not better.

Total size of the clips: 2.6 MB. Rebuild with `uv run python tools/make_standin_clips.py`.

| clip | source utterance | split | length | stand-in word at | what was changed |
|---|---|---|---|---|---|
| `spoken_banana_01.flac` | 460-172359-0062 | clean/train.100 | 6.1 s | 3.79-4.18 s (banana) | trimmed to the stretch around the word |
| `stretched_banana_01_x2.flac` | 460-172359-0062 | clean/train.100 | 6.6 s | 3.82-4.60 s (banana) | trimmed; the word alone stretched to 2x its length (phase vocoder), 390 ms -> 780 ms |
| `spoken_banana_02.flac` | 460-172359-0062 | clean/train.100 | 4.2 s | 3.24-3.60 s (banana) | trimmed to the stretch around the word |
| `spoken_banana_03.flac` | 460-172359-0070 | clean/train.100 | 2.6 s | 0.00-0.66 s (banana) | trimmed to the stretch around the word |
| `spoken_banana_04.flac` | 460-172359-0088 | clean/train.100 | 4.7 s | 2.56-3.00 s (banana) | trimmed to the stretch around the word |
| `stretched_banana_04_x1_5.flac` | 460-172359-0088 | clean/train.100 | 4.9 s | 2.58-3.23 s (banana) | trimmed; the word alone stretched to 1.5x its length (phase vocoder), 440 ms -> 660 ms |
| `stretched_banana_04_x3.flac` | 460-172359-0088 | clean/train.100 | 5.7 s | 2.62-3.94 s (banana) | trimmed; the word alone stretched to 3x its length (phase vocoder), 440 ms -> 1320 ms |
| `spoken_banana_05.flac` | 460-172359-0078 | clean/train.100 | 5.5 s | 3.60-3.98 s (banana) | trimmed to the stretch around the word |
| `spoken_banana_06.flac` | 460-172357-0012 | clean/train.100 | 5.8 s | 3.74-4.20 s (banana) | trimmed to the stretch around the word |
| `stretched_banana_06_x1_5.flac` | 460-172357-0012 | clean/train.100 | 6.0 s | 3.75-4.45 s (banana) | trimmed; the word alone stretched to 1.5x its length (phase vocoder), 460 ms -> 690 ms |
| `spoken_banana_07.flac` | 460-172359-0067 | clean/train.100 | 5.8 s | 3.47-4.01 s (banana) | trimmed to the stretch around the word |
| `spoken_banana_08.flac` | 1556-139350-0006 | clean/train.360 | 4.4 s | 2.28-2.70 s (banana) | trimmed to the stretch around the word |
| `stretched_banana_08_x2.flac` | 1556-139350-0006 | clean/train.360 | 4.9 s | 2.31-3.15 s (banana) | trimmed; the word alone stretched to 2x its length (phase vocoder), 420 ms -> 840 ms |
| `stretched_banana_08_x3.flac` | 1556-139350-0006 | clean/train.360 | 5.3 s | 2.34-3.60 s (banana) | trimmed; the word alone stretched to 3x its length (phase vocoder), 420 ms -> 1260 ms |
| `spoken_pineapple_01.flac` | 7226-86964-0047 | clean/train.100 | 5.5 s | 2.98-3.75 s (pineapple) | trimmed to the stretch around the word |
| `stretched_pineapple_01_x1_5.flac` | 7226-86964-0047 | clean/train.100 | 6.0 s | 3.00-4.15 s (pineapple) | trimmed; the word alone stretched to 1.5x its length (phase vocoder), 770 ms -> 1155 ms |
| `stretched_pineapple_01_x2.flac` | 7226-86964-0047 | clean/train.100 | 6.4 s | 3.01-4.55 s (pineapple) | trimmed; the word alone stretched to 2x its length (phase vocoder), 770 ms -> 1540 ms |
| `stretched_pineapple_01_x3.flac` | 7226-86964-0047 | clean/train.100 | 7.2 s | 3.04-5.35 s (pineapple) | trimmed; the word alone stretched to 3x its length (phase vocoder), 770 ms -> 2310 ms |
| `clean_apple_01.flac` | 445-123857-0008 | clean/train.100 | 8.2 s | none (clean clip) | unchanged |
| `clean_apple_02.flac` | 4214-7146-0027 | clean/train.100 | 8.4 s | none (clean clip) | unchanged |
| `clean_apple_03.flac` | 2790-142824-0040 | clean/train.360 | 4.2 s | none (clean clip) | unchanged |
| `clean_apple_04.flac` | 7976-110124-0013 | clean/validation | 6.6 s | none (clean clip) | unchanged |
| `clean_apple_05.flac` | 1472-142848-0003 | clean/train.360 | 7.5 s | none (clean clip) | unchanged |
| `clean_plain_01.flac` | 460-172359-0063 | clean/train.100 | 8.9 s | none (clean clip) | trimmed |
| `clean_plain_02.flac` | 7226-86964-0048 | clean/train.100 | 9.2 s | none (clean clip) | trimmed |

## How each word was located

- `spoken_banana_01` (460-172359-0062): faster-whisper 4.04-4.30 s, Vosk 4.02-4.41 s (in the source utterance); span loudness -27 dB against a quiet level of -60 dB
- `spoken_banana_02` (460-172359-0062): faster-whisper 14.66-14.90 s, Vosk 14.64-15.00 s (in the source utterance); span loudness -29 dB against a quiet level of -60 dB
- `spoken_banana_03` (460-172359-0070): faster-whisper 0.00-0.60 s, Vosk 0.15-0.66 s (in the source utterance); span loudness -25 dB against a quiet level of -64 dB
- `spoken_banana_04` (460-172359-0088): faster-whisper 2.56-2.88 s, Vosk 2.58-3.00 s (in the source utterance); span loudness -29 dB against a quiet level of -62 dB
- `spoken_banana_05` (460-172359-0078): faster-whisper 9.82-10.10 s, Vosk 9.84-10.20 s (in the source utterance); span loudness -27 dB against a quiet level of -63 dB
- `spoken_banana_06` (460-172357-0012): faster-whisper 10.76-11.16 s, Vosk 10.83-11.22 s (in the source utterance); span loudness -21 dB against a quiet level of -63 dB
- `spoken_banana_07` (460-172359-0067): faster-whisper 6.66-7.02 s, Vosk 6.66-7.20 s (in the source utterance); span loudness -27 dB against a quiet level of -62 dB
- `spoken_banana_08` (1556-139350-0006): faster-whisper 2.28-2.58 s, Vosk 2.28-2.70 s (in the source utterance); span loudness -24 dB against a quiet level of -55 dB
- `spoken_pineapple_01` (7226-86964-0047): faster-whisper 2.98-3.70 s, Vosk 3.06-3.75 s (in the source utterance); span loudness -28 dB against a quiet level of -65 dB

## Clean clips

- `clean_apple_01` (445-123857-0008): says "apple"
- `clean_apple_02` (4214-7146-0027): says "apples"
- `clean_apple_03` (2790-142824-0040): says "apple tart"
- `clean_apple_04` (7976-110124-0013): says "apples" and "apple"
- `clean_apple_05` (1472-142848-0003): says "apple and pear juice"
- `clean_plain_01` (460-172359-0063): same reader as most banana clips, no fruit
- `clean_plain_02` (7226-86964-0048): same reader as the pineapple clip, no fruit
