# Testing N-Pass with your own voice

N-Pass has so far only been measured on the stand-in clips in `standin\`. Those are audiobook
readers saying "banana" and "pineapple". They are not you, they are not singing, and they are not
your words. The only way to know how well N-Pass works for you is to record a few clips and run
them through it. That takes about 15 minutes.

Everything you put in this folder stays on this PC. Git ignores this folder (apart from this file
and `standin\`), so your recordings and your words are never uploaded with the code.

## 1. What to record

Record with the microphone you stream with, in the room you stream in. Each clip can be short,
5 to 20 seconds. A good first set is about 12 clips:

- **Spoken, with a blocked word** (3 clips). Say a normal sentence that has the word in the middle.
- **Sung, with a blocked word** (4 clips). Sing along to a song the way you do on stream. Have the
  music in your headphones, not on speakers, so the clip holds your voice only.
- **Sung, with the word held long** (2 clips). Stretch the word out, the way a chorus does.
- **Without any blocked word** (3 clips). Talk and sing normally. Add words that sound close to
  a blocked word. These clips show whether N-Pass cuts things it should leave alone.

If your list has several words, make sure each word is in at least two clips.

## 2. Format

Any normal sound file works: `.wav`, `.mp3`, `.flac` or `.m4a`. A phone voice memo is fine, but a
recording from your stream microphone tells you more. You do not need to convert anything.

Windows has a recorder built in: open the Start menu, type `Sound Recorder`, press the red button.

## 3. Where to put them

Put the files straight into this folder:

    N-Pass\test_clips\

## 4. The little text file next to each clip

N-Pass has to know where the blocked word really is, so it can tell you if it missed it.
For each clip, make a text file with the **same name** ending in `.txt`:

    chorus take 1.mp3
    chorus take 1.txt

Open the `.txt` in Notepad. Write one line for every time a blocked word is said:

    start end word

`start` and `end` are seconds from the beginning of the clip. For example, if you say the word
from 3.2 seconds to 3.8 seconds, and again from 9.1 to 10.4 seconds:

    3.2 3.8 yourword
    9.1 10.4 yourword

Write the word the way it is written in your blocked word list.

To find the times, open the clip in any player that shows the time, or in the free program
Audacity, where you can see the word in the picture of the sound. Being off by a tenth of a second
is fine. It is better to mark slightly too wide than too narrow.

For a clip with **no** blocked word, make no `.txt` file at all, or leave it empty.

Optional: if you have synced lyrics for a clip (an `.lrc` file whose times start at the beginning
of the clip), give it the same name too, like `chorus take 1.lrc`. N-Pass then treats lines with a
blocked word as risky lines, the way lyric mode does live.

## 5. Run it

Close N-Pass first, so the two do not share the graphics card. Then open a terminal in
the N-Pass folder and run:

    uv run python tools/run_clips.py test_clips

It uses your own blocked word list and your own settings. The clips are played through the filter
at normal speed, so a 10 second clip takes a little over 10 seconds. No sound comes out of your
speakers and the microphone is not used.

## 6. Reading the result

For each clip you get lines like these:

    chorus take 1.mp3  (12.4 s)
      expected 2, caught 1, part played 0, MISSED 1, wrong cuts 0, late cuts 0
      caught      word 1 at 3.20-3.80 s: 100.0% of its sound removed, audible for 0 ms
      MISSED      word 1 at 9.10-10.40 s: 0.0% of its sound removed, audible for 1300 ms

- **caught**: at least 90% of the word's sound was removed.
- **part played**: some of the word was removed, but a piece of it could be heard.
- **MISSED**: the word went through.
- **WRONG CUT**: N-Pass silenced something that was not a blocked word.
- **late cuts**: the order to cut arrived after the start of the word had already been sent out.

Your words are not printed. The report says `word 1`, `word 2`, counted from the top of your list.
Add `--show-words` to the command if you want to see them and what the recogniser heard.

The filtered versions of your clips are saved in `out\clips\`, with `.filtered.wav` at the end of
the name. Listen to them: that is what your viewers would have heard. The full report is in
`out\clips\report.txt`.

To try another look-ahead, add it to the command. More look-ahead gives the filter more time:

    uv run python tools/run_clips.py test_clips --lookahead 600

## The stand-in clips

`standin\` holds 25 short clips of audiobook readers, used to test N-Pass before anyone had
recorded anything. `standin\SOURCES.md` says where each one comes from. To run them:

    uv run python tools/run_clips.py test_clips/standin --words pineapple,banana --show-words
