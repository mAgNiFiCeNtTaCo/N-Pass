# N-Pass

N-Pass sits between your microphone and OBS. It listens to your voice a fraction of a second
before OBS gets it. When it hears one of your blocked words, it silences that word. The rest of
your voice passes through unchanged.

It is made for singing along to Spotify on stream. It runs on your own PC. Your voice is never
sent anywhere.

Written 2026-11-23. Everything under "What has been measured" was measured on this PC on that day.

## What it does and does not do

It does:

- Hold your voice back by about half a second (the "look-ahead"), so there is time to cut a word
  before it goes out.
- Recognise speech on your graphics card and compare each word with your own list of blocked words.
- Silence a matching word with a short fade in and out. No beep, no replacement voice.
- Optionally follow the song you are playing on Spotify, look up its lyrics, and watch the lines
  that contain one of your words more closely.

It does not:

- Catch every word. **It can miss a word, or cut it too late, so that part of it is heard.**
  The numbers are in the next section. Treat N-Pass as a safety net, not as a guarantee.
- Change anything in OBS by itself. You point OBS at the filtered microphone once, by hand.
- Remove a word from the music. It only filters your microphone.
- Send audio, or text of what you said, over the network.

### What has been measured

So far N-Pass has only been measured on 25 short "stand-in" clips: audiobook readers saying
"banana" or "pineapple", with those two words as the blocked list. The clips were played into the
filter at normal speed, on the graphics card, with the default settings.

**Not tested yet:** real singing, music in the background, your voice, your microphone, your
words, and a real stream through the virtual cable and OBS. Lyric mode has not been tried with a
real Spotify account. Until you have run your own clips (see "Testing with your own voice"), you
do not know how well it works for you.

**Words spoken at normal speed** (9 words, each 0.4 to 0.8 seconds long):

| Look-ahead | Removed at least 90% | Completely silent | Start of the word heard |
|---|---|---|---|
| 450 ms | 9 of 9 | 5 of 9 | 4 of 9, up to 78 ms |
| 550 ms | 9 of 9 | 7 of 9 | 2 of 9, up to 31 ms |
| 600 ms | 9 of 9 | 7 of 9 | 2 of 9, up to 31 ms |

These are from one run per setting, after the last change to the filter. Earlier runs at 550 ms
gave 5 and 6 of 9 completely silent, so expect this to vary from run to run.

**Words held long** (9 words, 0.7 to 2.3 seconds long). These are the same recordings with the
word slowed down by software, to stand in for a word that is held while singing. They are not real
singing.

| Look-ahead | Removed at least 90% | Partly heard | Missed |
|---|---|---|---|
| 450 ms | 5 of 9 | 3 of 9 | 1 of 9 |
| 550 ms | 5 of 9 | 4 of 9 | 0 of 9 |
| 600 ms | 5 of 9 | 4 of 9 | 0 of 9 |

At 550 ms, five of the nine held words were still heard for more than 0.2 seconds, and only three
were completely silent. The worst one was heard for almost 2 of its 2.3 seconds. **Held words are
the weak spot.**

Why held words go wrong: N-Pass has to hear most of a word before it knows which word it is. On
these clips it ordered the cut 0.23 to 0.53 seconds after a spoken word began, and 0.38 to 1.53
seconds after a held word began. The look-ahead is at most 0.6 seconds. So when a word is held
longer than that, its beginning has already gone out by the time N-Pass recognises it. A longer
look-ahead cannot fix this. Once N-Pass has recognised a held word, it keeps cutting until your
voice pauses (1.5 seconds at most), so the rest of the word is removed.

For songs where you hold the word, the safe choice is lyric mode with **Mute the whole risky line**
ticked (Settings tab). That does not wait for recognition. It has not been tested on real songs.

**Clips without a blocked word** (7 clips, five of them saying "apple" or "apples"): nothing was
cut by mistake, in any run. In the 600 ms run, the cut over two held words ran 3 and 32 ms past
the marked end of the word.

**Graphics card memory:** 816 MB for N-Pass itself.

The full tables are in `out\standin-results.md`.

## Before you start

You need Windows 10 or 11, an NVIDIA graphics card, and OBS. N-Pass uses about 0.8 GB of the
graphics card's memory.

On a fresh copy (a new PC, or after downloading the folder), open a terminal in the N-Pass folder
and run these two commands once. They need `uv` (https://docs.astral.sh/uv/). The second one downloads the speech models (about 0.5 GB):

    uv sync
    uv run python -m npass.detect download

## Step 1: install the virtual cable

N-Pass hands the filtered voice to OBS through a "virtual audio cable". That is a small free
driver that adds a pretend speaker and a pretend microphone to Windows. Whatever is played into
the pretend speaker comes out of the pretend microphone. N-Pass does not work without it.

1. Go to https://vb-audio.com/Cable/ and download VB-CABLE for Windows. It comes as a zip file.
   It is donationware: free to use, and you can pay what you want.
2. Right-click the zip file and choose **Extract All**. The installer does not work from inside
   the zip.
3. In the extracted folder, right-click `VBCABLE_Setup_x64.exe` and choose
   **Run as administrator**. Click the install button and wait until it says it is done.
4. Restart the PC.

After the restart, Windows has two new sound devices:

- **CABLE Input** (a playback device). N-Pass plays the filtered voice into this one.
- **CABLE Output** (a recording device). OBS listens to this one.

Windows may make the cable your default speaker or microphone after the install. If your sound
stops coming out of your speakers, open Windows **Settings > System > Sound** and choose your own
speakers and microphone again.

## Step 2: start N-Pass

Double-click `N-Pass.cmd` in the N-Pass folder. A small window opens, and an icon appears
near the clock. The first start takes a few seconds while the speech model loads.

Now choose your microphone:

1. Click the **Settings** tab.
2. Under **Microphone**, pick the microphone you stream with.

N-Pass looks for a microphone with "NVIDIA Broadcast" in its name first. If there is none, it
falls back to the Windows default microphone and tells you so in a yellow line. Picking your microphone under **Microphone** makes that line go away.

Never pick **CABLE Output** as the microphone here. That is the end OBS listens to, and N-Pass
would be listening to itself.

Talk. The **Mic** bar on the **Live** tab should move.

Closing the window with the X does not stop N-Pass. It keeps running next to the clock. To really
close it, right-click the icon and choose **Quit**.

## Step 3: point OBS at the filtered microphone

N-Pass never opens or changes OBS's own files. You make this change yourself, once.

1. In OBS, open **Settings > Audio**.
2. Under **Mic/Auxiliary Audio**, choose **CABLE Output (VB-Audio Virtual Cable)**.
3. Click **OK**.

If your microphone is a source inside a scene instead: right-click it, choose **Properties**, and
choose the same name under **Device**.

Make sure your real microphone is not also in OBS. If it is, viewers hear the unfiltered voice on
top of the filtered one, and every blocked word goes out. Set any other microphone entry in
**Settings > Audio** to **Disabled**, and remove or mute microphone sources in your scenes.

Leave the **Sync Offset** of the N-Pass microphone at 0.

## Step 4: put sound and picture back in sync

Your voice now reaches OBS late, by the delay N-Pass shows on the **Live** tab:

    Delay: 587 ms (look-ahead 550 ms)

Use the first number (yours will differ a little). Everything else in OBS has to be held back by
the same amount, or your voice lags behind the music and your lips.

N-Pass shows these steps with your own number filled in: **Settings** tab, **OBS sync** button.

**Other sound (desktop audio, game, music, alerts)**

1. In the OBS **Audio Mixer**, open the menu on any source and choose
   **Advanced Audio Properties**.
2. In the **Sync Offset** column, type the delay number for every source except the N-Pass
   microphone.

**Camera**

1. Right-click the camera source and choose **Filters**.
2. Click **+** and add **Video Delay (Async)**.
3. Set **Delay** to the same number of ms.

For a game capture or screen capture, use the **Render Delay** filter instead. It stops at 500 ms,
so it cannot fully match a longer delay.

If you move the look-ahead slider later, do these steps again with the new number.

**The optional one-click way, for the sound sources only**

1. In OBS, open **Tools > WebSocket Server Settings**. Tick **Enable WebSocket server**. Click
   **Show Connect Info** and copy the password.
2. In N-Pass, **Settings** tab, click **OBS sync**.
3. Paste the password into **OBS WebSocket password** and click **Connect**.
4. N-Pass lists your OBS sound sources with the Sync Offset each one has now and what it would
   become. Untick any you want left alone. The N-Pass microphone is never changed.
5. Click **Apply**. N-Pass sets the offsets and reads each one back to check it.

**Undo last change** puts the old values back. The password is used once and is not saved. This
button only changes Sync Offsets of sound sources. The camera delay is still the three camera
steps above.

## Step 5: your blocked words

The list ships empty. Until you add words, N-Pass cuts nothing and shows
"Your blocked word list is empty. Add words in Settings."

1. **Settings** tab. Tick **Show words**.
2. Under **Blocked words, one per line**, type your words, one on each line.
3. Click **Save**.
4. Untick **Show words** again. The list is then shown as dots, so it cannot be read on stream.

N-Pass matches by sound, not by spelling. A plural, or a slightly different ending, is treated as
the same word. Words that sound close to a blocked word can be cut too. Test this with your own
clips (see "Testing with your own voice").

The list is stored as plain text in `%LOCALAPPDATA%\NPass\blocked_words.txt`.

## Step 6: lyric mode with Spotify (optional)

With lyric mode on, N-Pass asks Spotify which song is playing, fetches timed lyrics for it, and
marks the lines that contain one of your blocked words as risky. During a risky line it does two
things:

- It accepts a weaker match as a hit.
- If the recogniser has not checked a stretch of that line by the time it is about to go out,
  N-Pass mutes that stretch. You lose some of the line, but nothing unchecked gets through.

Lyric mode only helps for songs that have timed lyrics on lrclib.net. It has not been tested with
a real Spotify account yet (see "What has been measured").

Spotify makes every user of its data register a small "app" of their own. This takes about five
minutes, once. Spotify changes its pages now and then, so the button names below may differ a
little from what you see.

1. Open https://developer.spotify.com/dashboard and log in with your Spotify account. Accept the
   developer terms if it asks.
2. Create a new app (look for a button like **Create app**).
3. Fill in the form:
   - Name and description: anything, for example `N-Pass`.
   - **Redirect URI**: type exactly `http://127.0.0.1:8898/callback` and add it. It must match
     letter for letter: `http` and not `https`, the numbers `127.0.0.1` and not the word
     `localhost`, and no slash at the end.
   - If it asks which API you will use, choose **Web API**.
4. Save the app.
5. Open the app's settings page and copy the **Client ID**, a long line of letters and numbers.
   You do not need the "Client secret". Never paste the secret into N-Pass or anywhere else.
6. In N-Pass, **Settings** tab: paste it into **Spotify client ID**, then click
   **Sign in to Spotify**. Your browser opens Spotify's own page. Agree there. You have 2 minutes.

N-Pass never sees your Spotify password. At the time of writing (2026-11-23), Spotify's developer
pages say the account that owns the app needs Spotify Premium.

To stop using lyric mode, untick **Use Spotify lyrics to watch risky lines**. **Sign out** deletes
the saved sign-in.

If the browser says the redirect URI is invalid, the address in the Spotify app is not exactly
`http://127.0.0.1:8898/callback`. Fix it on the app's settings page and save.

## The window explained

**Live tab**

- **ON / OFF**: the big button. OFF means your voice still passes through with the same delay,
  but nothing is cut. The delay stays, so OBS stays in sync. The hotkey Ctrl+Alt+N does the same.
- **Mic**: how loud your microphone is. It turns grey and says **Cutting** while a word is being
  silenced.
- **Delay**: how late your voice reaches OBS, measured. This is the number for Step 4.
- **Look-ahead**: how long N-Pass holds your voice back, from 350 to 600 ms. Lower is less delay.
  Higher gives the filter more time to catch a word.
- **Caught**: how many cuts were made since N-Pass started, and the last five. Each word is shown
  as its first letter and dots. The number after it is how sure the match was, from 0 to 1.
  "line muted" means a risky lyric line was muted because it could not be checked in time.
- **Listening with**: which recogniser is running and how long one check takes.
- **Lyrics** and **Song**: what lyric mode is doing. "Risky line now" means the line being sung
  contains one of your words.

**Settings tab**

- **Microphone**: which microphone N-Pass filters.
- **How a caught word is removed**: **Silence** removes it completely. **Soft dip** turns it down
  a lot (to about 3% of its level) instead, which sounds less abrupt but leaves a faint trace.
- **Recognition runs on**: **Auto** uses the graphics card and falls back to the processor if the
  card cannot be used. **GPU** and **CPU only** force one of them. See "CPU-only mode" first.
- **Song lyrics**, **Spotify client ID**, **Sign in to Spotify**: Step 6.
- **Mute the whole risky line (safest, cuts more)**: off by default. When it is on, every lyric
  line that contains one of your words is silenced from start to end, whether or not N-Pass heard
  the word. This is the safest setting for held notes, where N-Pass often recognises the word too
  late. The price: your whole voice is gone for those lines. It needs lyric mode and a song with
  timed lyrics, and the lyrics come from a public site that anyone can edit, so a line can be
  missing or wrong.
- **Blocked words, one per line**, **Show words**, **Save**: Step 5.
- **Open catch log**: opens the list of everything that was caught.
- **OBS sync**: Step 4.

The icon near the clock is green when N-Pass is on and all is well, grey when it is off, yellow
when there is a warning, and red when recognition is not running or the cable is missing.
Right-click it for **Show N-Pass**, **Turn off** / **Turn on** and
**Quit**.

## What each warning means

| The window says | What it means | What to do |
|---|---|---|
| The virtual cable is missing, so OBS cannot hear the filtered mic. | VB-CABLE is not installed, or Windows has it disabled. N-Pass sends your voice nowhere. It never falls back to your speakers. | Do Step 1. The **How to fix** button opens this file. |
| Recognition is starting. This can take a few seconds. | The speech model is loading. | Wait. If it stays for more than a minute, look at "Troubleshooting". |
| Recognition is falling behind | The recogniser did not check the audio before it went out. Words in that stretch were not checked. Risky lyric lines in that stretch are muted. | If it flashes now and then, raise the look-ahead. If it stays, close programs that use the graphics card heavily. In CPU-only mode it is always shown. |
| Recognition is not running. Risky lyric lines are being muted. | The speech model crashed or did not load. N-Pass retries every 5 seconds. Meanwhile nothing is recognised. | Stop singing lines you care about until it clears. If it does not clear, restart N-Pass. |
| CPU mode is too slow to cut a word in time. It only mutes risky lyric lines. Use GPU mode for a stream. | N-Pass is running on the processor. | See "CPU-only mode". |
| No microphone with "NVIDIA Broadcast" in its name was found. Using "..." instead. | N-Pass picked the Windows default microphone. | Pick your microphone in **Settings > Microphone**. |
| The microphone "..." is not connected. N-Pass is sending silence and carries on by itself when it is back. | The microphone was unplugged or switched off. | Plug it back in. Nothing needs restarting. |
| "..." is set to 44100 Hz in Windows, so Windows is converting the sound to 48000 Hz. | The device runs at another sample rate. It works, with slightly more delay. | Optional: Windows Sound settings, that device, **Advanced**, choose 48000 Hz. |
| The total delay is ... ms, more than the 700 ms limit. Lower the look-ahead in N-Pass. | Look-ahead plus the delay of your sound devices is over 700 ms. | Lower the look-ahead slider, then redo Step 4. |
| The look-ahead was lowered to ... ms to keep the total delay under 700 ms. | The delay stayed over the limit, so N-Pass lowered the look-ahead by itself. | Redo Step 4 with the new delay number. |
| Spotify is not answering. Risky lines are timed from the last known place in the song and may be off. | N-Pass lost contact with Spotify in the middle of a song. It keeps following the song from where it last was. | Check your internet connection. Do not skip around in the song meanwhile. |
| Lyric mode had a problem. Risky lines are not being watched. | Lyric mode failed for the moment. Only the recogniser is protecting you. | If it stays, untick and tick **Use Spotify lyrics to watch risky lines**, or restart N-Pass. |
| N-Pass could not restart its audio. Close N-Pass and start it again. | Switching between GPU and CPU did not work. | Quit N-Pass from the icon near the clock and start it again. |
| Your blocked word list is empty. Add words in Settings. | There is nothing to filter. | Do Step 5. |
| The hotkey Ctrl+Alt+N is already used by another program. | Another program took that key combination first. | Use the ON / OFF button. |

## CPU-only mode

N-Pass can run its recogniser on the processor instead of the graphics card (**Settings >
Recognition runs on > CPU only**). It also switches to this by itself if the graphics card model
cannot load.

**CPU-only mode is not equal to GPU mode and does not work as a word filter.** The processor
recogniser names a word too late. Measured on the stand-in clips:

- It ordered the cut 0.54 to 0.85 seconds after a spoken word began. The graphics card recogniser
  took 0.23 to 0.53 seconds.
- At the default look-ahead of 550 ms, 1 of the 9 spoken words was completely silent. 4 lost
  at least 90% of their sound, and 5 were partly heard, for up to 0.31 seconds.
- At 450 ms, 1 of the 9 spoken words lost 90% of its sound.
- Of the 9 held words, none was caught at any look-ahead.

What it still does is the lyric-mode safety net. It can never check a risky lyric line in time, so
it mutes the whole line instead. In a test with hand-made lyric files, that silenced all 18 words,
and with them 70 of the 97 seconds of audio. Without lyric mode, or for a song without timed
lyrics, CPU-only mode protects very little.

In CPU-only mode the window always shows "Recognition is falling behind". Do not stream on
CPU-only mode and expect words to be cut.

## The catch log

Every cut is written to `%LOCALAPPDATA%\NPass\catches.log`. Paste that path into the address bar
of File Explorer, or click **Open catch log** on the **Settings** tab. The first click only shows
a reminder that the file holds your words in full. Click again to open it. A line looks like this:

    2026-11-22 21:14:03 | caught "yourword" (0.71, lyric window) | Artist - Song @ 1:42

Reading it from left to right:

- The date and time of the cut.
- The blocked word that matched. **This is written in full, so do not open this file on stream.**
- The score, from 0 to 1. Higher means a closer match.
- Why it was cut. `asr` means the recogniser heard it. `lyric window` means the recogniser heard
  it during a risky lyric line. `failsafe` means a risky line was muted without being checked, and
  the word is shown as `(unchecked lyric line)`.
- The song and the position in it, or `no song`.
- If the line ends with something like `| late 35 ms`, the cut started that much too late, and
  the very beginning of the word went out.

The log tells you what was cut. It cannot tell you what was missed. Only your own ears, or the
clip test below, can.

A second file in the same folder, `npass.log`, records problems the program ran into. It is the
file to look at, or to send along, when something does not work.

## Testing with your own voice

Record a few short clips of yourself, with and without your blocked words, and run them through
the filter. The report tells you for every clip what was caught, what was missed, and what was
cut by mistake. You also get the filtered clips to listen to.

`test_clips\README.md` explains it step by step. The short version:

    uv run python tools/run_clips.py test_clips

Do this before you rely on N-Pass on stream. It is the only test that uses your voice and your
words.

## Privacy

- Your voice is processed on this PC only. No audio, and no text of what you said, leaves it.
- With lyric mode on, N-Pass asks Spotify about once a second which song is playing and how far in
  it is. It can only read that.
- With lyric mode on, N-Pass sends the artist, title, album and length of the song to lrclib.net
  to fetch the lyrics. Fetched lyrics are kept on this PC so they are not asked for again.
- The **OBS sync** button talks to OBS on this same PC, and only when you click it.
- The one-time set-up command `uv run python -m npass.detect download` downloads the speech models
  from the internet. The running program does not download anything.
- There is no telemetry and no update check.
- Your settings, your word list, the catch log and the Spotify sign-in are in
  `%LOCALAPPDATA%\NPass`. The Spotify sign-in is locked to your Windows account.
- Recordings you put in `test_clips` stay there. Git ignores that folder.

## Troubleshooting

**OBS hears nothing from the microphone.** Check that N-Pass is running and the **Mic** bar moves.
Check that there is no red "virtual cable is missing" box. In OBS, check that the microphone is
**CABLE Output (VB-Audio Virtual Cable)** and that it is not muted in the Audio Mixer.

**Viewers hear my voice twice, or blocked words still go out although N-Pass says "Caught".** Your
real microphone is still in OBS next to the filtered one. See the end of Step 3.

**My voice is behind the music, or my lips are out of sync.** Do Step 4 again with the number the
window shows now.

**My voice sounds chopped although I said nothing wrong.** A word that sounds like one of your
blocked words was cut. Open the catch log to see what matched. Record that sentence as a test clip
and run it, to see how often it happens.

**A word got through.** That can happen (see "What has been measured"). Raise the look-ahead to
600 ms, turn lyric mode on for songs that have lyrics, and run your own clips to see how often it
happens with your voice.

**"Recognition is starting" never goes away.** The speech model did not load. On a fresh copy, run
`uv run python -m npass.detect download`. Otherwise look at the last lines of
`%LOCALAPPDATA%\NPass\npass.log`.

**"N-Pass is already running."** Only one copy can run. Look for its icon near the clock. It may
be hidden behind the small arrow there.

**Double-clicking N-Pass.cmd says "N-Pass is not set up yet".** Run `uv sync` in a terminal in the
N-Pass folder, then try again.

**The microphone list does not show my microphone.** Plug it in, wait a few seconds, and open the
list again. If Windows itself does not show it under **Settings > System > Sound**, N-Pass cannot
see it either.

## For the curious

- `npass\` is the program. `tools\run_clips.py` runs clips through it. `tests\` holds the
  automatic tests (`uv run pytest -q`).
- `test_clips\standin\` holds the stand-in clips. They are cut from LibriSpeech, a collection of
  audiobook recordings under the CC BY 4.0 licence. `test_clips\standin\SOURCES.md` lists every
  clip and its source.
- `out\standin-results.md` has the full measurement tables.
