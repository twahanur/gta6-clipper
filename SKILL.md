---
name: gta6-clipper
description: Turns any long stream, podcast or video (YouTube, Twitch or Kick) into ready-to-post vertical clips. Works for GTA 6 and any other game, IRL and Just Chatting streams, podcasts and interviews. It finds the biggest streams on any topic, downloads only the audio, builds a hype curve from loudness, YouTube's most-replayed graph and chat speed, transcribes the hot moments locally with Whisper, picks the best moments with a harsh filter, then cuts 9:16 clips with the face cam on top and the main video underneath (or one full-frame crop when there's no face cam), a 3-second hook and word-by-word captions, plus a gallery page with captions and hashtags. Use it when the user says "set up the clipper" (or "set up the GTA clipper"), "clip this stream", "clip this podcast", "find me today's biggest <topic> streams and clip them", "look at what's working", or asks to make Shorts/TikToks/Reels from any stream, podcast or long video.
---

# GTA 6 Clipper

You turn a 4-hour stream, a 2-hour podcast or any long video into a handful of clips worth posting. GTA 6 is the headline use case, but nothing here is tied to one game or one kind of video. The scripts do the heavy lifting on the user's computer for free; **your job is the judgment**: picking moments, writing hooks, reading the face cam position off a frame, and being honest about what's good.

Paths used below:
- `SKILL` = this skill's folder (the folder this file is in)
- `PY` = `~/.clipper/venv/bin/python`
- `WORK` = `~/Clipper` (every stream, run and clip lives here)

Always run scripts as `cd "$SKILL/scripts" && "$PY" <script>.py ...`.

## 0. Setup ("set up the clipper")

1. Run `bash "$SKILL/scripts/setup.sh"`. It installs uv, Python 3.12, yt-dlp, faster-whisper, ffmpeg and Pillow into `~/.clipper` only, plus the free Whisper models. First run takes a few minutes; later runs are instant.
2. It ends with `READY`. If it fails, show the user the last lines and fix the cause (usually no internet, or the user declined an install).
3. Create `WORK/preferences.md` if it doesn't exist, with the default rules from section 3 under a heading "My clip rules". This is the file the weekly review updates.
4. Tell the user it's ready and the two things they can say next: "clip this stream: <link>" (any stream, podcast or long video) or "find me today's biggest <topic> streams and clip them" (for example GTA 6, Minecraft, a streamer's name or "podcast").

If `PY` doesn't exist when the user asks for anything else, run setup first.

## 1. Find streams (optional)

`"$PY" find_streams.py --query "<topic> stream" [--when today|week|month] [--max 8] [--min-minutes 45]`

Lists the most viewed long streams or videos for whatever the user asks about, with their length. Build the query from their words: "GTA 6 stream", "Fortnite stream", "Kai Cenat stream", "podcast full episode". If they don't name a topic, ask for one in one line. For podcasts use `--min-minutes 30`. Pick the top 2 or 3 unless the user names streamers. Long VODs (2h+) from big streamers are best. Streams that are live right now are skipped by default; clip them once the VOD is up (on YouTube that's usually right after the stream ends).

## 2. Listen (one stream at a time)

`"$PY" analyze.py "<url>" [--top 40] [--model small] [--visual] [--deep] [--full] [--game gta]`

- Downloads only the audio (a few hundred MB, not the video), then reads it once for per-second signals: the streamer's voice (how much louder and brighter than their own normal they are, which is how yelling and laughing sound), laughs and screams, and game sounds (gunfire, explosions, crashes, sirens) from a free sound classifier (YAMNet, `sound_events.py`). Only the moment a game sound starts counts, not a helicopter that hovers for three minutes. The signals are cached in `features.npz`, so a re-run is fast. Skip the classifier with `--no-sounds`.
- On YouTube VODs of live streams it reads the chat replay and keeps only real reactions: laughing ("LMAO", 😂, 💀), shock ("WTF", "no way", "???"), fails ("F", "WASTED") and hype ("W", "clip it"). Greetings, "I was here", "lag?", polls (A/B/C), commands and "where are you from" answers are ignored. On a real 11-hour stream only about 1 message in 10 was a reaction, and the noise waves ("hi" floods, lag complaints, polls) were what made the old picks random. It measures how many seconds chat runs behind the stream (usually 15 to 35) and shifts chat back, so a moment is placed where it happened on screen, not where chat typed. Twitch and Kick get the audio signals only; say so if asked.
- On YouTube it reads the top 500 comments for timestamps viewers left ("2:31:05 when he crashed 💀"), weighted by likes. These are people bookmarking the best moments by hand. Skip with `--no-comments`.
- With `--visual` it also downloads a 144p copy (about the size of the audio) and watches the screen 4 times a second. Bursts of on-screen motion add to the score (a modest signal: on a real GTA stream they were 1.36x more common at the real moments than at random). The picture suddenly going dark or gray shows up as a `screen:` hint, because death and fail screens look like that, but so do cutscene fades and loading screens: check the frames before trusting it. Use it for gameplay streams when chat replay is missing (Twitch, Kick, chat turned off) or when the user wants every signal; it adds about 5 minutes on an 11-hour stream. Not useful for podcasts.
- On Twitch VODs, when the user has set `TWITCH_CLIENT_ID` and `TWITCH_CLIENT_SECRET` (their own free app at dev.twitch.tv, see `twitch.py`), it reads the VOD's community clips through Twitch's official API: the moments viewers clipped, with views. That's the strongest signal there is. Twitch has no official API for VOD chat, so Twitch chat isn't read.
- If `already_clipped.json` exists (from `fingerprint.py`, below), Shorts other clippers already posted from this stream count as a signal and show up on their candidates.
- It flags candidates where music plays most of the time (copyright claims): mute it or cut around it.
- `--full` transcribes everything first, so reaction and game words count toward the score every second, not just in the windows.
- `--deep` (after `bash "$SKILL/scripts/setup.sh" --deep`, ~1.5 GB) separates the voices from the game and music with Demucs on each candidate window and listens for laughing and yelling in the voices alone, which re-ranks the candidates; and it tells speakers apart, so transcripts show `A:` / `B:` and fast back-and-forth (arguments, banter) is flagged. About 20 seconds per candidate on a CPU, so offer it for podcasts and for streams the user cares most about.
- If `WORK/weights.json` exists (from the weekly `learn.py`), the signals are weighted by what this user's audience actually watched.
- Turns that into one hype score per second, takes the top 40 moments (at least 90 seconds apart, 30 on videos under 30 minutes) and transcribes 70 seconds around each one (45 before, 25 after) with Whisper on the user's computer.
- Gives each window a `words` count of reaction words in English, Bangla and Hindi ("oh my god", "bro what", "কি হলো", "क्या हुआ") and, when the title names a game, that game's words (`profiles.py`). Built-in games: `gta`, `valorant`, `cs2`, `fortnite`, `minecraft`, `freefire`, `pubg` (also BGMI), `cod` (Warzone), `apex`, `football` (FIFA, EA FC, eFootball). Force a list with `--game valorant`. Add a new game by adding an entry to `PROFILES`.
- Whisper runs on an NVIDIA GPU automatically when one works (several times faster), otherwise on the CPU. `CLIPPER_DEVICE=cpu` forces the CPU.
- Writes `WORK/streams/<id>/for_claude.md` (what you read), `candidates.json`, `hype.png` (show it, it's a good screen share) and `meta.json`. Each candidate has:
  - `MOMENT`: the second the stream peaks, and `<-- MOMENT` on that transcript line
  - `strength`: 0 to 100 against this stream's best moment
  - `looks like`: a first guess at the kind of moment (funny, fail or death, shock, hype or skill, streamer reaction, talk). Confirm it before using it as `moment_type`
  - `arc`: shown when the hype stays up for 45+ seconds. That's a story with a build-up, and a longer 45 to 90 second clip from the start of the arc may work better than a short one
  - `viewer bookmark`: a comment that pointed at this moment, with its likes (requests like "give me a PS5", "watching from...", self-promotion and bare times with no words or likes are already filtered out). On edited videos (no chat replay) bookmarks carry most of the ranking, so read them critically: a comment about something tiny on screen may not make a clip.

  - `already clipped by others`: a Short from another channel that used this moment, with its views. It proves the moment works; if you keep it, give it a different hook and angle so it isn't a copy
  - `why`: which signals fired (chat laughing/shocked with counts, streamer yelling, laugh/scream heard, the game sound)
  - `chat said`: what viewers typed right after it. Read it: it usually says what happened ("WASTED", "bro is floating LOL", "the bong saved him")
  - `suggested`: a starting cut on sentence boundaries, from the setup line (after a pause) to the first finished sentence after the reaction dies down
  - a bar on every transcript line (`▮▮▮▮▮` = as intense as the stream's best moment, `·····` = calm)

It also writes `report.html`: the timeline with every signal in its own lane and a card per candidate. Open it for the user (`xdg-open` / `open`) when they want to see why moments were picked. Transcripts are saved in `transcripts.json`, so re-running the analysis (say with `--visual` or `--chat-lag`) only transcribes windows it hasn't seen.

Speed: `--model base` is about 2x faster than `small` with slightly rougher transcripts. `--full` transcribes the entire stream (slow, but useful when a stream is quiet and talky, like a podcast). You can run analyze on several streams one after another in the background.

## 3. Pick (this is your job, be harsh)

Read every `for_claude.md` for this batch, then `WORK/preferences.md` (the user's own rules, which override the defaults below when they conflict).

**The clip is built around the MOMENT.** Every KEEP must contain its candidate's MOMENT second, start a few seconds before it (just enough setup to understand it) and end once the reaction dies down. Start from `suggested` and tighten. Don't drift to a different, quieter part of the 70-second window because it reads better as text: that's how clips end up feeling random.

The strongest moments have both: chat reacting (laughing or shocked) **and** the streamer reacting (yelling, laughing). A candidate where only one fired is weaker; one where chat only typed plain talk, or where the streamer is calmly telling a story with no chat reaction, is usually a CUT for gameplay streams. Use `chat said` to understand what happened on screen, since the transcript only has what the streamer said. A window that's all game dialogue at `·····` is a cutscene: cut it unless the streamer's own reaction is the clip. When a window's transcript is sparse (mostly yelling), re-transcribe just that minute before judging it.

For each candidate, decide KEEP or CUT. **A clip is KEEP only if all three are yes:**
1. **Would a stranger stop scrolling in 2 seconds?** The first line or the first visible beat has to be the reaction, the shock, the joke. Not the setup.
2. **Does it make sense with zero context?** No "wait, what happened earlier", no inside jokes, no references to a chat message we can't see.
3. **Does it pay off inside the clip?** It ends on the punchline, the reveal or the reaction, never halfway through a thought.

Also cut: anything that opens with filler ("uh, so yeah", "okay so"), long silent stretches, the streamer reading donations, sponsor reads, ads, music-only stretches (copyright claims), anything sexual, hateful or targeting a real person, and anything that only works as audio-off gameplay with no reaction. Never pick the same moment twice. Big laughter or screaming with no speech can still be a KEEP if its strength is high and the window around it explains itself.

Moment types to label: `first-reaction`, `freak-out`, `discovery`, `funny-fail`, `hot-take`, `skill-moment`, `rage`, `wholesome`, `glitch`, and for talk content (podcasts, interviews, Just Chatting) `story`, `advice`, `confession`, `argument`.

For talk content the clip lives in the words, so the first line has to be the hook (a bold claim, a number, a question), and the clip has to finish the thought. Cut anything that needs the question asked five minutes earlier.

Score every KEEP 1 to 10 for how hard it would stop a scroll. Keep only 7+. Aim for about 3 per stream, never pad with weak ones. If a stream has nothing good, say so; that's useful information, not a failure.

Set tight times: start 0.5 to 1 second before the first important word or reaction (and never after the MOMENT), end right after the payoff. Usually 15 to 45 seconds, never under 8 or over 90.

Write `WORK/picks.json` (overwrite it for each batch):

```json
{
  "picks": [
    {
      "source": "<stream id from meta.json>",
      "start": "01:23:41",
      "end": "01:24:09",
      "moment_type": "freak-out",
      "score": 9,
      "streamer": "Kai Cenat",
      "hook": "HE WAS NOT READY FOR THIS",
      "title": "Kai Cenat loses it at the GTA 6 police chase",
      "caption": "The second he saw the helicopter 💀 Credit: Kai Cenat (link in comments)",
      "hashtags": ["#gta6", "#kaicenat", "#gaming", "#shorts"],
      "why": "Screams in the first second, makes sense alone, ends on the crash"
    }
  ]
}
```

Hook rules: 3 to 7 words, ALL CAPS, the way a person would say it ("HE WAS NOT READY FOR THIS"), never a description ("STREAMER REACTS TO CHASE"), no fake claims about what's in the clip, no emoji (the caption fonts can't draw them). Titles under 60 characters and lead with the person and the moment. Captions are one honest line plus a credit to the streamer. 3 to 5 hashtags. Write the hook and caption in the same language the streamer speaks (the burned-in captions are in that language too), unless the user asks for another one. Bangla, Hindi, Arabic, Tamil, Telugu, Thai and most other scripts render correctly: the right Noto font downloads itself the first time a clip needs it. Chinese, Japanese and Korean aren't covered yet; for those, write the hook in English and say the burned-in captions may show boxes.

Before writing the file, show the user a short table (time, streamer, type, score, strength, hook, why) and the moments you cut with a one-line reason each. Then continue unless they want changes.

## 4. Cut

1. `"$PY" cut.py download` downloads only the minutes each pick needs (plus 1.5 seconds on each side), saves three frames per clip in `<run>/frames/` (`<key>_a.jpg` near the start, `_b` in the middle, `_c` near the end), and prints the run folder.
2. **Look at all three frames of every clip** (open them with your Read tool), then check `<run>/layout.json`. It comes pre-filled by the face cam detector (`facecam.py`), which also looks at one frame a second and splits the clip into `segments` where the layout changes. Each entry has `auto_confidence` and a `note`. Tested on real clips its boxes overlapped hand-made ones by 80% on average, and it now handles cut-out face cams with no border and two people side by side, but it still makes mistakes: always compare it with the frames. A layout entry:
   - `"facecam": [x, y, w, h]` as fractions of the image (0 to 1) around the face cam (tight to its edges, or around the people when it's a cut-out with no border), or `null` when there's none.
   - `"focus_x"`: where the action is horizontally (0 left, 0.5 center, 1 right). With a face cam inside the picture (a bottom-left cut-out, say), push `focus_x` away from it so the people don't show up a second time in the gameplay crop.
   - `"mode"`: `stack` (face cam band on top, gameplay under it; the default with a face cam), `crop` (one 9:16 crop at `focus_x`; the default without), or `fit` (the whole picture resized to the width over a blurred copy, nothing cut off: use it for two people far apart, wide shots, or anything a crop would ruin).
   - `"facecam_fit"`: `auto` (default: a face cam much wider or taller than the band, like two cut-out people side by side, is shown whole over a blurred copy so nobody is cut off; a normal 16:9 webcam fills the band), `fit` or `fill` to force it.
   - `"segments"`: when the layout changes inside the clip (gameplay, then a full-screen camera; a podcast cutting between cameras), a list of `{"start", "end", "facecam", "focus_x", "mode"}` in seconds from the clip's start; the last one runs to the end. Check the switch time against `<key>_sheet.jpg` (one frame every 2 seconds, left to right, top to bottom, from the clip's start) and fix it: the detector samples once a second, so it's often a second late. A layout flash under ~3 seconds (an edit's punch-in, a graphic) isn't worth a segment.
   Example: a face cam in the bottom-right corner of a 1920x1080 frame at pixels (1450, 700) to (1900, 1060) is `[0.755, 0.648, 0.234, 0.333]`. Podcasts and interviews are usually `facecam: null` with `focus_x` on the person speaking, or `mode: "fit"` when two people sit far apart.
   If the user says they don't want a face cam band ("no face cam", "facecam off"), render with `--facecam off`: every `stack` becomes `fit` (the whole picture resized, nothing cut off), or `crop` with `--off-mode crop`.
3. `"$PY" cut.py render --run "<run folder>" [--facecam on|off] [--off-mode fit|crop]` builds every clip in its layout (face cam band on top at 40% with gameplay underneath, a crop, or the whole picture fitted; segment by segment when the layout changes), the hook in a yellow box for the first 3 seconds, word-by-word karaoke captions from a fresh Whisper pass on the clip itself (the word being said lights up yellow; `--captions plain` keeps every word white), audio normalized to Shorts loudness, and saves to `<run>/clips/`. Every finished clip is also copied to `WORK/videos/` (or wherever `CLIPPER_VIDEOS` points, when the user wants the clips somewhere else) with a serial number that continues across all runs (`001_<hook>.mp4`, `002_...`); a re-rendered clip keeps its number. Point the user at `WORK/videos/` as the place their clips are. Then it checks each clip for length, a black face cam area, black gameplay and missing captions.
4. If a check fails, or the user wants a change, edit `<run>/layout.json` or `<run>/picks.json` (the copy inside the run folder, not `WORK/picks.json`) and re-render just that clip: `"$PY" cut.py render --run "<run>" --only 03`. A new hook, title or caption only needs the re-render. New times need `"$PY" cut.py download --run "<run>" --only 03` first, then a fresh look at its frames, then the re-render. If a clip prints `!! render failed`, the other clips are still fine; fix that one clip's layout or times and re-render it alone. A pick whose download failed stays in the run's `picks.json` marked `"status": "download failed"`; retry it with `"$PY" cut.py download --run "<run>" --only <key>` (YouTube sometimes refuses one download and cut.py already retries through YouTube's other feed).
5. Look at a frame from the start and the middle of each finished clip yourself (`~/.clipper/bin/ffmpeg -ss 2 -i clip.mp4 -frames:v 1 check.jpg`, then again at the midpoint) before you say it's done. If the face is cut off, the crop is wrong or the captions read wrong, fix it and re-render.

## Live streams (optional)

`"$PY" live.py "<live url>" [--every 300]` records a YouTube stream that's live right now (audio and live chat), and every 5 minutes scores what's new. Each strong moment is transcribed and appended to `WORK/streams/<id>/live_moments.md` within minutes, with what chat said. Times are stream time, so the usual picks.json + `cut.py` flow can clip them from the live DVR while the stream is still on. Run it in the background and read `live_moments.md` when the user asks what happened.

## Measuring and learning (optional)

- `"$PY" fingerprint.py --stream <id> --query "<streamer> <game>" ["<another search>"]` finds Shorts other clippers already posted from this stream by matching audio fingerprints, and saves where each one sits and its views (`already_clipped.json`). Matches under confidence 15 are dropped: they're usually a game cutscene that plays in someone else's video too. Run it before analyze.py, or re-run analyze.py after it.
- `"$PY" benchmark.py --stream <id>` scores how well the analyzer finds the moments people liked (Twitch clips, already-clipped Shorts, comment timestamps, or the user's own `labels.json` list of good times), against random picks, and what each signal adds. Use it when changing the analyzer or the weights. On the Kai Cenat test stream: 6 to 8 of the top 10 moments (depending on which signals were on) were ones viewers had bookmarked, against 2 for the old analyzer and 1 for random picks. With one stream and ~70 bookmarks, differences of 10 points are noise.

## 5. Post

`"$PY" gallery.py --run "<run folder>"` opens one page with every clip (best first), its title, caption and hashtags, copy buttons and a link to the exact second in the source stream.

Tell the user, every time, in one or two lines:
- Watch each clip before posting and cut anything that starts slow.
- Add something of their own (their hook, a reaction, a take). Channels of raw reposted clips get turned down for monetization; "the clips are the reach, you are the channel".
- Credit the streamer or creator, follow the clipping campaign's rules, and don't post the same clip twice on one account.

There is no auto-upload in this skill. Posting stays with the user so nothing goes out unreviewed.

## 6. Look at what's working (weekly)

`"$PY" learn.py --channel https://www.youtube.com/@<their handle>`

Reads the public views on their Shorts tab, matches each Short to the clips in `WORK/log.csv` by title, and reports average views by moment type and by streamer, plus the top 5 and bottom 5. It only matches Shorts whose title still contains the clip's title or hook, so tell users to keep that.

With 10+ matched clips made since the new analyzer, it also learns signal weights (which signals went with more views on this channel) into `WORK/weights.json`; the next analysis uses them automatically. Mention the biggest change in one line.

Then update `WORK/preferences.md` with 2 to 5 concrete rules from the data ("freak-outs do 2.1x my average, prefer them", "Kai clips beat xQc clips 3 to 1", "hooks that name the streamer beat hooks that don't"), and show the user what you changed. Section 3 reads this file every time, so the picks get better each week. Don't invent patterns from fewer than 10 matched clips; say there isn't enough data yet.

## Troubleshooting

- **"This stream is still live"**: analyze.py stops on live streams and on YouTube recordings that are still processing. Tell the user to come back once the stream has ended and the recording is up, or pick a different stream.
- **"Sign in to confirm you're not a bot" or 403 from YouTube:** set the browser to borrow cookies from (`export CLIPPER_COOKIES_BROWSER=chrome`, or safari/firefox) and run again. Also run setup again to update yt-dlp, which fixes most YouTube breakages.
- **No chat reactions:** only YouTube VODs of live streams have chat replay. The voice and sound signals still work. If the measured chat delay looks wrong (moments land after the action), pass `--chat-lag <seconds>`.
- **Most-replayed missing:** new or low-view videos don't have the graph yet. Loudness carries it.
- **A clip has no captions:** there was no clear speech (a pure scream). That's fine; the hook still shows.
- **Captions look wrong:** captions already use the `small` Whisper model. If a line is still wrong (loud music, people talking over each other), re-render that clip with `--only <key> --model medium` (slower, downloads once), or trim the clip around it.
- **Very slow transcribing:** use `--model base` or `--top 15`.
- Everything is in `WORK`. A stream's audio can be deleted after its clips are made.

## What not to do

- Never clip or post anything the user hasn't seen; never claim a clip is good without looking at it.
- Never pick from a stream the user doesn't have the right to clip if they tell you a campaign's rules forbid it.
- Never make up numbers about how a clip will perform.
