"""Turn per-second signals into one hype score and find the real moments.

What was learned from real streams (an 11-hour Kai Cenat VOD):
  * Raw chat speed is mostly noise: "hi" waves, "I was here", "lag?", polls (A/B/C), commands, and
    "where are you from" answers. Only reaction messages (laughing, shock, fail, hype) mark moments.
  * YouTube chat reacts 20 to 35 seconds AFTER the moment (stream delay plus reading time), so the
    chat signal has to be shifted back, or every clip starts after the moment is over.
  * The streamer's own voice (louder and brighter than their normal) peaks at the moment itself.
  * Game sounds that last (a helicopter overhead, a long gunfight) don't mark moments; only their
    onset can, and only weakly.
"""
import json
import re

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ---------------------------------------------------------------- chat

LAUGH = re.compile(
    r"\b(l+o+l+|l+m+f*a+o+|ha(ha)+h*|a?ha(ha)+|he(he)+|kek(w)?|omegalul|rofl|i'?m dead|crying|xd)\b"
    r"|:[a-z_-]*(tears_of_joy|rolling_on_the_floor|skull|crying|laugh|lol|lmao|kek)[a-z_-]*:"
    r"|💀|😂|🤣|😭|হা(হা)+|हा(हा)+")
SHOCK = re.compile(
    r"\b(wtf+|tf+|no+ ?wa+y+|bru+h+|o+m+g+|na+h+|damn+|wa+i+t+ ?what|a+yo+|she+sh+|holy|wo+w+|what+ the|"
    r"what+\?+|no{3,}|insane|crazy)\b|\?{3,}|!{3,}|😱|😳|🤯|:[a-z_-]*(wide-eyes|scream|exploding)[a-z_-]*:")
FAIL = re.compile(r"^(f+|rip+|l+|unlucky|fail|sold|throw)$|\b(rip|unlucky|failed|sold|throwing|game over|wasted)\b")
HYPE = re.compile(r"^(w+|gg+|ez+|clip)$|\b(clip (it|that)|let'?s go+|goat(ed)?|cracked|aura)\b|🔥|🐐")
NOISE = re.compile(
    r"^(!\S+.*|i was here.*|h+i+\b.*|hello+.*|hey+( \w+)?|yo+( \w+)?|wsg.*|lag+.*|no lag.*|laggy.*|[a-e1-9x]|\d+|[^flw]?)$"
    r"|https?://|\b(where('?s| is) .*from|first time|watching from)\b")
WEIGHT = {"laugh": 1.5, "shock": 1.2, "fail": 1.2, "hype": 0.6}


def is_noise(text):
    s = (text or "").strip().lower()
    return not s or bool(NOISE.search(s))


def classify_chat(text):
    """'laugh', 'shock', 'fail', 'hype', or None (noise or plain talk)."""
    s = (text or "").strip().lower()
    if is_noise(s):
        return None
    for name, rx in (("laugh", LAUGH), ("fail", FAIL), ("shock", SHOCK), ("hype", HYPE)):
        if rx.search(s):
            return name
    return None


def message_text(runs):
    """YouTube sends emoji as their own runs: keep the emoji itself, or ':name:' for channel emotes."""
    out = []
    for r in runs or []:
        if "text" in r:
            out.append(r["text"])
        elif "emoji" in r:
            e = r["emoji"]
            if e.get("isCustomEmoji"):
                out.append((e.get("shortcuts") or [":emote:"])[0])
            else:
                out.append(e.get("emojiId") or (e.get("shortcuts") or [""])[0])
    return "".join(out)


def read_chat(path, n):
    """Per-second reaction counts by kind, and every message that isn't noise [(t, text, kind or None)]
    (plain messages like "franklin hates you" often say what just happened)."""
    counts = {k: np.zeros(n) for k in WEIGHT}
    total = np.zeros(n)
    msgs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                o = json.loads(line)
                r = o["replayChatItemAction"]
                # replays carry the offset inside; yt-dlp's live chat puts it outside (from when it started)
                t = int(r.get("videoOffsetTimeMsec", o.get("videoOffsetTimeMsec"))) / 1000
                item = r["actions"][0]["addChatItemAction"]["item"]
                m = item.get("liveChatTextMessageRenderer") or item.get("liveChatPaidMessageRenderer")
                if not m:
                    continue
                text = message_text(m.get("message", {}).get("runs"))
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            s = int(t)
            if not 0 <= s < n:
                continue
            total[s] += 1
            if is_noise(text):
                continue
            kind = classify_chat(text)
            if kind:
                counts[kind][s] += 1
            msgs.append((t, text.strip()[:60], kind))
    return counts, total, msgs


def chat_burst(counts, smooth=10, baseline=600):
    """How far reaction chat is above its own normal for that part of the stream (log ratio, >= 0)."""
    react = sum(counts[k] * w for k, w in WEIGHT.items())
    sm = np.convolve(react, np.ones(smooth), "same")
    base = rolling_median(sm, baseline, step=10)
    return np.clip(np.log1p(sm) - np.log1p(base), 0, None)


# ---------------------------------------------------------------- audio

def rolling_median(x, w, step=5):
    """Median of a centered window around every point (subsampled by `step` so 11 hours stay fast)."""
    n = len(x)
    w = max(3, min(w, n if n % 2 else n - 1)) if n > 3 else 1
    pad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.median(sliding_window_view(pad, w)[:n, ::step], axis=1)


def robust_z(x, w=180):
    med = rolling_median(x, w)
    mad = rolling_median(np.abs(x - med), w) * 1.4826 + 1e-3
    return (x - med) / mad


def voice_arousal(feat):
    """How worked up the streamer sounds: voice-band level and brightness above their own normal of
    the last few minutes, counted only while someone is talking or yelling."""
    talk = feat.get("speech")
    if talk is None:  # no sound model: use voice-band level alone
        talk = np.ones_like(feat["voice"])
    talk = np.clip(np.maximum(talk, feat.get("reaction", 0 * talk)) * 1.5, 0, 1)
    vz = np.clip(robust_z(feat["voice"]), 0, None)
    bz = np.clip(robust_z(feat["bright"]), 0, None)
    return smooth(vz * talk + 0.5 * bz * (talk > 0.2), 5)


def onset(x, w=60):
    """A sound that just started (above its median of the last minute), not one that keeps going."""
    return np.clip(x - rolling_median(x, w), 0, None)


def smooth(x, w):
    return np.convolve(x, np.ones(w) / w, "same") if w > 1 else x


def scale(x, floor=0.0, cap=3.0):
    """Divide by the stream's own 99.5th percentile (at least `floor`), so signals are comparable,
    then cap, so one freak second (the stream starting, a mic bump) can't outweigh everything else.
    Use floor=0.3 for classifier probabilities: a laugh at 0.05 is noise even if it's this stream's loudest."""
    top = max(float(np.percentile(x, 99.5)) if len(x) else 0.0, floor)
    return np.minimum(x / top, cap) if top > 1e-9 else np.zeros_like(x)


# ---------------------------------------------------------------- combine

def estimate_lag(chat, arousal, default=15, lo=3, hi=50, peaks=40):
    """Seconds between a moment and chat reacting to it: where the streamer's voice peaks, on average,
    before the biggest reaction-chat bursts."""
    tops = pick_peaks(chat, peaks, 120)
    if len(tops) < 10:
        return default
    offs = np.arange(-hi, -lo + 1)
    n = len(arousal)
    prof = np.array([np.mean([arousal[t + o] for t in tops if 0 <= t + o < n] or [0]) for o in offs])
    prof = smooth(prof, 5)
    if prof.max() < 1.5 * (np.median(arousal) + 1e-6):  # no clear lead: fall back
        return default
    return int(-offs[int(np.argmax(prof))])


def shift_back(x, lag):
    """Move a signal `lag` seconds earlier (chat at t+lag describes the moment at t)."""
    if lag <= 0:
        return x
    return np.concatenate([x[lag:], np.zeros(lag)])


WEIGHTS = {"chat_reactions": 1.0, "voice": 0.8, "both": 0.6, "laugh_scream": 0.4, "game_action": 0.15,
           "visual": 0.35, "viewer_bookmarks": 0.8, "community_clips": 1.2, "already_clipped": 1.0, "most_replayed": 1.0,
           "facecam_motion": 0.4,
           "words": 0.3}


def combine(signals, weights=None):
    """One score per second. Chat reactions and the streamer's voice lead, and a moment where both
    fire together ("both") gets a bonus; laughs/screams, game-sound onsets, what happens on screen,
    viewer bookmarks, Twitch community clips and most-replayed add to it.
    `weights` (from learn.py, in ~/Clipper/weights.json) multiplies the defaults per signal."""
    n = len(next(iter(signals.values())))
    w = dict(WEIGHTS)
    for k, mult in (weights or {}).items():
        if k in w:
            w[k] *= float(mult)
    get = lambda k: signals.get(k, np.zeros(n))
    c, v = get("chat_reactions"), get("voice")
    score = w["both"] * np.sqrt(c * v)
    for k, x in signals.items():
        score = score + w.get(k, 0.3) * x
    edge = min(180, n // 20)  # intros ("hi chat", the mic coming on) and outros are rarely moments
    if edge:
        score[:edge] *= np.linspace(0.2, 1.0, edge)
        score[-edge:] *= 0.6
    return score


def clip_signal(clips, n):
    """Twitch community clips: viewers clipping a moment, weighted by the clip's views.
    clips = [{"offset": seconds into the VOD, "duration": s, "views": v}]"""
    sig = np.zeros(n)
    for c in clips:
        a = int(c.get("offset") or 0)
        b = a + int(c.get("duration") or 30)
        if 0 <= a < n:
            sig[a:min(n, b)] += 1 + min(np.log1p(int(c.get("views") or 0)), 12.0)  # cap: one mega-view clip isn't everything
    return sig


def pick_peaks(score, top, spacing):
    order = np.argsort(-score)
    taken = []
    for i in order:
        if score[i] <= 0:
            break
        if all(abs(i - j) >= spacing for j in taken):
            taken.append(int(i))
            if len(taken) >= top:
                break
    return taken


def moment_end(score, peak, longest=30):
    """Where the reaction has died down: score under a third of the peak for 4 seconds in a row."""
    thr = score[peak] / 3
    n = len(score)
    for t in range(peak + 3, min(n - 4, peak + longest)):
        if (score[t:t + 4] < thr).all():
            return t + 2
    return min(n - 1, peak + longest)


def level_bar(x, top):
    k = int(round(5 * min(1.0, max(0.0, x / top)))) if top > 0 else 0
    return "▮" * k + "·" * (5 - k)


# ---------------------------------------------------------------- viewer bookmarks (comments)

_TS = re.compile(r"(?<![\d:])(\d{1,2}):([0-5]\d)(?::([0-5]\d))?(?![\d:])")
# comments that carry a timestamp but aren't about a moment: requests, giveaways, self-promotion,
# "watching from", sponsor talk. On a MrBeast video these were a good share of the timestamps.
SPAM = re.compile(
    r"\b(ps ?[345]|playstation|xbox|console|gift|give ?away|giv(e)? me|send me|please (give|notice|pin|reply|heart)|"
    r"notice me|pin (this|me)|heart (this|me)|subscribe|sub to|my channel|check out my|watching from|"
    r"from (india|bangladesh|pakistan|nigeria|egypt|brazil|philippines|indonesia)|my dream|dream (gift|come)|"
    r"who('?s| is) (here|watching)|anyone (here|watching|in 20)|in 20\d\d|first comment|early squad|"
    r"sponsor|promo ?code|discount|link in)\b", re.I)


def comment_timestamps(text, duration):
    """[(seconds, snippet)] for each time a comment mentions ("2:31:05 when he crashed", "45:10 lmao"),
    within the video. The snippet is the text that goes with that one time, not the whole comment."""
    found = []
    for m in _TS.finditer(text or ""):
        h, mi, se = m.groups()
        t = int(h) * 3600 + int(mi) * 60 + int(se) if se else int(h) * 60 + int(mi)
        if 0 < t <= duration:
            found.append((t, m.start()))
    out = []
    for i, (t, pos) in enumerate(found):
        end = found[i + 1][1] if i + 1 < len(found) else len(text)
        snippet = text[pos:end].strip(" /|-,\n")
        if len(snippet) < 12 and i == 0:  # "Wow 11:07:03": the words came before the time
            snippet = text.strip()
        out.append((t, " ".join(snippet.split())[:100]))
    return out


def comment_signal(comments, n):
    """Viewers bookmarking moments in the comments, weighted by likes. A comment that lists many
    timestamps counts much less per timestamp than one that points at a single moment.
    Returns (per-second signal, [(t, likes, snippet)])."""
    sig = np.zeros(n)
    marks = []
    for c in comments or []:
        text = (c.get("text") or "").strip()
        ts = comment_timestamps(text, n - 1)
        if not ts or SPAM.search(text):
            continue
        likes = int(c.get("like_count") or 0)
        w = (1 + np.log1p(likes)) / (np.sqrt(len(ts)) if len(ts) <= 5 else len(ts))
        if likes == 0:  # nobody agreed with it: a hint, not a vote
            w *= 0.4
        for t, snippet in ts:
            if len(re.sub(r"[\d:\s.,!?]", "", snippet)) < 2 and likes < 5:  # a bare "12:30" with no words or likes
                continue
            sig[max(0, t - 3):min(n, t + 20)] += w  # people stamp just before or at the moment
            marks.append((t, likes, snippet))
    return sig, marks


# ---------------------------------------------------------------- reading a moment

def arc_length(score, moment, floor=0.4):
    """How long the hype stays up around a moment (seconds), on a 15-second average."""
    s = smooth(score, 15)
    thr = floor * score[moment]
    a = b = moment
    while a > 0 and s[a - 1] >= thr and moment - a < 120:
        a -= 1
    while b < len(s) - 1 and s[b + 1] >= thr and b - moment < 120:
        b += 1
    return b - a, a, b


def moment_kind(chat_kinds, voice, laugh, label=""):
    """A first guess at what kind of moment this is, for Claude to confirm or correct."""
    total = sum(chat_kinds.values())
    top = max(chat_kinds, key=chat_kinds.get) if total else None
    if top == "fail" and chat_kinds["fail"] >= 3:
        return "fail or death"
    if top == "laugh" and chat_kinds["laugh"] >= 0.5 * total:
        return "funny"
    if top == "shock":
        return "shock"
    if top == "hype":
        return "hype or skill"
    if laugh > 1:
        return "streamer laughing"
    if voice > 1:
        return "streamer reaction"
    return "talk"


def words_signal(segs, n, profile=None):
    """Reaction and game words said, per second, from a full transcript (analyze.py --full).
    Repeats and screams in words ("no no no", "oh my god") light up seconds the audio alone can miss."""
    from profiles import keyword_score
    sig = np.zeros(n)
    for s in segs:
        k = keyword_score(s["text"], profile)
        if k:
            a, b = int(s["start"]), int(max(s["start"] + 1, s["end"]))
            sig[max(0, a):min(n, b)] += k
    return smooth(sig, 5)
