"""Twitch community clips for a VOD, through Twitch's official Helix API.

Viewers clip the moments they think are best, and every clip records where in the VOD it came from
(vod_offset) and how many views it got. That's the strongest "this moment is good" signal there is.

It needs your own free Twitch app (https://dev.twitch.tv/console/apps -> Register Your Application,
any name, OAuth redirect http://localhost, category "Application Integration"), then:
  export TWITCH_CLIENT_ID=...  TWITCH_CLIENT_SECRET=...
Without them, Twitch VODs are analyzed from the audio only.

Twitch has no official API for VOD chat replay, so Twitch chat isn't read.
"""
import datetime as dt
import json
import os
import re
import urllib.parse
import urllib.request

API = "https://api.twitch.tv/helix"


def vod_id(url):
    m = re.search(r"twitch\.tv/(?:[^/]+/)?videos?/(\d+)", url or "")
    return m.group(1) if m else None


def credentials():
    cid, secret = os.environ.get("TWITCH_CLIENT_ID"), os.environ.get("TWITCH_CLIENT_SECRET")
    return (cid, secret) if cid and secret else None


def _http_json(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def app_token(cid, secret, http=_http_json):
    body = urllib.parse.urlencode({"client_id": cid, "client_secret": secret, "grant_type": "client_credentials"})
    return http("https://id.twitch.tv/oauth2/token", data=body.encode())["access_token"]


def parse_duration(s):
    """Helix durations look like '11h10m20s'."""
    total = 0
    for num, unit in re.findall(r"(\d+)([hms])", s or ""):
        total += int(num) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total


def community_clips(url, http=_http_json, max_pages=20):
    """[{offset, duration, views, title, url}] for clips made from this VOD, or None without credentials."""
    vid, creds = vod_id(url), credentials()
    if not vid or not creds:
        return None
    cid, secret = creds
    headers = {"Client-Id": cid, "Authorization": f"Bearer {app_token(cid, secret, http)}"}
    video = http(f"{API}/videos?id={vid}", headers=headers)["data"][0]
    start = dt.datetime.fromisoformat(video["created_at"].replace("Z", "+00:00"))
    end = start + dt.timedelta(seconds=parse_duration(video["duration"]) + 7 * 86400)  # clips come in for days
    q = {"broadcaster_id": video["user_id"], "first": 100,
         "started_at": start.isoformat().replace("+00:00", "Z"), "ended_at": end.isoformat().replace("+00:00", "Z")}
    out, cursor = [], None
    for _ in range(max_pages):
        page = http(f"{API}/clips?" + urllib.parse.urlencode({**q, **({"after": cursor} if cursor else {})}), headers=headers)
        for c in page.get("data", []):
            if c.get("video_id") == vid and c.get("vod_offset") is not None:
                out.append({"offset": int(c["vod_offset"]), "duration": float(c.get("duration") or 30),
                            "views": int(c.get("view_count") or 0), "title": c.get("title", ""), "url": c.get("url", "")})
        cursor = (page.get("pagination") or {}).get("cursor")
        if not cursor:
            break
    return out
