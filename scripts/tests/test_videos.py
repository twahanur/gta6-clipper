import os, sys, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import next_serial, publish_video

def test_serial_and_publish():
    with tempfile.TemporaryDirectory() as t:
        vids = os.path.join(t, "videos"); run = os.path.join(t, "run"); os.makedirs(run)
        assert next_serial(vids) == 1
        os.makedirs(vids)
        for n in ("001_a.mp4", "007_b.mp4", "notes.txt"):
            open(os.path.join(vids, n), "w").close()
        assert next_serial(vids) == 8
        clip = os.path.join(t, "01_hook.mp4"); open(clip, "w").write("v1")
        p1 = publish_video(clip, run, "01", "hook", vids)
        assert os.path.basename(p1) == "008_hook.mp4"
        # re-render with a new hook keeps serial 008 and removes the old file
        open(clip, "w").write("v2")
        p2 = publish_video(clip, run, "01", "new-hook", vids)
        assert os.path.basename(p2) == "008_new-hook.mp4"
        assert not os.path.exists(p1) and open(p2).read() == "v2"
        assert os.path.basename(publish_video(clip, run, "02", "x", vids)) == "009_x.mp4"

if __name__ == "__main__":
    test_serial_and_publish(); print("ok")
