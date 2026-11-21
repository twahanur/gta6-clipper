import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cut
import facecam
import hype

CAM = [0.0, 0.62, 0.406, 0.38]


def test_resolve_layout_modes_and_facecam_off():
    assert cut.resolve_layout({"facecam": CAM})[0]["mode"] == "stack"
    assert cut.resolve_layout({"facecam": None})[0]["mode"] == "crop"
    assert cut.resolve_layout({"facecam": None, "mode": "stack"})[0]["mode"] == "crop"  # stack needs a box
    assert cut.resolve_layout({"facecam": CAM}, facecam="off")[0]["mode"] == "fit"
    assert cut.resolve_layout({"facecam": CAM}, facecam="off", off_mode="crop")[0]["mode"] == "crop"
    segs = cut.resolve_layout({"facecam": CAM, "segments": [{"start": 0, "end": 39.8, "facecam": CAM},
                                                            {"start": 39.8, "facecam": None, "mode": "fit"}]})
    assert [s["mode"] for s in segs] == ["stack", "fit"] and segs[-1]["end"] is None


def test_segment_filters():
    vf, cap = cut.layout_filter(1920, 1080, {"facecam": CAM, "segments": [
        {"start": 0, "end": 39.8, "facecam": CAM, "focus_x": 0.66}, {"start": 39.8, "facecam": None, "mode": "fit"}]}, dur=53)
    assert "trim=start=0.000:end=39.800" in vf and "concat=n=2" in vf and vf.endswith("[base]")
    assert callable(cap) and cap(10) == cut.TOP_H + 70 and cap(45) > 1300  # captions move with the layout
    vf, cap = cut.layout_filter(1920, 1080, {"facecam": None, "mode": "fit"})
    assert "boxblur" in vf and cap > 1300


def test_wide_facecam_is_fitted_not_cropped():
    wide, _ = cut.layout_filter(1920, 1080, {"facecam": CAM})            # 780x410: much wider than the band
    square, _ = cut.layout_filter(1920, 1080, {"facecam": [0, 0.25, 0.206, 0.206]})  # close to the band's shape... cropped
    assert "force_original_aspect_ratio=decrease" in wide and "boxblur" in wide
    assert "boxblur" not in square
    forced, _ = cut.layout_filter(1920, 1080, {"facecam": CAM, "facecam_fit": "fill"})
    assert "boxblur" not in forced


def test_focus_x_zero_still_far_left():
    vf, _ = cut.layout_filter(1920, 1080, {"facecam": None, "focus_x": 0})
    assert "crop=606:1080:0:0" in vf


def test_full_camera_rules():
    W, H = 960, 540
    assert facecam.is_full_camera([(400, 100, 110, 120)], W, H)          # big face in the middle: podcast/IRL
    assert not facecam.is_full_camera([(40, 400, 120, 110)], W, H)       # fair-sized face at the bottom: face cam
    assert facecam.is_full_camera([(10, 80, 120, 140), (620, 100, 70, 80)], W, H)  # two people up high: camera
    assert not facecam.is_full_camera([(60, 190, 25, 30)], W, H)         # tiny face cam face


def test_same_face_and_union_box():
    assert facecam.same_face((100, 100, 25, 30), (115, 102, 25, 30))     # small face that moved a little
    assert not facecam.same_face((100, 100, 25, 30), (300, 100, 25, 30))
    x, y, w, h = facecam.union_box([(40, 345, 120, 110), (230, 365, 100, 105)], 960, 540)
    assert x == 0 and y + h == 540 and 360 <= x + w <= 400


def test_layout_segments_from_frame_kinds(monkeypatch=None):
    kinds = ["facecam"] * 40 + ["none"] + ["camera"] * 12 + ["facecam"]  # a 1-frame flash at the end is ignored
    orig_kind, orig_suggest, orig_imread = facecam.frame_kind, facecam.suggest, None
    import cv2
    orig_imread = cv2.imread
    try:
        facecam.frame_kind = lambda img: kinds[img]
        cv2.imread = lambda p: int(p)
        facecam.suggest = lambda paths: {"facecam": CAM, "focus_x": 0.5, "mode": "stack"} if int(paths[0]) < 40 \
            else {"facecam": None, "focus_x": 0.5, "mode": "fit"}
        segs = facecam.layout_segments([str(i) for i in range(len(kinds))], [i + 0.5 for i in range(len(kinds))],
                                       {"facecam": CAM, "focus_x": 0.5})
    finally:
        facecam.frame_kind, facecam.suggest, cv2.imread = orig_kind, orig_suggest, orig_imread
    assert [s["mode"] for s in segs] == ["stack", "fit"]
    assert segs[0]["start"] == 0.0 and 40 <= segs[1]["start"] <= 42


def test_spam_comments_are_not_bookmarks():
    comments = [{"text": "12:30 I don't have a PS5 😢", "like_count": 27},
                {"text": "Watching from Bangladesh 🇧🇩 PS5 Pro would be my dream gift 16:18", "like_count": 4},
                {"text": "15:15 MrBeast gaslighting Tareq to lose everything 😭", "like_count": 419},
                {"text": "3:35", "like_count": 0}]
    sig, marks = hype.comment_signal(comments, 1300)
    assert [m[0] for m in marks] == [915]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
