import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from sound_events import bin_scores

def test_bin_scores_separates_action_from_speech():
    fpb = 10 / 0.48
    n = int(fpb * 3) + 1
    probs = np.zeros((n, 521), dtype=np.float32)
    probs[: int(fpb), 0] = 0.9                   # bin 0: calm speech
    probs[int(2 * fpb):, 421] = 0.8              # bin 2: gunfire
    probs[int(fpb):int(2 * fpb), 11] = 0.7       # bin 1: screaming
    g = bin_scores(probs, fpb, 3)
    assert g["action"][2] > g["action"][0], g
    assert g["calm"][0] > g["calm"][2], g
    assert g["reaction"][1] > g["reaction"][0], g

if __name__ == "__main__":
    test_bin_scores_separates_action_from_speech(); print("ok")
