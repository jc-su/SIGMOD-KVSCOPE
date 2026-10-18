"""Shared plumbing for the consumption-plane GPU probes: the in-process engine environment, real-text
corpora, and vLLM construction defaults. Controller logic lives in observe/controller.py (do not
duplicate it in probes); the captured rows and cycles are read through observe/capture.py."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"     # in-proc engine: pass publishes our pid


PARA = ("The survey ship kept a log of every sounding, and the charts grew more honest with each "
        "season. Sandbars moved, lights were rebuilt, and the pilots amended their notes in the "
        "margins. What the harbor sold, in the end, was predictability: a captain who trusted the "
        "chart could carry more cargo and less fear. ")
