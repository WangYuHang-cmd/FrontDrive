"""Cheap diagnostics that run on frozen features (plan v4.1 section 3.1 "配套诊断", section 5 E3).

- ``d2_info_gap``: D2, how much more about the ego dynamic state / near future can be read out of the 8-frame
  (3.5 s) pretraining input than out of the 2-frame (0.5 s) deployment input of the Drive-JEPA encoder.
"""
