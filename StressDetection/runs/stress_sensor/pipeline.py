"""
pipeline.py
===========
Minimal, readable end-to-end pipeline helper for StressDetectionV3.
"""

from __future__ import annotations


def build_v3_paragraph() -> str:
    return (
        "STRESS DETECTION\n"
        "The model's objective is to predict cow stress state in advance using multimodal "
        "time-series signals from THI (temperature-humidity index), neck temperature, and lying "
        "behavior. Each modality is encoded by a Bidirectional LSTM (BiLSTM) that reads the "
        "sequence forward and backward, together with step-to-step delta features so the model "
        "sees both level and rate of change. The three sensor embeddings are fused with "
        "multi-head self-attention across sensors, then combined with a cow identity embedding "
        "and passed through a small classifier to assign one of three supervised classes: "
        "Normal, At-Risk, or Stressed. The training pipeline aligns and normalizes streams, "
        "uses sliding windows, and shifts labels into the future so predictions act as early "
        "warnings rather than only mirroring the present. These stress outputs can feed "
        "higher-level farm workflows such as health monitoring, heat detection, welfare scoring, "
        "and production-related decisions."
    )


if __name__ == "__main__":
    print(build_v3_paragraph())
