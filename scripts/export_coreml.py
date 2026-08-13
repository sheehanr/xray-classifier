from pathlib import Path

import coremltools as ct
import torch
import torch.nn as nn
from torchvision import models

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = (SCRIPT_DIR / "../models").resolve()
WTS_PATH = MODELS_DIR / "best_acc_model.pth"

CLASSES = [
    "xr_ankle",
    "xr_chest",
    "xr_elbow",
    "xr_finger",
    "xr_foot",
    "xr_forearm",
    "xr_hand",
    "xr_hip",
    "xr_humerus",
    "xr_knee",
    "xr_other",
    "xr_shoulder",
    "xr_wrist",
]


def load_model(classes, wts_path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(wts_path, map_location=torch.device("cpu"), weights_only=True))
    model.eval()

    return model


def main():
    model = load_model(CLASSES, WTS_PATH)
    dummy_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, dummy_input)
    classifier_config = ct.ClassifierConfig(CLASSES)

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="image", shape=dummy_input.shape)],
        classifier_config=classifier_config,
    )

    save_path = MODELS_DIR / "voxel_classifier.mlpackage"
    mlmodel.save(str(save_path))


if __name__ == "__main__":
    main()
