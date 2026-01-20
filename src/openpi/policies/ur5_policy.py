import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

def make_ur5_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "joints": np.random.rand(6),
        "gripper": np.random.rand(1),
        "base_rgb": np.random.randint(256, size=(3, 240, 320), dtype=np.uint8),
        "wrist_rgb": np.random.randint(256, size=(3, 240, 320), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")    
    return image

@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:

        # breakpoint()

        mask_padding = self.model_type == _model.ModelType.PI0

        # First, concatenate the joints and gripper into the state vector.
        # Pad to the expected input dimensionality of the model (same as action_dim).
        j = np.array(data["joints"])
        g = np.array(data["gripper"])
        # if g is 0D, convert to 1D
        if g.ndim == 0:
            g = np.expand_dims(g, axis=0)
        state = np.concatenate((j, g), axis=0)
        state = transforms.pad_to_dim(state, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        # Create inputs dict.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Since there is no right wrist, replace with zeros
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Since the "slot" for the right wrist is not used, this mask is set
                # to False
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Pad actions to the model action dimension.
        if "actions" in data:
            # The robot produces 7D actions (6 DoF + 1 gripper), and we pad these.
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        # Since the robot has 7 action dimensions (6 DoF + gripper), return the first 7 dims
        return {"actions": np.asarray(data["actions"][:, :7])}