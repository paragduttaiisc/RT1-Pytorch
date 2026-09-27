"""PyTorch runtime for the OXE RT-1 (TF) policy used by SimplerEnv.

This module is a drop-in replacement for
``simpler_env.policies.rt1.rt1_model.RT1Inference`` (the TensorFlow
implementation) and has **zero TensorFlow at runtime**:

  * policy network: TF SavedModel -> ONNX (tf2onnx) -> PyTorch
    (onnx2pytorch), saved as ``models/rt1_pt_model.pt``;
  * language embedding: TF Hub ``universal-sentence-encoder-large/5``
    ported to PyTorch as ``use_v5_embed.pt`` (512-d output) with the full
    200004-entry hash-table vocabulary in ``use_v5_vocab.npz``, both
    vendored by the git submodule ``universal_sentence_encoder``
    (paragduttaiisc/universal_sentence_encoder_pytorch; see
    ``universal_sentence_encoder/use_embed.py``; embeddings match the
    TF Hub module to ~2e-7);
  * image preprocessing: ``tf.image.resize_with_pad`` re-implemented in
    PyTorch (``resize_with_pad`` below): float32 fit-within-target size
    math, bilinear half-pixel-centers interpolation, zero pad to the exact
    target size, truncating cast to uint8.  On SimplerEnv's 256x256 frames
    the resize is a no-op (centered with 32 px left/right zero padding), so
    the port is pixel-exact on the frames actually used.

Everything else -- the ~4800-node EfficientNetB3 + FiLM + 8-layer
transformer + token-learner policy, and the ring-buffer policy state
(``action_tokens`` / ``context_image_tokens`` / ``seq_idx``) -- runs in
PyTorch.
"""

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# The USE-large language embedder is vendored as the git submodule
# ``universal_sentence_encoder`` (repo
# paragduttaiisc/universal_sentence_encoder_pytorch). Its modules import
# each other by top-level name and resolve their data files relative to
# their own directory, so the submodule root is prepended to sys.path.
_USEN_SUBMODULE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "universal_sentence_encoder"
)
if _USEN_SUBMODULE not in sys.path:
    sys.path.insert(0, _USEN_SUBMODULE)

from use_embed import USEEmbedder

__all__ = ["RT1Inference", "resize_with_pad"]


# Initial policy state: verified all-zeros by TF get_initial_state(batch_size=1).
INITIAL_STATE = {
    "action_tokens": np.zeros((1, 6, 8, 1, 1), dtype=np.int32),
    "context_image_tokens": np.zeros((1, 6, 8, 1, 512), dtype=np.float32),
    "seq_idx": np.zeros((1, 1, 1, 1, 1), dtype=np.int32),
}

# SimplerEnv builds its time step via ts.transition(...) => step_type MIDDLE.
# The step_type input is inert inside the model (verified), but we pass the
# same value SimplerEnv would for fidelity.
_STEP_TYPE_MIDDLE = np.full(1, 2, dtype=np.int32)


def resize_with_pad(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """TF-free port of ``tf.image.resize_with_pad`` (uint8 HWC in, HWC out).

    Mirrors TF's exact semantics:
      1. fit-within-target size math in float32:
         ``ratio = max(w/tw, h/th)``, ``rh = floor(h/ratio)``,
         ``rw = floor(w/ratio)``;
      2. center the resized image: top pad ``floor((th-rh)/2)``,
         left pad ``floor((tw-rw)/2)``;
      3. bilinear resize with half-pixel centers (TF
         ``resize_bilinear(half_pixel_centers=True)`` == PyTorch
         ``F.interpolate(bilinear, align_corners=False)``);
      4. zero-pad bottom/right to exactly ``(target_height, target_width)``;
      5. cast to uint8 by truncation (matches ``tf.cast(float, tf.uint8)``).

    Verified against TF: pixel-exact on no-op/upscale cases (incl. the real
    256x256 SimplerEnv frames) and within 1/255 on a handful of pixels for
    downscaling cases (float32 rounding at the truncation boundary).
    """
    h, w, c = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
    f_h, f_w = np.float32(h), np.float32(w)
    f_th, f_tw = np.float32(target_height), np.float32(target_width)
    ratio = max(f_w / f_tw, f_h / f_th)
    rh_f = f_h / ratio
    rw_f = f_w / ratio
    rh = int(np.floor(rh_f).astype(np.int32))
    rw = int(np.floor(rw_f).astype(np.int32))
    p_h = max(0, int(np.floor((f_th - rh_f) / np.float32(2)).astype(np.int32)))
    p_w = max(0, int(np.floor((f_tw - rw_f) / np.float32(2)).astype(np.int32)))

    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    r = F.interpolate(x, size=(rh, rw), mode="bilinear", align_corners=False)
    r = r[0].permute(1, 2, 0)  # (rh, rw, c) float32

    out = torch.zeros(target_height, target_width, c, dtype=torch.float32)
    out[p_h:p_h + rh, p_w:p_w + rw] = r
    # tf.cast(float, tf.uint8) truncates toward zero (values lie in [0, 255]).
    return torch.floor(out).to(torch.uint8).numpy()


class RT1Inference:
    """RT-1 policy inference with an identical API to SimplerEnv's TF class.

    Usage:
        pol = RT1Inference(saved_model_path=..., lang_embed_model_path=...)
        pol.reset("pick up the coke can")
        raw_action, action = pol.step(image_uint8_hwc, task_description)
        raw_action, action, act_logits, tok_logits = pol.step_logits(
            image_uint8_hwc, task_description
        )
    """

    def __init__(
        self,
        pt_model_path: str,
        image_width: int = 320,
        image_height: int = 256,
        action_scale: float = 1.0,
        policy_setup: str = "google_robot",
    ) -> None:
        if policy_setup != "google_robot":
            raise NotImplementedError(
                f"PyTorch RT1Inference currently supports policy_setup='google_robot', got {policy_setup!r}"
            )
        self.image_width = image_width
        self.image_height = image_height
        self.action_scale = action_scale
        self.policy_setup = policy_setup
        self.action_rotation_mode = "axis_angle"

        use_embed_model_path = os.path.join(_USEN_SUBMODULE, "use_v5_embed.pt")
        use_vocab_path = os.path.join(_USEN_SUBMODULE, "use_v5_vocab.npz")
        print(f"[rt1_pytorch] loading USE embedder from {use_embed_model_path} ...")
        self.lang_embed_model = USEEmbedder(
            vocab_path=use_vocab_path, model_path=use_embed_model_path
        )

        print(f"[rt1_pytorch] loading PyTorch policy from {pt_model_path} ...")
        self.pt_model_path = pt_model_path
        self.policy = torch.load(pt_model_path, weights_only=False)
        self.policy.eval()

        # output-name -> index map, for name-based output mapping. Works for
        # both the original RT-1 policy (9 outputs) and RT-1-X (11 outputs,
        # which adds the mobile-base heads base_displacement_vector /
        # base_displacement_vertical_rotation).
        self._output_index = {name: i for i, name in enumerate(self.policy.output_names)}
        required = {
            "action/gripper_closedness_action", "action/rotation_delta",
            "action/terminate_episode", "action/world_vector",
            "state/action_tokens", "state/context_image_tokens", "state/seq_idx",
        }
        if not required.issubset(self._output_index):
            raise ValueError(
                f"policy output names missing {required - set(self._output_index)}: "
                f"{list(self.policy.output_names)}"
            )

        self.task_description: Optional[str] = None
        self.task_description_embedding: Optional[np.ndarray] = None
        self.policy_state: Dict[str, np.ndarray] = self._initial_state()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _initial_state_path(self) -> Optional[str]:
        """Sidecar file with the checkpoint's TF ``get_initial_state`` arrays,
        written at conversion time next to the .pt model."""
        if not self.pt_model_path.endswith(".pt"):
            return None
        return self.pt_model_path[:-3] + "_initial_state.npz"

    def _initial_state(self) -> Dict[str, np.ndarray]:
        """Initial policy state: the sidecar npz when present (exact TF
        ``get_initial_state(batch_size=1)`` values/shapes), else the classic
        all-zero state. Both checkpoints are all-zero; only the ring-buffer
        shapes differ (6x8 vs 15x11/15x81)."""
        sidecar = self._initial_state_path()
        if sidecar is not None and os.path.exists(sidecar):
            data = np.load(sidecar)
            return {
                "action_tokens": data["action_tokens"].astype(np.int32),
                "context_image_tokens": data["context_image_tokens"].astype(np.float32),
                "seq_idx": data["seq_idx"].astype(np.int32),
            }
        return {k: v.copy() for k, v in INITIAL_STATE.items()}

    def _reset_state(self) -> None:
        self.policy_state = self._initial_state()

    # ------------------------------------------------------------------
    # SimplerEnv-compatible API
    # ------------------------------------------------------------------
    def reset(self, task_description: str) -> None:
        self._initialize_task_description(task_description)
        self._reset_state()

    def _initialize_task_description(self, task_description: Optional[str]) -> None:
        if task_description is not None:
            self.task_description = task_description
            emb = self.lang_embed_model.embed(task_description).astype(np.float32)
            self.task_description_embedding = emb
        else:
            self.task_description = ""
            self.task_description_embedding = np.zeros((512,), dtype=np.float32)

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """TF-free port of SimplerEnv RT1Inference._resize_image."""
        return resize_with_pad(
            image, target_height=self.image_height, target_width=self.image_width
        )

    @staticmethod
    def _small_action_filter_google_robot(
        raw_action: Dict[str, np.ndarray], arm_movement: bool = False, gripper: bool = True
    ) -> Dict[str, np.ndarray]:
        """Copied from SimplerEnv RT1Inference._small_action_filter_google_robot."""
        if arm_movement:
            raw_action["world_vector"] = np.where(
                np.abs(raw_action["world_vector"]) < 5e-3,
                np.zeros_like(raw_action["world_vector"]),
                raw_action["world_vector"],
            )
            raw_action["rotation_delta"] = np.where(
                np.abs(raw_action["rotation_delta"]) < 5e-3,
                np.zeros_like(raw_action["rotation_delta"]),
                raw_action["rotation_delta"],
            )
            raw_action["base_displacement_vector"] = np.where(
                np.abs(raw_action["base_displacement_vector"]) < 5e-3,
                np.zeros_like(raw_action["base_displacement_vector"]),
                raw_action["base_displacement_vector"],
            )
            raw_action["base_displacement_vertical_rotation"] = np.where(
                np.abs(raw_action["base_displacement_vertical_rotation"]) < 1e-2,
                np.zeros_like(raw_action["base_displacement_vertical_rotation"]),
                raw_action["base_displacement_vertical_rotation"],
            )
        if gripper:
            raw_action["gripper_closedness_action"] = np.where(
                np.abs(raw_action["gripper_closedness_action"]) < 1e-2,
                np.zeros_like(raw_action["gripper_closedness_action"]),
                raw_action["gripper_closedness_action"],
            )
        return raw_action

    def _forward(
        self, image: np.ndarray, embedding: np.ndarray
    ) -> Tuple[
        Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]
    ]:
        """One policy forward pass in PyTorch.

        Returns (raw_action, new_state) using SimplerEnv's raw-action key
        names.
        """
        image_b = torch.from_numpy(image[None]).to(torch.uint8)
        emb_b = torch.from_numpy(embedding[None].astype(np.float32))
        at_b = torch.from_numpy(self.policy_state["action_tokens"].astype(np.int32))
        cit_b = torch.from_numpy(self.policy_state["context_image_tokens"].astype(np.float32))
        seq_b = torch.from_numpy(self.policy_state["seq_idx"].astype(np.int32))

        # Feed inputs by name in the model's own positional input order
        # (policy.input_names): the ONNX input order is set by the export
        # and can differ between checkpoints.
        with torch.no_grad():
            inputs = {
                "observation_image": image_b,
                "observation_natural_language_embedding": emb_b,
                "step_type": torch.from_numpy(_STEP_TYPE_MIDDLE),
                "action_tokens": at_b,
                "context_image_tokens": cit_b,
                "seq_idx": seq_b,
            }
            out = self.policy(*[inputs[n] for n in self.policy.input_names])

        # Map outputs by name (robust across the original RT-1 policy with
        # 9 outputs and RT-1-X with 11 outputs). raw_action key order follows
        # the TF signature output order so it matches SimplerEnv's
        # policy_step.action dict.
        oi = self._output_index

        def _np(name, dtype, batch=True):
            t = out[oi[name]]
            if batch:
                t = t[0]
            return t.detach().cpu().numpy().astype(dtype)

        raw_action = {}
        if "action/base_displacement_vector" in oi:
            raw_action["base_displacement_vector"] = _np(
                "action/base_displacement_vector", np.float32
            )
            raw_action["base_displacement_vertical_rotation"] = _np(
                "action/base_displacement_vertical_rotation", np.float32
            )
        raw_action["gripper_closedness_action"] = _np(
            "action/gripper_closedness_action", np.float32
        )
        raw_action["rotation_delta"] = _np("action/rotation_delta", np.float32)
        raw_action["terminate_episode"] = _np("action/terminate_episode", np.int32)
        raw_action["world_vector"] = _np("action/world_vector", np.float32)
        new_state = {
            "action_tokens": _np("state/action_tokens", np.int32, batch=False),
            "context_image_tokens": _np("state/context_image_tokens", np.float32, batch=False),
            "seq_idx": _np("state/seq_idx", np.int32, batch=False),
        }
        # Pre-ArgMax teacher logits, exposed only by the *_logits models
        # (porting/add_logits_outputs.py); None for the original .pt models.
        logits = None
        if "action_logits" in oi and "token_logits" in oi:
            logits = {
                "action_logits": _np("action_logits", np.float32),
                "token_logits": _np("token_logits", np.float32),
            }
        return raw_action, new_state, logits

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """SimplerEnv RT1Inference.step() semantics, PyTorch runtime.

        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str]; if different from the previous
                task description, policy state is reset.

        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action, keys:
                'world_vector' (3,), 'rot_axangle' (3,),
                'gripper' (1,), 'terminate_episode'
        """
        image = self._prepare_forward(image, task_description)
        raw_action, new_state, _ = self._forward(image, self.task_description_embedding)
        raw_action, action = self._postprocess_action(raw_action)

        # Update policy state (ring buffer carried across steps).
        self.policy_state = new_state

        return raw_action, action

    def step_logits(
        self, image: np.ndarray, task_description: Optional[str] = None
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
    ]:
        """step() plus the pre-ArgMax teacher logits (for distillation).

        Runs the identical forward pass as step() and advances the policy
        state the same way, and additionally returns the final dense-head
        logits that the checkpoint ArgMax's into action bin ids:

        Output:
            raw_action: dict; raw policy action output (filtered, as in step())
            action: dict; processed action (as in step())
            action_logits: float32 [n_act, bins] -- one row per action token
                of the most recent context slot; exactly the logits over
                which the bin ids are chosen, i.e.
                np.argmax(action_logits, axis=-1) reproduces the ids the
                model writes into state/action_tokens. Original RT-1 family:
                (8, 255) = world xyz (rows 0-2), rotation axis-angle
                (rows 3-5), gripper (row 6), 1 spare (row 7). RT-1-X:
                (11, 512).
            token_logits: float32 [n_tokens, bins] -- head output over every
                transformer token (original: 96 = 6 slots x (8 image + 8
                action placeholders); RT-1-X: 1380).

        Raises RuntimeError if the loaded .pt lacks the logit outputs.
        """
        image = self._prepare_forward(image, task_description)
        raw_action, new_state, logits = self._forward(
            image, self.task_description_embedding
        )
        if logits is None:
            raise RuntimeError(
                f"model {self.pt_model_path!r} has no pre-ArgMax logit outputs "
                "(missing 'action_logits'/'token_logits'); use a model built by "
                "porting/add_logits_outputs.py + porting/convert_pt_logits.py "
                "(default: models/rt1_logits_pt_model.pt)"
            )
        raw_action, action = self._postprocess_action(raw_action)

        # Update policy state (ring buffer carried across steps).
        self.policy_state = new_state

        return raw_action, action, logits["action_logits"], logits["token_logits"]

    def _prepare_forward(
        self, image: np.ndarray, task_description: Optional[str]
    ) -> np.ndarray:
        """Shared step()/step_logits() preamble: optional task reset,
        image resize, embedding check."""
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        assert self.task_description_embedding is not None

        return image

    def _postprocess_action(
        self, raw_action: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Small-action filter + SimplerEnv action dict; shared by
        step() and step_logits()."""
        if self.policy_setup == "google_robot":
            raw_action = self._small_action_filter_google_robot(
                raw_action, arm_movement=False, gripper=True
            )
        for k in raw_action.keys():
            raw_action[k] = np.asarray(raw_action[k])

        action = {}
        action["world_vector"] = (
            np.asarray(raw_action["world_vector"], dtype=np.float64) * self.action_scale
        )
        if self.action_rotation_mode == "axis_angle":
            action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
            action_rotation_angle = float(np.linalg.norm(action_rotation_delta))
            action_rotation_ax = (
                action_rotation_delta / action_rotation_angle
                if action_rotation_angle > 1e-6
                else np.array([0.0, 1.0, 0.0])
            )
            action["rot_axangle"] = action_rotation_ax * action_rotation_angle * self.action_scale
        else:
            raise NotImplementedError()

        raw_gripper_closedness = raw_action["gripper_closedness_action"]
        if self.policy_setup == "google_robot":
            action["gripper"] = np.asarray(raw_gripper_closedness, dtype=np.float64)
        else:
            raise NotImplementedError()

        action["terminate_episode"] = raw_action["terminate_episode"]

        return raw_action, action
