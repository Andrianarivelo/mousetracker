from collections import defaultdict

import numpy as np
import torch

from app.core.sam3_engine import SAM3Engine


class _FakeTracker:
    def __init__(self) -> None:
        self.received_masks: list[torch.Tensor] = []
        self.preflight_calls: list[dict] = []

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask, add_mask_to_memory=False):
        self.received_masks.append(mask.detach().cpu().clone())
        inference_state.setdefault("obj_ids", [])
        if obj_id not in inference_state["obj_ids"]:
            inference_state["obj_ids"].append(obj_id)
        video_res_masks = mask.unsqueeze(0).unsqueeze(0)
        return frame_idx, list(inference_state["obj_ids"]), None, video_res_masks

    def propagate_in_video_preflight(self, tracker_state, run_mem_encoder=True):
        self.preflight_calls.append(
            {"tracker_state": tracker_state, "run_mem_encoder": run_mem_encoder}
        )


class _FakeModel:
    def __init__(self) -> None:
        self.tracker = _FakeTracker()
        self.rank = 0
        self.world_size = 1
        self.device = torch.device("cpu")
        self.max_num_objects = 8
        self.masklet_confirmation_consecutive_det_thresh = 3
        self.actions: list[tuple[str, int, tuple[int, ...]]] = []
        self.cleared: list[tuple[int, int]] = []
        self.prepared_frames: list[int] = []
        self.inference_flags: list[bool] = []

    def _initialize_metadata(self):
        return {
            "obj_ids_per_gpu": [np.array([], np.int64)],
            "obj_ids_all_gpu": np.array([], np.int64),
            "num_obj_per_gpu": np.zeros(1, np.int64),
            "max_obj_id": -1,
            "obj_id_to_score": {},
            "obj_id_to_tracker_score_frame_wise": defaultdict(dict),
            "rank0_metadata": {
                "removed_obj_ids": set(),
                "suppressed_obj_ids": defaultdict(set),
                "masklet_confirmation": {
                    "status": np.array([], np.int64),
                    "consecutive_det_num": np.array([], np.int64),
                },
            },
        }

    def _get_gpu_id_by_obj_id(self, inference_state, obj_id):
        for rank, obj_ids in enumerate(inference_state["tracker_metadata"]["obj_ids_per_gpu"]):
            if obj_id in obj_ids:
                return rank
        return None

    def _prepare_backbone_feats(self, inference_state, frame_idx, reverse=False):
        self.prepared_frames.append(frame_idx)
        self.inference_flags.append(torch.is_inference_mode_enabled())

    def _assign_new_det_to_gpus(self, new_det_num, prev_workload_per_gpu):
        return np.zeros(new_det_num, dtype=np.int64)

    def _init_new_tracker_state(self, inference_state):
        return {"obj_ids": []}

    def add_action_history(self, inference_state, action_type, frame_idx, obj_ids):
        self.actions.append((action_type, frame_idx, tuple(obj_ids)))

    def _get_tracker_inference_states_by_obj_ids(self, inference_state, obj_ids):
        return [
            state
            for state in inference_state["tracker_inference_states"]
            if set(obj_ids) & set(state.get("obj_ids", []))
        ]

    def clear_detector_added_cond_frame_in_tracker(self, tracker_state, obj_id, refined_frame_idx):
        self.cleared.append((obj_id, refined_frame_idx))

    def _build_tracker_output(self, inference_state, frame_idx, override):
        return override or inference_state.get("cached_frame_outputs", {}).get(frame_idx, {})

    def _cache_frame_outputs(self, inference_state, frame_idx, obj_id_to_mask, suppressed_obj_ids=None):
        inference_state.setdefault("cached_frame_outputs", {})[frame_idx] = obj_id_to_mask

    def _postprocess_output(self, inference_state, out, suppressed_obj_ids=None):
        obj_id_to_mask = out["obj_id_to_mask"]
        obj_ids = np.array(sorted(obj_id_to_mask), dtype=np.int64)
        if len(obj_ids) == 0:
            return {
                "out_obj_ids": np.array([], dtype=np.int64),
                "out_binary_masks": np.zeros((0, 0, 0), dtype=bool),
                "out_probs": np.zeros((0, 1), dtype=np.float32),
                "out_boxes_xywh": np.zeros((0, 4), dtype=np.float32),
            }

        masks = np.stack(
            [
                obj_id_to_mask[obj_id].detach().cpu().numpy().squeeze(0).astype(bool)
                for obj_id in obj_ids
            ]
        )
        return {
            "out_obj_ids": obj_ids,
            "out_binary_masks": masks,
            "out_probs": np.ones((len(obj_ids), 1), dtype=np.float32),
            "out_boxes_xywh": np.zeros((len(obj_ids), 4), dtype=np.float32),
        }


class _FakePredictor:
    def __init__(self, state, model=None) -> None:
        self.model = model
        self._state = state
        self.requests: list[dict] = []

    def _get_session(self, session_id):
        return {"state": self._state}

    def handle_request(self, request):
        self.requests.append(request)
        return {"outputs": {"request": request}}


def _make_engine(state, predictor):
    engine = SAM3Engine()
    engine._loaded = True
    engine.session_id = "session"
    engine.predictor = predictor
    return engine


def test_add_mask_prompt_uses_full_tracker_mask():
    state = {
        "num_frames": 3,
        "cached_frame_outputs": {},
        "tracker_metadata": {},
        "tracker_inference_states": [],
    }
    model = _FakeModel()
    predictor = _FakePredictor(state, model=model)
    engine = _make_engine(state, predictor)

    mask = np.zeros((6, 7), dtype=bool)
    mask[1:5, 2:6] = True

    outputs = engine.add_mask_prompt(0, mask, obj_id=5, frame_shape=mask.shape)

    assert predictor.requests == []
    assert model.actions == [("add", 0, (5,))]
    assert model.prepared_frames == [0]
    assert model.inference_flags == [True]
    assert model.cleared == []
    np.testing.assert_array_equal(
        model.tracker.received_masks[0].numpy(),
        mask.astype(np.float32),
    )
    np.testing.assert_array_equal(outputs["out_obj_ids"], np.array([5], dtype=np.int64))
    np.testing.assert_array_equal(outputs["out_binary_masks"][0], mask)


def test_add_mask_prompt_falls_back_to_points_when_mask_api_missing():
    state = {"num_frames": 1, "cached_frame_outputs": {}}
    predictor = _FakePredictor(state, model=None)
    engine = _make_engine(state, predictor)

    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True

    outputs = engine.add_mask_prompt(0, mask, obj_id=2, frame_shape=mask.shape)

    assert len(predictor.requests) == 1
    request = predictor.requests[0]
    assert request["type"] == "add_prompt"
    assert request["obj_id"] == 2
    assert len(request["points"]) > 0
    assert 1 in request["point_labels"]
    assert outputs["request"] == request
