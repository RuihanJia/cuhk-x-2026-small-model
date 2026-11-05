"""Versioned per-view MViTv2 CLS plus a small signed temporal-token residual.

The original torchvision forward runs once. Its final normalized non-CLS tokens
are observed with a temporary hook; no backbone or cross-view gate is copied.
"""
from pathlib import Path
import copy

import torch
from torch import nn

ARCHITECTURE = "MVIT_V2_S_ACTIVITY_TOKEN_READOUT_V1"
SOURCE_FORMAT = "ACTIVITY_VIDEO_ONLY_V1"
VARIANTS = {"CONTROL", "TOKEN_DIFF"}
READOUT_CONFIG = dict(
    version="SIGNED_ADJACENT_TOKEN_ATTENTION_V1", token_shape=[8, 7, 7, 768],
    projection_dim=64, position="learned_absolute_time",
    difference="separate_positive_and_negative_adjacent_same_spatial_position",
    pooling="independent_learned_attention_for_appearance_and_nonlinear_difference",
    residual="zero_initialized_128_to_40_linear", max_new_parameters=500000,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class SignedTokenReadout(nn.Module):
    """Keep intermediate time points: this is not a mean of signed differences."""
    def __init__(self):
        super().__init__()
        self.input_norm = nn.LayerNorm(768)
        self.projection = nn.Linear(768, 64)
        self.time_position = nn.Parameter(torch.empty(1, 8, 1, 1, 64))
        self.difference_position = nn.Parameter(torch.empty(1, 7, 1, 1, 64))
        self.appearance_score = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 1))
        self.difference_mlp = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 64))
        self.difference_score = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 1))
        self.residual = nn.Linear(128, 40)
        nn.init.normal_(self.time_position, std=.02)
        nn.init.normal_(self.difference_position, std=.02)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        require(sum(p.numel() for p in self.parameters()) <= 500000, "READOUT_PARAMETER_CAP")

    @staticmethod
    def attention_pool(tokens, score):
        values = tokens.flatten(1, 3)
        # FP32 softmax avoids excessive concentration/overflow under FP16 AMP.
        weights = score(values).float().softmax(dim=1).to(values.dtype)
        return (weights * values).sum(dim=1)

    def representation(self, tokens):
        require(tokens.ndim == 3 and tokens.shape[1:] == (392, 768), "FINAL_NONCLS_TOKEN_SHAPE")
        projected = self.projection(self.input_norm(tokens)).reshape(-1, 8, 7, 7, 64)
        appearance = projected + self.time_position.to(projected.dtype)
        delta = projected[:, 1:] - projected[:, :-1]
        signed = torch.cat((delta.clamp_min(0), (-delta).clamp_min(0)), dim=-1)
        difference = self.difference_mlp(signed) + self.difference_position.to(projected.dtype)
        return torch.cat((self.attention_pool(appearance, self.appearance_score),
                          self.attention_pool(difference, self.difference_score)), dim=-1)

    def forward(self, tokens):
        return self.residual(self.representation(tokens))


class ActivityTokenModel(nn.Module):
    def __init__(self, video, variant):
        super().__init__()
        require(variant in VARIANTS, "UNKNOWN_TOKEN_ARM")
        require(isinstance(video.head, nn.Linear) and video.head.in_features == 768
                and video.head.out_features == 40, "SOURCE_CLASSIFIER_STRUCTURE")
        self.video = video
        self.variant = variant
        # Head initialization must not shift the shared dropout/shuffle RNG.
        with torch.random.fork_rng(devices=[]):
            self.readout = SignedTokenReadout() if variant == "TOKEN_DIFF" else None

    def forward(self, inputs, return_components=False, auxiliary_backbone_stopgrad=False):
        require(not auxiliary_backbone_stopgrad or (return_components and self.readout is not None),
                "STOPGRAD_REQUIRES_TOKEN_COMPONENTS")
        if self.readout is None:
            return self.video(inputs)
        captured = []
        def capture(module, arguments, output):
            captured.append(output)
        handle = self.video.norm.register_forward_hook(capture)
        try:
            logits = self.video(inputs)
        finally:
            handle.remove()
        require(len(captured) == 1 and captured[0].ndim == 3
                and captured[0].shape[1:] == (393, 768), "MVIT_FINAL_TOKEN_CONTRACT")
        token_logits = self.readout(captured[0][:, 1:])
        combined = logits + token_logits
        if auxiliary_backbone_stopgrad:
            # Detach only the input to this second call of the same readout.
            # The combined loss retains its original attached backbone route.
            auxiliary_logits = self.readout(captured[0][:, 1:].detach())
            return combined, token_logits, auxiliary_logits
        return (combined, token_logits) if return_components else combined

    def optimizer_groups(self, backbone_lr, head_lr, readout_lr):
        groups = [
            dict(params=[p for n, p in self.video.named_parameters() if not n.startswith("head.")],
                 lr=backbone_lr, group_name="backbone"),
            dict(params=list(self.video.head.parameters()), lr=head_lr, group_name="original_cls_head"),
        ]
        if self.readout is not None:
            groups.append(dict(params=list(self.readout.parameters()), lr=readout_lr,
                               group_name="token_difference_readout"))
        flattened = [id(p) for g in groups for p in g["params"]]
        require(len(flattened) == len(set(flattened))
                and set(flattened) == {id(p) for p in self.parameters()}, "OPTIMIZER_PARAMETER_COVERAGE")
        return groups


def validate_state(state, expected, storage_dtype=torch.float16):
    require(set(state) == set(expected), "CHECKPOINT_STATE_KEYS")
    for name, value in state.items():
        require(isinstance(value, torch.Tensor) and value.shape == expected[name].shape,
                "CHECKPOINT_STATE_SHAPE:" + name)
        if value.is_floating_point():
            require(value.dtype == storage_dtype and torch.isfinite(value).all().item(),
                    "CHECKPOINT_STATE_DTYPE_OR_FINITE:" + name)
        else:
            require(value.dtype == expected[name].dtype, "CHECKPOINT_INTEGER_DTYPE:" + name)


def load_activity_source(root, spec):
    """Accept only the bound completed Activity_LOCAL checkpoint and provenance."""
    from torchvision.models.video import mvit_v2_s
    from research_runtime.common import sha, read, assert_disjoint
    root = Path(root)
    require(spec["source_format"] == SOURCE_FORMAT, "SOURCE_FORMAT")
    for path_key, hash_key in [
        ("source_path", "source_sha256"), ("source_done_path", "source_done_sha256"),
        ("source_provenance_path", "source_provenance_sha256"),
    ]:
        require(sha(root / spec[path_key]) == spec[hash_key], "SOURCE_FILE_HASH:" + path_key)
    done = read(root / spec["source_done_path"])
    provenance = read(root / spec["source_provenance_path"])
    require(done["status"] == "COMPLETE" and done["scientific_training_complete"] is True
            and done["kind"] == "VIDEO_ACTIVITY_VIEW_TRAINING", "SOURCE_NOT_COMPLETE_ACTIVITY")
    require(done["epochs"] == 6 and done["amp_skips"] == 0, "SOURCE_TRAINING_CONTRACT")
    require(done["checkpoint_sha256"] == spec["source_sha256"]
            and done["artifacts"]["model_fp16.pt"] == spec["source_sha256"]
            and done["artifacts"]["PROVENANCE.json"] == spec["source_provenance_sha256"],
            "SOURCE_ARTIFACT_BINDING")
    require(done["protocol_sha256"] == spec["source_protocol_sha256"], "SOURCE_DONE_PROTOCOL")
    old = done["spec"]
    require(old["fold"] == spec["fold"] and old["use_activity_view"] is True
            and old["activity_method"] == spec["activity_method"]
            and old["epochs"] == 6 and old["eval_epochs"] == [6], "SOURCE_ACTIVITY_SPEC")
    require(sorted(old["train_users"]) == sorted(spec["train_users"])
            and sorted(old["held_users"]) == sorted(spec["held_users"]), "SOURCE_SPLIT")
    require(old["source_sha256"] == spec["source_initialization_sha256"], "SOURCE_PARENT_BINDING")
    require(old["frame_index_sha256"] == spec["frame_index_sha256"], "SOURCE_FRAME_INDEX")
    require(provenance["identity"] == done["identity"]
            and provenance["identity"]["spec"] == old
            and provenance["identity"]["protocol_sha256"] == spec["source_protocol_sha256"],
            "SOURCE_PROVENANCE_BINDING")
    cp = torch.load(root / spec["source_path"], map_location="cpu", weights_only=True)
    require(set(cp) == {"state_dict", "protocol_sha256", "train_users", "initialization_sha256"},
            "SOURCE_CHECKPOINT_KEYS")
    require(cp["protocol_sha256"] == spec["source_protocol_sha256"]
            and cp["initialization_sha256"] == spec["source_initialization_sha256"],
            "SOURCE_CHECKPOINT_LINEAGE")
    require(sorted(cp["train_users"]) == sorted(spec["train_users"]), "SOURCE_TRAIN_USERS")
    assert_disjoint(cp["train_users"], spec["held_users"])
    video = mvit_v2_s(weights=None)
    video.head = nn.Linear(768, 40)
    validate_state(cp["state_dict"], video.state_dict())
    video.load_state_dict({k: v.float() if v.is_floating_point() else v
                           for k, v in cp["state_dict"].items()}, strict=True)
    video.eval().requires_grad_(False)
    return video


def checkpoint_metadata(model, protocol_sha256, spec):
    return dict(architecture=ARCHITECTURE, variant=model.variant,
                readout_config=copy.deepcopy(READOUT_CONFIG) if model.readout is not None else None,
                protocol_sha256=protocol_sha256, train_users=spec["train_users"],
                initialization_sha256=spec["source_sha256"], source_format=SOURCE_FORMAT)


def validate_token_payload(payload, model, protocol_sha256, spec):
    expected = checkpoint_metadata(model, protocol_sha256, spec)
    require(set(payload) == set(expected) | {"state_dict"}, "TOKEN_CHECKPOINT_KEYS")
    require(all(payload[k] == v for k, v in expected.items()), "TOKEN_CHECKPOINT_METADATA")
    validate_state(payload["state_dict"], model.state_dict())


def save_token_model(path, model, protocol_sha256, spec):
    """Versioned FP16 inference checkpoint, then exact FP16-to-FP32 reload."""
    from dual_host_runtime.io import save_torch
    state = {}
    for name, value in model.state_dict().items():
        require(not value.is_floating_point() or torch.isfinite(value).all().item(),
                "NONFINITE_TRAINED_STATE:" + name)
        state[name] = value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
    payload = dict(state_dict=state, **checkpoint_metadata(model, protocol_sha256, spec))
    validate_token_payload(payload, model, protocol_sha256, spec)
    save_torch(path, payload)
    require(Path(path).stat().st_size <= 70000000, "VIDEO_CHECKPOINT_SIZE_ALLOCATION")
    reload_token_model(path, model, protocol_sha256, spec)


def reload_token_model(path, model, protocol_sha256, spec, expected_sha256=None):
    from research_runtime.common import sha
    if expected_sha256 is not None:
        require(sha(path) == expected_sha256, "TOKEN_CHECKPOINT_HASH")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_token_payload(payload, model, protocol_sha256, spec)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model


def load_token_model(path, spec, expected_sha256, protocol_sha256):
    """Strict deployment/replay loader for the versioned whole model."""
    from torchvision.models.video import mvit_v2_s
    video = mvit_v2_s(weights=None)
    video.head = nn.Linear(768, 40)
    model = ActivityTokenModel(video, spec["arm"])
    reload_token_model(path, model, protocol_sha256, spec, expected_sha256)
    return model.eval().requires_grad_(False)

