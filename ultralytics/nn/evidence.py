"""Validated controls for the S0-S4 downstream evidence experiments."""

import math


EVIDENCE_EXPERIMENTS = {
    "S0": ("shared", False, False, "single"),
    "S1": ("shared", True, False, "single"),
    "S2": ("shared", False, True, "single"),
    "S3": ("independent", True, True, "mean_two_branches"),
    "S4": ("shared", True, True, "single"),
}


class FrequencyEvidenceOutput(tuple):
    """Keep independent loss evidence attached to the forward that produced it.

    Existing RT-DETR tensor/tuple consumers keep their positional interface. The
    shared path returns its original tuple without this wrapper.
    """

    def __new__(cls, predictions, frequency_evidence):
        result = super().__new__(cls, predictions)
        result.frequency_evidence = frequency_evidence
        return result

    def __reduce__(self):
        return type(self), (tuple(self), self.frequency_evidence)


def validate_evidence_config(cfg):
    """Return explicit experiment controls, or None for the unchanged legacy path.

    These sections live in model YAMLs, not Ultralytics' flat training-argument
    namespace. The runner expands training arguments separately.
    """
    schemas = {
        "experiment": {"id"},
        "frequency_evidence": {"mode", "acsa_enabled", "query_enabled", "independent_heads",
                               "independent_calibration", "init_from_shared"},
        "model_control": {"num_queries", "decoder_layers", "preserve_hsg_internal_gate"},
        "mask_loss": {"enabled", "weight", "bce_weight", "dice_weight", "reduction"},
        "train": {"epochs", "batch", "seed"},
    }
    if not any(name in cfg for name in schemas):
        return None
    for name, allowed in schemas.items():
        if name not in cfg or not isinstance(cfg[name], dict):
            raise TypeError(f"{name} must be a mapping in an evidence experiment")
        unknown = set(cfg[name]) - allowed
        if unknown:
            raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
    exp = cfg["experiment"]
    if exp.get("id") not in EVIDENCE_EXPERIMENTS:
        raise ValueError("experiment.id must be S0, S1, S2, S3 or S4")
    evidence = dict(cfg["frequency_evidence"])
    loss = cfg["mask_loss"]
    control = cfg["model_control"]
    for name in ("acsa_enabled", "query_enabled"):
        if type(evidence.get(name)) is not bool:
            raise TypeError(f"frequency_evidence.{name} must be a boolean")
    for name in ("independent_heads", "independent_calibration", "init_from_shared"):
        if name in evidence and type(evidence[name]) is not bool:
            raise TypeError(f"frequency_evidence.{name} must be a boolean")
    actual = (evidence.get("mode"), evidence["acsa_enabled"], evidence["query_enabled"], loss.get("reduction"))
    if actual != EVIDENCE_EXPERIMENTS[exp["id"]]:
        raise ValueError(f"{exp['id']} requires {EVIDENCE_EXPERIMENTS[exp['id']]}, got {actual}")
    independent = evidence["mode"] == "independent"
    for name in ("independent_heads", "independent_calibration"):
        if evidence.get(name, False) != independent:
            raise ValueError(f"frequency_evidence.{name} must be {independent}")
    if evidence.get("init_from_shared", False) and not independent:
        raise ValueError("init_from_shared applies only to S3")
    if type(control.get("preserve_hsg_internal_gate")) is not bool or not control["preserve_hsg_internal_gate"]:
        raise ValueError("all evidence experiments must preserve the HSG internal gate")
    for name, expected in (("num_queries", 1000), ("decoder_layers", 6)):
        if type(control.get(name)) is not int or control[name] != expected:
            raise ValueError(f"model_control.{name} must be {expected}")
    if type(loss.get("enabled")) is not bool or not loss["enabled"]:
        raise ValueError("all evidence experiments must retain mask supervision")
    for name, expected in (("weight", 0.5), ("bce_weight", 0.7), ("dice_weight", 0.3)):
        value = loss.get(name)
        if type(value) not in (float, int) or not math.isfinite(value) or value != expected:
            raise ValueError(f"mask_loss.{name} must match the frozen baseline value {expected}")
    if cfg.get("obb_loss", {}).get("lambda_mask", 0.5) != loss["weight"]:
        raise ValueError("mask_loss.weight and obb_loss.lambda_mask disagree")
    for name in ("epochs", "batch", "seed"):
        value = cfg["train"].get(name)
        if type(value) is not int or value < (0 if name == "seed" else 1):
            raise ValueError(f"train.{name} must be an integer in its valid range")
    freq = cfg.get("freq_guidance", {})
    freq_schemas = {
        "root": {"enabled", "mask_source", "semantic_mode", "query_mode", "fixed_quotas",
                 "acsa_freq_enabled", "semantic_calibration", "query", "acsa_betas", "mask_loss", "diagnostics"},
        "semantic_calibration": {"enabled", "mode", "alpha_init", "alpha_max", "detach_semantic_anchor",
                                 "lambda34", "lambda23", "lambda24"},
        "query": {"mode", "num_queries", "level_betas", "centered_mask", "boost_min", "boost_max",
                  "peak_kernel", "non_peak_weight", "min_quotas", "max_quotas", "fallback_quotas",
                  "prior", "evidence_topr", "evidence_eps", "evidence_weights", "active_threshold_std",
                  "capacity_multiplier", "dynamic_ratio_start", "dynamic_ratio_mid", "dynamic_ratio_final",
                  "dynamic_ratio_mid_epoch", "dynamic_ratio_final_epoch", "temperature_start",
                  "temperature_final", "temperature_final_epoch", "base_quotas", "extra_queries", "temperature"},
        "mask_loss": {"use_calibrated_masks", "scale_weights"},
        "diagnostics": {"enabled", "store_last_batch", "fallback_to_global_topk", "strict_train_masks"},
    }
    for name, allowed in freq_schemas.items():
        section = freq if name == "root" else freq.get(name, {})
        if not isinstance(section, dict):
            raise TypeError(f"freq_guidance.{name} must be a mapping")
        unknown = set(section) - allowed
        if unknown:
            raise ValueError(f"unknown freq_guidance.{name} fields: {sorted(unknown)}")
        for key, value in section.items():
            if key in {"enabled", "detach_semantic_anchor", "centered_mask", "acsa_freq_enabled",
                       "use_calibrated_masks", "store_last_batch", "fallback_to_global_topk", "strict_train_masks"}:
                if type(value) is not bool:
                    raise TypeError(f"freq_guidance.{name}.{key} must be a boolean")
    if not freq.get("enabled") or freq.get("mask_source", "wavelet") != "wavelet":
        raise ValueError("S0-S4 require the original wavelet feature and evidence path")
    semantic = freq.get("semantic_calibration", {})
    if (semantic.get("enabled") is not True or semantic.get("mode") != "residual_logit"
            or semantic.get("detach_semantic_anchor") is not True
            or freq.get("semantic_mode", "residual_logit") != "residual_logit"):
        raise ValueError("S0-S4 require residual-logit calibration with detached anchors")
    if (freq.get("query", {}).get("mode") != "scene_conditioned_dynamic"
            or freq.get("query_mode", "scene_conditioned_dynamic") != "scene_conditioned_dynamic"):
        raise ValueError("the guided query baseline must be scene_conditioned_dynamic")
    if freq.get("mask_loss", {}).get("use_calibrated_masks") is not True:
        raise ValueError("S0-S4 supervise calibrated logits, as in the baseline")
    if "acsa_freq_enabled" in freq and freq["acsa_freq_enabled"] != evidence["acsa_enabled"]:
        raise ValueError("acsa_freq_enabled conflicts with frequency_evidence.acsa_enabled")
    evidence.update(id=exp["id"], reduction=loss["reduction"])
    return evidence
