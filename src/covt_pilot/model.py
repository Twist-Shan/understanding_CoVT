"""Native generation and full-prefix replay, without fabricated thought slots."""
import re

import torch


def depth_positions(ids, token_id, start=0):
    positions = (ids[0] == token_id).nonzero(as_tuple=False).flatten()
    positions = positions[positions >= start]
    if len(positions) != 4:
        raise ValueError(f"Expected exactly 4 generated depth pads; found {len(positions)}. Do not insert or silently select slots.")
    return positions


def extract_answer(text):
    tags = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    candidate = tags[-1] if tags else text.strip().splitlines()[-1] if text.strip() else ""
    boxed = re.fullmatch(r"\\boxed\{\s*([AB])\s*\}[.!]?", candidate.strip())
    if boxed:
        return boxed.group(1)
    match = re.fullmatch(r"\s*([AB])[.!]?\s*", candidate)
    return match.group(1) if match else None


def language_norm(model):
    modules = dict(model.named_modules())
    for name in ("model.norm", "model.language_model.norm"):
        if name in modules:
            return name, modules[name]
    raise ValueError("Unsupported model graph: cannot locate audited final language normalization.")


def reset_rope(model):
    for module in model.modules():
        if hasattr(module, "rope_deltas"):
            module.rope_deltas = None


@torch.inference_mode()
def replay(model, inputs, ids, positions, replacement=None):
    """Whole sequence, no KV reuse. The identity site is decoder-only, NOT E2."""
    name, norm = language_norm(model)
    captured = []

    def hook(module, args, output):
        index = positions.to(output.device)
        if replacement is not None:
            output = output.clone()
            output[:, index] = replacement.to(output.device, output.dtype)
        captured.append(output[:, index].detach().clone())
        return output

    handle = norm.register_forward_hook(hook)
    try:
        reset_rope(model)
        kwargs = dict(inputs)
        kwargs.update(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
        output = model(**kwargs)
        if len(captured) != 1:
            raise RuntimeError("Unexpected final norm invocation count.")
        return captured[0], output.logits.detach(), name
    finally:
        handle.remove()


class CoVTModel:
    def __init__(self, model_id, revision, device="cuda", dtype="bfloat16", attention="eager", max_gpu_memory=None):
        import importlib.metadata
        if importlib.metadata.version("transformers") != "4.50.1":
            raise RuntimeError("Native E0 uses transformers==4.50.1. Install the [model] extra in a separate environment.")
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        vocab = self.processor.tokenizer.get_vocab()
        if "<|depth_pad|>" not in vocab:
            raise ValueError("Checkpoint tokenizer does not contain the official depth-pad token.")
        self.depth_id = vocab["<|depth_pad|>"]
        kwargs = dict(revision=revision, torch_dtype=getattr(torch, dtype), attn_implementation=attention,
                      trust_remote_code=False)
        if device == "auto":
            kwargs["device_map"] = "auto"
            if max_gpu_memory:
                kwargs["max_memory"] = {0: max_gpu_memory, "cpu": "32GiB"}
        else:
            kwargs["device_map"] = {"": device}
        self.model, loading = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, output_loading_info=True, **kwargs)
        if loading.get("missing_keys") or loading.get("mismatched_keys"):
            raise ValueError(f"Backbone did not restore completely: {loading}")
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device
        self.provenance = {"model_id": model_id, "revision": revision, "dtype": dtype,
                           "attention": attention, "depth_token_id": self.depth_id,
                           "unexpected_checkpoint_keys": loading.get("unexpected_keys", []),
                           "hidden_state_semantics": "post-final-norm at the depth-pad input positions; no position shift"}

    @torch.inference_mode()
    def run(self, image, question, max_new_tokens=192, atol=0.02, rtol=0.01):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.input_device)
        prefix_length = inputs.input_ids.shape[1]
        generation = dict(max_new_tokens=max_new_tokens, do_sample=False)
        reset_rope(self.model)
        cached = self.model.generate(**inputs, **generation, use_cache=True)
        reset_rope(self.model)
        uncached = self.model.generate(**inputs, **generation, use_cache=False)
        continuation = cached[0, prefix_length:]
        text = self.processor.tokenizer.decode(continuation, skip_special_tokens=False)
        clean_text = self.processor.tokenizer.decode(continuation, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos_ids = [eos] if isinstance(eos, int) else eos or []
        row = {"text": text, "answer": extract_answer(clean_text),
               "generated_ids": continuation.cpu().tolist(), "prefix_length": prefix_length,
               "prompt": prompt, "image_grid_thw": inputs.image_grid_thw.cpu().tolist(),
               "cached_uncached_tokens_equal": torch.equal(cached, uncached),
               "generation_complete": bool(len(continuation) and continuation[-1].item() in eos_ids),
               "uncached_generated_ids": uncached[0, prefix_length:].cpu().tolist()}
        try:
            positions = depth_positions(cached, self.depth_id, prefix_length)
        except ValueError as error:
            return row | {"status": "ineligible_depth_span", "reason": str(error)}, None
        hidden, logits, site = replay(self.model, inputs, cached, positions)
        # Holding the generated sequence fixed is explicitly a local replay test.
        identity, replay_logits, _ = replay(self.model, inputs, cached, positions, replacement=hidden)
        row.update({"depth_positions": positions.cpu().tolist(), "identity_site": site,
                    "identity_scope": "terminal visualization branch only; does not establish shared answer ancestry",
                    "identity_hidden_max_abs": float((identity-hidden).abs().max()),
                    "identity_logits_max_abs": float((replay_logits-logits).abs().max()),
                    "identity_pass": bool(torch.allclose(identity, hidden, atol=atol, rtol=rtol)
                                          and torch.allclose(replay_logits, logits, atol=atol, rtol=rtol)),
                    "atol": atol, "rtol": rtol, "status": "decoded_state_available"})
        return row, hidden[0].float().cpu()
