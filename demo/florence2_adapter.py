# demo/florence2_adapter.py
import numpy as np
import torch
from PIL import Image
import cv2
import contextlib  # for nullcontext

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
except Exception as e:
    raise ImportError("pip install 'transformers>=4.41.0' 'accelerate' 'safetensors'") from e


def _classes_from_prompt(prompt: str):
    # "red car ." → ["red car"]
    return [c.strip().lower() for c in prompt.split('.') if c.strip()]


def _label_matches(label: str, targets):
    lab = label.lower()
    for t in targets:
        if t in lab or lab in t:
            return True
    return False


class Florence2Detector:
    """
    Florence-2 OD → [N,5] np.float32 (x1,y1,x2,y2,score).
    Hardened for processor/model diffs re: image placeholder tokens & dtypes.
    """
    def __init__(self, model_id="microsoft/Florence-2-large", device="cuda", fp16=False):
        self.device = device
        self.fp16 = bool(fp16)

        # Load processor & model
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        torch_dtype = torch.float16 if (self.fp16 and torch.cuda.is_available()) else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self.device).eval()

        # Tokenizer handle
        tok = getattr(self.processor, "tokenizer", None) or getattr(self.processor, "text_tokenizer", None)
        self.tokenizer = tok

        # Make sure the model's embedding table can index the tokenizer ids
        if self.tokenizer is not None:
            try:
                self.model.resize_token_embeddings(len(self.tokenizer))
            except Exception:
                # Some snapshots don’t like resize when sizes already match; ignore benign errors.
                pass

        # Generation config sanity (bos/eos/pad/decoder_start)
        gc = self.model.generation_config
        if self.tokenizer is not None:
            if getattr(self.model.config, "decoder_start_token_id", None) is None:
                self.model.config.decoder_start_token_id = getattr(self.tokenizer, "bos_token_id", None)
            if getattr(gc, "decoder_start_token_id", None) is None:
                gc.decoder_start_token_id = self.model.config.decoder_start_token_id
            if getattr(gc, "bos_token_id", None) is None:
                gc.bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
            if getattr(gc, "eos_token_id", None) is None:
                gc.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if getattr(gc, "pad_token_id", None) is None:
                gc.pad_token_id = getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None)
            self.model.generation_config = gc

        # Discover/set an image placeholder token id that BOTH tokenizer & model agree on
        self.candidate_image_tokens = ["<image>", "<|image|>", "<image_1>"]

        def _discover_image_tid():
            # Try model/proc hints first
            tid = (
                getattr(getattr(self.model, "generation_config", None), "image_token_id", None)
                or getattr(getattr(self.model, "config", None), "image_token_id", None)
                or getattr(self.processor, "image_token_id", None)
            )
            if tid is not None:
                return int(tid)

            if self.tokenizer is None:
                return None

            # Try known spellings
            for s in self.candidate_image_tokens:
                try:
                    t = self.tokenizer.convert_tokens_to_ids(s)
                    if t is not None and t != getattr(self.tokenizer, "unk_token_id", -1):
                        return int(t)
                except Exception:
                    continue
            return None

        self.image_token_id = _discover_image_tid()

        # If we found one, make sure model knows it too
        if self.image_token_id is not None:
            self.model.config.image_token_id = int(self.image_token_id)
            self.model.generation_config.image_token_id = int(self.image_token_id)

        # Final guard: ensure any tid we’ll use is < embedding size
        self._embed_size = self.model.get_input_embeddings().num_embeddings
        if self.tokenizer is not None and self._embed_size < len(self.tokenizer):
            # Resize again defensively
            self.model.resize_token_embeddings(len(self.tokenizer))
            self._embed_size = self.model.get_input_embeddings().num_embeddings

    def _move_to_device_and_cast(self, inputs):
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                if k == "pixel_values":
                    # Match model dtype for vision tower (prevents float/half mismatch)
                    inputs[k] = v.to(self.device, dtype=self.model.dtype)
                else:
                    inputs[k] = v.to(self.device)
        return inputs

    def _append_placeholder(self, inputs, token_id: int):
        if token_id is None or "input_ids" not in inputs:
            return inputs

        if token_id >= self._embed_size:
            # We must be able to index embeddings with this id
            self.model.resize_token_embeddings(max(self._embed_size, token_id + 1))
            self._embed_size = self.model.get_input_embeddings().num_embeddings

        iid = inputs["input_ids"].to(torch.long)
        # Already present?
        if (iid == token_id).any():
            inputs["input_ids"] = iid
            return inputs

        append = torch.tensor([[token_id]], dtype=iid.dtype, device=iid.device)
        inputs["input_ids"] = torch.cat([iid, append], dim=1)
        if "attention_mask" in inputs:
            am = inputs["attention_mask"]
            inputs["attention_mask"] = torch.cat([am, torch.ones_like(append)], dim=1)
        return inputs

    @torch.inference_mode()
    def predict(self, frame_bgr, text_prompt: str, box_threshold: float = 0.4, max_new_tokens: int = 512):
        """
        Returns numpy array dets: [N,5] with pixel xyxy + score.
        Filters by classes parsed from text_prompt (substring match).
        """
        H, W = frame_bgr.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        targets = _classes_from_prompt(text_prompt)

        # Florence-2 OD task token MUST be alone
        task_prompt = "<OD>"

        # Build processor inputs (no add_special_tokens, etc.)
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
        # Types/devices
        inputs = self._move_to_device_and_cast(inputs)
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].to(torch.long)

        gc = self.model.generation_config
        use_autocast = (self.model.dtype == torch.float16) and str(self.device).startswith("cuda")
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_autocast else contextlib.nullcontext()

        # 1) Try the clean path
        try:
            with amp_ctx:
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    decoder_start_token_id=getattr(gc, "decoder_start_token_id", None),
                    bos_token_id=getattr(gc, "bos_token_id", None),
                    eos_token_id=getattr(gc, "eos_token_id", None),
                    pad_token_id=getattr(gc, "pad_token_id", None),
                )
        except ValueError as e:
            # 2) Fallback: append placeholder token, ensure model/config know the tid, and retry
            msg = str(e).lower()
            needs_placeholder = (
                "image features and image tokens do not match" in msg
                or "placeholder" in msg
                or "tokens:" in msg
            )

            if not needs_placeholder:
                raise

            # Build candidate tids list
            candidates = []
            if self.image_token_id is not None:
                candidates.append(("configured", int(self.image_token_id)))
            if self.tokenizer is not None:
                for s in self.candidate_image_tokens:
                    try:
                        tid = self.tokenizer.convert_tokens_to_ids(s)
                        if tid is not None and tid != getattr(self.tokenizer, "unk_token_id", -1):
                            candidates.append((s, int(tid)))
                    except Exception:
                        pass

            last_err = e
            tried = set()
            for name, tid in candidates:
                if tid in tried:
                    continue
                tried.add(tid)

                # Ensure model knows this id as the image token
                self.model.config.image_token_id = int(tid)
                self.model.generation_config.image_token_id = int(tid)

                # Clone & append
                inputs2 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
                inputs2 = self._append_placeholder(inputs2, tid)
                try:
                    with amp_ctx:
                        gen_ids = self.model.generate(
                            **inputs2,
                            max_new_tokens=max_new_tokens,
                            decoder_start_token_id=getattr(gc, "decoder_start_token_id", None),
                            bos_token_id=getattr(gc, "bos_token_id", None),
                            eos_token_id=getattr(gc, "eos_token_id", None),
                            pad_token_id=getattr(gc, "pad_token_id", None),
                        )
                    inputs = inputs2  # success
                    break
                except Exception as ee:
                    last_err = ee
                    continue
            else:
                raise last_err

        # Post-process: HF changed signatures between snapshots
        try:
            parsed = self.processor.post_process_object_detection(
                generated_ids=gen_ids, inputs=inputs, threshold=box_threshold
            )[0]
        except TypeError:
            parsed = self.processor.post_process_object_detection(
                generated_ids=gen_ids, target_sizes=[(H, W)], threshold=box_threshold
            )[0]

        boxes = parsed.get("boxes", [])
        scores = parsed.get("scores", [])
        labels = parsed.get("labels", [])

        dets = []
        for (x1, y1, x2, y2), s, lab in zip(boxes, scores, labels):
            if targets and not _label_matches(lab, targets):
                continue
            x1 = max(0.0, min(float(x1), W - 1))
            y1 = max(0.0, min(float(y1), H - 1))
            x2 = max(0.0, min(float(x2), W - 1))
            y2 = max(0.0, min(float(y2), H - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append([x1, y1, x2, y2, float(s)])

        if not dets:
            return np.zeros((0, 5), dtype=np.float32)
        return np.asarray(dets, dtype=np.float32)
