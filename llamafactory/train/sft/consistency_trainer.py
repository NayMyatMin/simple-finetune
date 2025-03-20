# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer

import importlib.metadata
import torch.nn.functional as F

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler

from packaging import version
# from . import __version__

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments

import sys
sys.path.append('/root/miniconda3/lib/python3.8/site-packages')
from transformers.utils import is_peft_available

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)

if is_peft_available():
    from peft import PeftModel

    def _is_peft_model(model):
        if is_peft_available():
            classes_to_check = (PeftModel,) if is_peft_available() else ()
            # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
            if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
                from peft import PeftMixedModel

                classes_to_check = (*classes_to_check, PeftMixedModel)
            return isinstance(model, classes_to_check)
        return False

logger = get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.processor = processor

        # if finetuning_args.pissa_convert:
        #     self.save_model(os.path.join(self.args.output_dir, "pissa_init"))

        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, "torch.Tensor"]] = None) -> None:
        super()._save(output_dir, state_dict)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        # if self.finetuning_args.pissa_convert:
        #     convert_pissa_adapter(output_dir, state_dict, self.accelerator, self.model, self.args)

        if self.processor is not None:
            getattr(self.processor, "image_processor").save_pretrained(output_dir)

    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate(
                    (preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1
                )  # move pad token to last

        decoded_inputs = self.tokenizer.batch_decode(
            dataset["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))


    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     Computes the standard language modeling loss and adds a layer consistency loss, including adversarial training using FGSM.
    #     """
    #     inputs = inputs.copy()

    #     unwrapped_model = self.accelerator.unwrap_model(model)
    #     inputs_embeds = unwrapped_model.get_input_embeddings()(inputs["input_ids"]).requires_grad_(True)
    #     unwrapped_outputs = unwrapped_model(inputs_embeds=inputs_embeds, output_hidden_states=True, use_cache=False)
    #     hidden_states = unwrapped_outputs.hidden_states

    #     h_states = torch.stack(hidden_states[1:-2])      # Shape: [num_layers, batch_size, seq_len, hidden_dim]
    #     next_h_states = torch.stack(hidden_states[2:-1]) # Shape: [num_layers, batch_size, seq_len, hidden_dim]

    #     cos_sims_vec = F.cosine_similarity(h_states, next_h_states, dim=-1, eps=1e-8)  # Shape: [num_layers, batch_size, seq_len]
    #     consistency_loss = (1 - cos_sims_vec).mean()

    #     # Zero gradients
    #     model.zero_grad()
    #     if inputs_embeds.grad is not None:
    #         inputs_embeds.grad.zero_()

    #     # Backward pass for consistency_loss
    #     consistency_loss.backward(retain_graph=True)

    #     # Extract gradients w.r.t. inputs_embeds
    #     gradients = inputs_embeds.grad.detach()

    #     # Zero gradients in model parameters to prevent updates from consistency_loss
    #     model.zero_grad()
    #     if inputs_embeds.grad is not None:
    #         inputs_embeds.grad.zero_()

    #     # Generate adversarial perturbations
    #     epsilon = 0.1
    #     perturbation = epsilon * gradients.sign()
    #     perturbed_embeds = inputs_embeds + perturbation

    #     # Forward pass with perturbed inputs for consistency regularization
    #     perturbed_outputs = model(inputs_embeds=perturbed_embeds, output_hidden_states=True, use_cache=False)
    #     perturbed_hidden_states = perturbed_outputs.hidden_states

    #     # Compute perturbed consistency loss using vectorized method
    #     perturbed_h_states = torch.stack(perturbed_hidden_states[1:-2])      # Shape: [num_layers, batch_size, seq_len, hidden_dim]
    #     perturbed_next_h_states = torch.stack(perturbed_hidden_states[2:-1]) # Shape: [num_layers, batch_size, seq_len, hidden_dim]

    #     perturbed_cos_sims_vec = F.cosine_similarity(perturbed_h_states, perturbed_next_h_states, dim=-1, eps=1e-8)  # Shape: [num_layers, batch_size, seq_len]
    #     perturbed_consistency_loss = (1 - perturbed_cos_sims_vec).mean()
    #     print("Perturbed Consistency Loss: ", perturbed_consistency_loss.item())

    #     outputs = model(**inputs)
    #     standard_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    #     # Combined Loss
    #     alpha = 5.5 # Hyperparameter for the consistency loss
    #     total_loss =  standard_loss + alpha * perturbed_consistency_loss

    #     return (total_loss, outputs) if return_outputs else total_loss


    def _relative_top_filter(
        self,
        final_logits: torch.FloatTensor,
        base_logits: torch.FloatTensor,
        relative_top: float = 0.1,
        filter_value: float = -float("inf"),
        base_filter_value: float = -1e-3,
        min_tokens_to_keep: int = 1):
        """
        Applies an adaptive threshold to keep only tokens whose log-prob
        is above 'relative_top' * max_log_prob. Tokens below that become 'filter_value' 
        or 'base_filter_value' for the baseline.
        
        final_logits, base_logits: unnormalized logits for final and baseline distributions
        relative_top: fraction of the max probability to keep
        filter_value: typically -inf for the final distribution
        base_filter_value: a negative constant for the baseline distribution (not fully -inf)
        min_tokens_to_keep: ensure at least this many tokens remain unfiltered
        
        Returns:
            (filtered_final_logits, filtered_base_logits)
        """
        # 1) Normalize both
        final_log_probs = final_logits.log_softmax(dim=-1)
        base_log_probs = base_logits.log_softmax(dim=-1)
        
        # 2) Sort final_log_probs descending to find min threshold for tokens we keep
        sorted_logits, sorted_indices = torch.sort(final_log_probs, descending=True, dim=-1)
        # min_thresh is the log-prob of the 'min_tokens_to_keep'-th token
        min_thresh = sorted_logits[..., min_tokens_to_keep - 1]

        # 3) Compute a threshold based on the max log-prob + log(relative_top)
        probs_max, _ = torch.max(final_log_probs, dim=-1, keepdim=True)
        # log(relative_top) is negative. e.g. log(0.1) ~ -2.3026
        probs_thresh = probs_max + np.log(relative_top)
        
        # 4) The actual threshold is whichever is smaller: 'min_thresh' or 'probs_thresh'
        #    so we never keep fewer than 'min_tokens_to_keep' tokens
        # shape: [batch, seq] => expand last dim for broadasting across vocab
        probs_thresh = torch.min(min_thresh, probs_thresh.squeeze(-1))
        probs_thresh = probs_thresh.unsqueeze(-1)
        
        # 5) Apply threshold
        # If final_log_probs < probs_thresh => set to filter_value or base_filter_value
        mask = (final_log_probs < probs_thresh)  # shape [batch, seq, vocab]
        final_log_probs_filtered = final_log_probs.masked_fill(mask, filter_value)
        base_log_probs_filtered = base_log_probs.masked_fill(mask, base_filter_value)
        
        # 6) Return them in "logits" space again (unexponentiated)
        # Optionally, you can keep them in log-prob space if your next step does log-ratio.
        # But let's keep a consistent "logits" style by re-exponentiating and taking log again
        # or we can just treat these as "logits" for the next step.
        return final_log_probs_filtered, base_log_probs_filtered

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     Implements combined loss:
    #     L_total = LM_loss
    #             + alpha_eigen * L_eigen_weighted
    #             + alpha_distill * L_distill
    #             + alpha_consistency * L_adversarial_consistency

    #     where:
    #     - L_eigen_weighted: diagonal variance of attention across layers
    #     - L_distill: weighted layer-wise distillation (contrastive teacher)
    #     - L_adversarial_consistency: layer consistency loss computed on adversarial inputs
    #     """

    #     # Make a copy of 'inputs' so we don't mutate outside data structures
    #     inputs = inputs.copy()
    #     unwrapped_model = self.accelerator.unwrap_model(model)

    #     # ===================================================
    #     # 1. First Forward Pass with Trainable Input Embeddings
    #     #    for: base_loss, distillation, eigen variance, and  "clean" layer-consistency
    #     # ===================================================
    #     inputs_embeds = unwrapped_model.get_input_embeddings()(inputs["input_ids"]).requires_grad_(True)
    #     outputs = unwrapped_model(
    #         **{k: v for k, v in inputs.items() if k != "input_ids"},  # pass other items (attention_mask, etc.)
    #         inputs_embeds=inputs_embeds,
    #         output_hidden_states=True,
    #         output_attentions=True,
    #         use_cache=False
    #     )
    #     base_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    #     # Hidden states and attentions
    #     hidden_states = outputs.hidden_states  # list of length (num_layers+1)
    #     attentions = outputs.attentions
    #     num_layers = len(hidden_states) - 1
    #     middle_idx = num_layers // 2

    #     # ===================================================
    #     # 2. Compute Contrastive Teacher Distillation
    #     # ===================================================
    #     if hasattr(unwrapped_model, "base_model"):
    #         lm_head = unwrapped_model.base_model.model.lm_head
    #     else:
    #         lm_head = unwrapped_model.lm_head

    #     with torch.no_grad():
    #         middle_logits = lm_head(hidden_states[middle_idx])
    #         final_logits = lm_head(hidden_states[-1])

    #         # Relative top filtering (if needed)
    #         filtered_final, filtered_base = self._relative_top_filter(
    #             final_logits, 
    #             middle_logits,
    #             relative_top=0.1,
    #             filter_value=-float("inf"),
    #             base_filter_value=-1e-3,
    #             min_tokens_to_keep=1
    #         )

    #         log_ratio = filtered_final - filtered_base
    #         teacher_probs = F.softmax(log_ratio / 1.0, dim=-1)  # T_contrast = 1.0

    #     distill_loss = 0.0
    #     T_distill = 1.0
    #     for l_idx in range(num_layers):
    #         w_l = l_idx / (num_layers - 1) if num_layers > 1 else 1.0
    #         student_logits = lm_head(hidden_states[l_idx])
    #         student_log_probs = F.log_softmax(student_logits / T_distill, dim=-1)
    #         layer_kl = F.kl_div(student_log_probs, teacher_probs, reduction='none')
    #         layer_kl = layer_kl.mean(dim=-1).mean()  # average over vocab & batch
    #         distill_loss += w_l * layer_kl

    #     # ===================================================
    #     # 3. Compute "Eigenvalue Variance" (Diagonal Variance) Loss
    #     # ===================================================
    #     sum_weighted_var = 0.0
    #     for l_idx in range(num_layers):

    #         # If you want weighting per layer:
    #         # w_l = l_idx / (num_layers - 1) if num_layers > 1 else 1.0

    #         attn_map = attentions[l_idx]  # shape [batch_size, num_heads, seq_len, seq_len]
    #         bsz, num_heads, seq_len, _ = attn_map.shape
    #         layer_variance_sum = 0.0

    #         for head_i in range(num_heads):
    #             A_i = attn_map[:, head_i, :, :]
    #             diag_vals = A_i.diagonal(dim1=-2, dim2=-1)  # shape [bsz, seq_len]
    #             diag_vals_flat = diag_vals.reshape(-1)

    #             mean_val = diag_vals_flat.mean()
    #             var_val = ((diag_vals_flat - mean_val) ** 2).mean()
    #             layer_variance_sum += var_val

    #         layer_variance_mean = layer_variance_sum / float(num_heads)
    #         sum_weighted_var += layer_variance_mean  # or multiply by w_l if you prefer

    #     # ===================================================
    #     # 4. "Clean" Layer Consistency Loss (for gradient)
    #     # ===================================================
    #     # Compare adjacent hidden layers via cosine similarity
    #     h_stack = torch.stack(hidden_states[1:-2])      # shape [num_layers, bsz, seq_len, hidden_dim]
    #     next_h_stack = torch.stack(hidden_states[2:-1]) # same shape
    #     cos_sims = F.cosine_similarity(h_stack, next_h_stack, dim=-1, eps=1e-8)
    #     consistency_loss = (1.0 - cos_sims).mean()

    #     # ===================================================
    #     # 5. Partial Backward on Consistency Loss to get FGSM
    #     # ===================================================
    #     # Zero out any existing grads
    #     model.zero_grad()
    #     if inputs_embeds.grad is not None:
    #         inputs_embeds.grad.zero_()

    #     consistency_loss.backward(retain_graph=True)
    #     embed_grads = inputs_embeds.grad.detach()

    #     # Zero out model param grads so they won't be updated by consistency_loss now
    #     model.zero_grad()
    #     inputs_embeds.grad.zero_()
    #     epsilon = 0.1
    #     adv_embeds = inputs_embeds + epsilon * embed_grads.sign()

    #     # ===================================================
    #     # 6. Second Forward Pass: Adversarial Consistency
    #     # ===================================================
    #     adv_outputs = unwrapped_model(
    #         **{k: v for k, v in inputs.items() if k != "input_ids"},
    #         inputs_embeds=adv_embeds,
    #         output_hidden_states=True,
    #         use_cache=False
    #     )
    #     adv_hidden_states = adv_outputs.hidden_states

    #     adv_h_stack = torch.stack(adv_hidden_states[1:-2])
    #     adv_next_h_stack = torch.stack(adv_hidden_states[2:-1])
    #     adv_cos_sims = F.cosine_similarity(adv_h_stack, adv_next_h_stack, dim=-1, eps=1e-8)
    #     adv_consistency_loss = (1.0 - adv_cos_sims).mean()

    #     # ===================================================
    #     # 7. Combine All Losses into One
    #     # ===================================================
    #     alpha_eigen = 1.0
    #     alpha_distill = 1.0
    #     alpha_consistency = 1.0

    #     total_loss = (
    #         base_loss
    #         + alpha_eigen * sum_weighted_var
    #         + alpha_distill * distill_loss
    #         + alpha_consistency * adv_consistency_loss
    #     )

    #     # Debug prints
    #     print(f"[Combined] #layers = {num_layers}")
    #     print(f"  base_loss               = {base_loss.item():.5f}")
    #     print(f"  eigen_var (diag)        = {sum_weighted_var.item():.5f}")
    #     print(f"  distill_loss            = {distill_loss.item():.5f}")
    #     print(f"  adv_consistency_loss    = {adv_consistency_loss.item():.5f}")

    #     return (total_loss, outputs) if return_outputs else total_loss

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     # 1. Copy the inputs
    #     inputs = inputs.copy()

    #     # 2. "Normal" forward to get the usual final-layer "base_loss"
    #     unwrapped_model = self.accelerator.unwrap_model(model)
    #     outputs_main = unwrapped_model(**inputs)
    #     base_loss = outputs_main["loss"] if isinstance(outputs_main, dict) else outputs_main[0]
    #     print(f"  base_loss = {base_loss.item():.5f}")

    #     # 3. Manually extract hidden states for shallow-layer partial alignment
    #     #    (We'll skip the last layer to avoid double-counting)
    #     inputs_embeds = unwrapped_model.get_input_embeddings()(inputs["input_ids"]).requires_grad_(True)
    #     outputs_hs = unwrapped_model(
    #         inputs_embeds=inputs_embeds,
    #         output_hidden_states=True,
    #         use_cache=False
    #     )
    #     hidden_states = outputs_hs.hidden_states
    #     # hidden_states[1..L] are the actual L blocks. We'll do partial alignment only up to L-1

    #     # 4. Prepare your shallow-layer weighting scheme: lam_ell for ell=1..(L-1)
    #     L = len(hidden_states) - 1  # total number of Transformer layers
    #     lam = []
    #     for ell in range(1, L):
    #         lam_ell = 2.0 * ell / (L * (L + 1))  # or whatever weighting you want
    #         lam.append(lam_ell)

    #     # 5. Compute partial alignment losses for layers 1..(L-1)
    #     vocab_size = unwrapped_model.lm_head.weight.shape[0]
    #     aux_losses = []
    #     for idx, layer_hs in enumerate(hidden_states[1:-1], start=1):
    #         # hidden_states[1] => layer 1, hidden_states[L-1] => layer L-1
    #         layer_logits = unwrapped_model.lm_head(layer_hs)

    #         shift_logits = layer_logits[:, :-1, :].contiguous()
    #         shift_labels = inputs["input_ids"][:, 1:].contiguous()

    #         layer_loss = F.cross_entropy(
    #             shift_logits.view(-1, vocab_size),
    #             shift_labels.view(-1)
    #         )
    #         weighted_loss = lam[idx - 1] * layer_loss
    #         aux_losses.append(weighted_loss)

    #     partial_loss = torch.stack(aux_losses).sum() if aux_losses else 0.0

    #     # 6. Combine "base loss" + partial alignment from shallow layers
    #     total_loss = base_loss + partial_loss
    #     print(f"  partial_loss = {partial_loss.item() if aux_losses else 0.0:.5f}")
    #     print(f"  total_loss   = {total_loss.item():.5f}")

    #     return (total_loss, outputs_main) if return_outputs else total_loss

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     Implements a log-based attention penalty in addition to the base LM loss.

    #     L_total = LM_loss
    #             + alpha_a * mean_log_det_score

    #     where:
    #     - mean_log_det_score is computed by averaging the log-diagonal across tokens and
    #     then averaging across heads to get a single scalar per layer.
    #     """

    #     # 1. Standard forward pass
    #     inputs = inputs.copy()
    #     unwrapped_model = self.accelerator.unwrap_model(model)
    #     outputs_main = unwrapped_model(**inputs)
    #     base_loss = outputs_main["loss"] if isinstance(outputs_main, dict) else outputs_main[0]
    #     print(f"  base_loss = {base_loss.item():.5f}")

    #     # 2. Extract attention matrices
    #     outputs_hs = unwrapped_model(
    #         inputs_embeds=unwrapped_model.get_input_embeddings()(inputs["input_ids"]).requires_grad_(True),
    #         output_attentions=True,  # Request attention maps
    #         use_cache=False
    #     )

    #     attentions = outputs_hs.attentions  # Extract attention scores
    #     num_layers = len(attentions)

    #     # Scaling hyperparameter for the attention penalty
    #     lambda_a = 1.0  

    #     # We'll accumulate a log-determinant score across layers
    #     sum_log_det_score = 0.0

    #     # 3. Compute Attention Score Loss using log-determinant approach (single scalar per layer)
    #     for attn_matrix in attentions:
    #         # attn_matrix shape: (batch_size, num_heads, seq_len, seq_len)
    #         bsz, num_heads, seq_len, _ = attn_matrix.shape

    #         # Extract diagonal elements (self-attention probabilities on the diagonal)
    #         diagonal_values = torch.diagonal(attn_matrix, dim1=-2, dim2=-1)  # shape: (batch_size, num_heads, seq_len)
    #         diagonal_values = torch.clamp(diagonal_values, min=1e-6)         # prevent log(0)

    #         # Compute the mean log of these diagonal values across tokens => shape: (batch_size, num_heads)
    #         log_per_head = torch.mean(torch.log(diagonal_values), dim=-1)

    #         # Average across heads => shape: (batch_size,)
    #         log_mean_across_heads = log_per_head.mean(dim=1)

    #         # Finally, average over the batch to get one scalar for this layer
    #         layer_score = log_mean_across_heads.mean()

    #         print(f"  layer_score (mean log det) = {layer_score.item():.5f}")
    #         sum_log_det_score += layer_score

    #     # 4. Average over layers
    #     # This yields a single "mean_log_det_score" across all layers
    #     mean_log_det_score = sum_log_det_score / num_layers

    #     # 5. Combine base loss with attention penalty
    #     # Add the negative if you want to *reward* more negative logs.
    #     # But if you want to encourage more negative logs by direct addition, just do:
    #     attention_loss = lambda_a * mean_log_det_score

    #     total_loss = base_loss + attention_loss
    #     print(f"  attention_loss = {attention_loss.item():.5f}")
    #     print(f"  total_loss     = {total_loss.item():.5f}")

    #     return (total_loss, outputs_main) if return_outputs else total_loss


    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     Computes a combined loss of:
    #         total_loss = base_loss + attention_loss

    #     Where 'attention_loss' is a single-scalar log-based measure on the final layer only,
    #     summing or averaging the log(diagonal) across tokens and heads.
    #     This approach does NOT compute variance across heads, but rather a single scalar
    #     for the entire batch.
    #     """

    #     # 1. Copy inputs, enabling attention outputs
    #     inputs = inputs.copy()
    #     inputs["output_attentions"] = True
    #     inputs["use_cache"] = False
        
    #     # Single forward pass
    #     unwrapped_model = self.accelerator.unwrap_model(model)
    #     outputs = unwrapped_model(**inputs)
        
    #     # 2. Base loss
    #     base_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    #     print(f"  base_loss = {base_loss.item():.5f}")
        
    #     # 3. Extract the final layer's attention
    #     #    'attentions' is typically a list of length = #layers
    #     attentions = outputs.attentions if hasattr(outputs, "attentions") else outputs["attentions"]
    #     final_attn = attentions[-1]  # shape: (batch_size, num_heads, seq_len, seq_len)
        
    #     # 4. Diagonal extraction & log
    #     diagonal_values = torch.diagonal(final_attn, dim1=-2, dim2=-1)  # shape: (batch_size, num_heads, seq_len)
    #     diagonal_values = torch.clamp(diagonal_values, min=1e-6)        # avoid log(0)
        
    #     # log_det per head: average log across tokens => shape (batch_size, num_heads)
    #     log_per_head = torch.mean(torch.log(diagonal_values), dim=-1)
        
    #     # average across heads => shape (batch_size,)
    #     log_mean_across_heads = log_per_head.mean(dim=1)
        
    #     # final scalar: average over batch
    #     single_layer_score = log_mean_across_heads.mean()
        
    #     # 5. Define your hyperparameter
    #     lambda_a = 1.0
        
    #     # If single_layer_score is negative, adding it lowers total_loss => rewarding more negative logs
    #     attention_loss = lambda_a * single_layer_score
    #     print(f"  attention_loss = {attention_loss.item():.5f}")
        
    #     # 6. Final combined loss
    #     total_loss = base_loss + attention_loss
    #     print(f"  total_loss   = {total_loss.item():.5f}")
        
    #     return (total_loss, outputs) if return_outputs else total_loss



    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes a combined training objective for fine-tuning a language model with multiple loss terms:

        L_total = LM_loss
                + alpha_eigen * log_attention_penalty
                + alpha_distill * distill_loss
                + alpha_consistency * adversarial_consistency_loss

        where:

        - LM_loss:
            The standard language modeling loss (e.g., cross-entropy on next-token prediction).

        - log_attention_penalty (mean_log_det_penalty):
            A log-based attention penalty derived from the model's self-attention diagonals. 
            Specifically, it measures the average log of the diagonal entries for each layer's 
            attention, encouraging the model to maintain more negative logs when beneficial 
            (i.e., normal or grounded self-attention).

        - distill_loss (L_distill):
            A layer-wise contrastive distillation objective that aligns intermediate layers 
            to a teacher distribution (e.g., final-layer logits).

        - adversarial_consistency_loss (L_adversarial_consistency):
            A consistency regularization term computed on adversarial embeddings generated 
            via FGSM. It measures how robust consecutive hidden layers remain under such 
            perturbations, encouraging stable representations.

        alpha_eigen, alpha_distill, and alpha_consistency are scaling factors that 
        weight each auxiliary term. The final training objective is returned as 'total_loss'.
        """

        # Make a copy of 'inputs' so we don't mutate outside data structures
        inputs = inputs.copy()
        unwrapped_model = self.accelerator.unwrap_model(model)

        # ===================================================
        # 1. First Forward Pass with Trainable Input Embeddings
        #    for: base_loss, distillation, eigen variance, and  "clean" layer-consistency
        # ===================================================
        inputs_embeds = unwrapped_model.get_input_embeddings()(inputs["input_ids"]).requires_grad_(True)
        outputs = unwrapped_model(
            **{k: v for k, v in inputs.items() if k != "input_ids"},  # pass other items (attention_mask, etc.)
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False
        )
        base_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Hidden states and attentions
        hidden_states = outputs.hidden_states  # list of length (num_layers+1)
        attentions = outputs.attentions
        num_layers = len(hidden_states) - 1
        middle_idx = num_layers // 2

        # ===================================================
        # 2. Compute Contrastive Teacher Distillation
        # ===================================================
        if hasattr(unwrapped_model, "base_model"):
            lm_head = unwrapped_model.base_model.model.lm_head
        else:
            lm_head = unwrapped_model.lm_head

        with torch.no_grad():
            middle_logits = lm_head(hidden_states[middle_idx])
            final_logits = lm_head(hidden_states[-1])

            # Relative top filtering (if needed)
            filtered_final, filtered_base = self._relative_top_filter(
                final_logits, 
                middle_logits,
                relative_top=0.1,
                filter_value=-float("inf"),
                base_filter_value=-1e-3,
                min_tokens_to_keep=1
            )

            log_ratio = filtered_final - filtered_base
            teacher_probs = F.softmax(log_ratio / 1.0, dim=-1)  

        distill_loss = 0.0
        T_distill = 1.0
        for l_idx in range(num_layers):
            w_l = l_idx / (num_layers - 1) if num_layers > 1 else 1.0
            student_logits = lm_head(hidden_states[l_idx])
            student_log_probs = F.log_softmax(student_logits / T_distill, dim=-1)
            layer_kl = F.kl_div(student_log_probs, teacher_probs, reduction='none')
            layer_kl = layer_kl.mean(dim=-1).mean()  # average over vocab & batch
            distill_loss += w_l * layer_kl

        # ===================================================
        # 3. Compute "Log-based Attention Penalty" (single-scalar approach)
        # ===================================================
        sum_log_det_penalty = 0.0
        for l_idx in range(num_layers):
            attn_map = attentions[l_idx]  # shape [batch_size, num_heads, seq_len, seq_len]

            # Extract diagonal
            diag_vals = attn_map.diagonal(dim1=-2, dim2=-1)  # shape: (batch_size, num_heads, seq_len)
            diag_vals = torch.clamp(diag_vals, min=1e-6)

            # log_det per head: average log across tokens => shape: (batch_size, num_heads)
            log_per_head = torch.mean(torch.log(diag_vals), dim=-1)

            # Then we average across heads => shape: (batch_size,)
            # This yields 1 scalar per sample per layer
            log_mean_across_heads = log_per_head.mean(dim=1)

            # Then we average over the batch => single scalar for the entire layer
            layer_score = log_mean_across_heads.mean()

            # We accumulate across layers
            sum_log_det_penalty += layer_score

        # Finally, average across layers
        mean_log_det_penalty = sum_log_det_penalty / num_layers

        # ===================================================
        # 4. "Clean" Layer Consistency Loss (for gradient)
        # ===================================================
        # Compare adjacent hidden layers via cosine similarity
        h_stack = torch.stack(hidden_states[1:-2])      # shape [num_layers, bsz, seq_len, hidden_dim]
        next_h_stack = torch.stack(hidden_states[2:-1]) # same shape
        cos_sims = F.cosine_similarity(h_stack, next_h_stack, dim=-1, eps=1e-8)
        consistency_loss = (1.0 - cos_sims).mean()

        # ===================================================
        # 5. Partial Backward on Consistency Loss to get FGSM
        # ===================================================
        # Zero out any existing grads
        model.zero_grad()
        if inputs_embeds.grad is not None:
            inputs_embeds.grad.zero_()

        consistency_loss.backward(retain_graph=True)
        embed_grads = inputs_embeds.grad.detach()

        # Zero out model param grads so they won't be updated by consistency_loss now
        model.zero_grad()
        inputs_embeds.grad.zero_()
        epsilon = 0.1
        adv_embeds = inputs_embeds + epsilon * embed_grads.sign()

        # ===================================================
        # 6. Second Forward Pass: Adversarial Consistency
        # ===================================================
        adv_outputs = unwrapped_model(
            **{k: v for k, v in inputs.items() if k != "input_ids"},
            inputs_embeds=adv_embeds,
            output_hidden_states=True,
            use_cache=False
        )
        adv_hidden_states = adv_outputs.hidden_states

        adv_h_stack = torch.stack(adv_hidden_states[1:-2])
        adv_next_h_stack = torch.stack(adv_hidden_states[2:-1])
        adv_cos_sims = F.cosine_similarity(adv_h_stack, adv_next_h_stack, dim=-1, eps=1e-8)
        adv_consistency_loss = (1.0 - adv_cos_sims).mean()

        # ===================================================
        # 7. Combine All Losses into One
        # ===================================================
        alpha_eigen = 1.0
        alpha_distill = 1.0
        alpha_consistency = 1.0

        total_loss = (
            base_loss
            + alpha_eigen * mean_log_det_penalty
            + alpha_distill * distill_loss
            + alpha_consistency * adv_consistency_loss
        )

        # Debug prints
        print(f"[Combined] #layers = {num_layers}")
        print(f"  base_loss               = {base_loss.item():.5f}")
        print(f"  mean_log_det        = {mean_log_det_penalty.item():.5f}")
        print(f"  distill_loss            = {distill_loss.item():.5f}")
        print(f"  adv_consistency_loss    = {adv_consistency_loss.item():.5f}")
        print(f"  total_loss              = {total_loss.item():.5f}")
        print("---------------------------------")

        return (total_loss, outputs) if return_outputs else total_loss
    
