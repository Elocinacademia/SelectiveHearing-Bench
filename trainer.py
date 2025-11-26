import os
from typing import Dict, List, Optional, Sequence, Union, Any
# from contextlib import contextmanager, nullcontext

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)

import os
import re
import multiprocessing
import time
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, RandomSampler
from packaging import version
from transformers import Trainer
from transformers.cache_utils import Cache
from qwenvl.data.modality_sampler import WeightedRoundRobinBatchSampler
from transformers.integrations.deepspeed import is_deepspeed_available
from transformers.utils import (
    is_torch_compile_available,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    is_accelerate_available
)
from transformers.training_args import OptimizerNames

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        AutocastKwargs,
        DistributedDataParallelKwargs,
        DistributedType,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("1.3.0"):
        from accelerate.utils import TorchTensorParallelPlugin
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

from transformers.trainer import (
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import torch.distributed as dist

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    SaveStrategy,
)

from transformers.trainer_callback import (
    ExportableState,
)

try:
    from cruise.utilities.distributed import DIST_ENV
    from cruise.utilities.hdfs_io import hcopy, hmkdir, hlist_files, hrm
except:
    pass

import re
from liger_kernel.chunked_loss.dpo_loss import LigerFusedLinearDPOLoss
            

HDFS_BASE_PATH = "hdfs://harunava/home/byte_malia_gcp_aiic/user/guangzhisun/models/"

TRAINER_STATE_NAME = "trainer_state.json"


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
        "The final answer is:\n",
        "<answer>",
    ]
    for answer_prefix in answer_prefixes:
        s = s.split(answer_prefix)[-1]
        # s = s.replace(answer_prefix, "")
    if s == "":
        return s
    if s[0].lower() == s[0]:
        s = s[0].upper() + s[1:]
    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return ""
    matches = re.search(r'[ABCDE]', s)
    if matches is None:
        return ""
    return matches[0]

class QwenOmniTrainer(Trainer):

    def __init__(
        self, 
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dpo_loss_fct = LigerFusedLinearDPOLoss()

    def _get_train_sampler(self, train_dataset=None) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None
        return RandomSampler(train_dataset)

    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        # Run validation first
        val_loss = 0
        if self.args.do_validation:
            val_loss = self.validate()
            torch.distributed.reduce(val_loss, 0)
            val_loss = val_loss.item() / dist.get_world_size()
            if DIST_ENV.rank == 0:
                print("Validation Loss: {:.5f}".format(val_loss))

        self.save_model(output_dir, _internal_call=True)

        if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
            # Wait for everyone to get here so we are sure the model has been saved by process 0
            # before we check if the best_checkpoint_dir exists
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()
                
            best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
            best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

            if os.path.exists(best_checkpoint_dir):
                self.state.best_model_checkpoint = best_checkpoint_dir

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            self._save_scaler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Save the Trainer state
        if self.args.should_save:
            # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
            for cb in [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

    def calc_dpo_loss(self, policy_input, policy_target, ref_input, ce_loss=None, beta=0.1):
        lm_head = self.model.lm_head.weight
        dpo_loss, (chosen_logp, reject_logp, chosen_logit, reject_logit, chosen_nll_loss, chosen_rewards, reject_rewards) = self.dpo_loss_fct(lm_head, policy_input, policy_target, ref_input=ref_input, ref_weight=lm_head)
        if ce_loss is not None:
            loss = dpo_loss + beta * ce_loss
        else:
            loss = dpo_loss
        print(f"RANK {dist.get_rank()} chosen: {chosen_rewards.item()}, reject: {reject_rewards.item()}")
        return (loss, dpo_loss, chosen_rewards, reject_rewards)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        train_type = inputs.get("train_type", "sft")
        if train_type == "sft":
            # print(inputs["input_ids"].size())
            outputs = model(**inputs)
        elif train_type == "dpo":
            policy_input, policy_target = model(**inputs)
            with self.accelerator.unwrap_model(self.model).disable_adapter():
                with torch.no_grad():
                    reference_input, reference_target = model(**inputs)
            outputs = self.calc_dpo_loss(policy_input, policy_target, reference_input)
        else:
            raise NotImplementedError

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # Prepare buffers for context parallelism

        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        # Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                if is_torch_xpu_available():
                    torch.xpu.empty_cache()
                elif is_torch_mlu_available():
                    torch.mlu.empty_cache()
                elif is_torch_musa_available():
                    torch.musa.empty_cache()
                elif is_torch_npu_available():
                    torch.npu.empty_cache()
                elif is_torch_mps_available():
                    torch.mps.empty_cache()
                elif is_torch_hpu_available():
                    pass
                else:
                    torch.cuda.empty_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learning rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
                loss = loss / self.current_gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)
            # model.backward(loss, **kwargs)
            # model.optimizer.check_overflow()
            # model.step()

            return loss.detach()

    def validate(self):
        model = self.model
        test_dataloader = DataLoader(self.eval_dataset, batch_size=1, collate_fn=self.data_collator, num_workers=2)
        test_dataloader = self.accelerator.prepare(test_dataloader)
        model.eval()
        total_tokens = 0
        total_hits = 0
        if dist.get_rank() == 0:
            for inputs in tqdm(test_dataloader):
                refanswer = inputs.pop("ref_answer", [None])[0]
                generated_tokens = self.model.generate(**inputs)
                preds = self.tokenizer.decode(generated_tokens[0][inputs["input_ids"].size(1):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                pred = extract_characters_regex(preds)
                refanswer = extract_characters_regex(refanswer)
                pred_correctness = 1.0 if pred == refanswer else 0.0
                total_hits += pred_correctness
                total_tokens += 1
        else:
            for inputs in test_dataloader:
                refanswer = inputs.pop("ref_answer", [None])[0]
                generated_tokens = self.model.generate(**inputs)
                preds = self.tokenizer.decode(generated_tokens[0][inputs["input_ids"].size(1):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                pred = extract_characters_regex(preds)
                refanswer = extract_characters_regex(refanswer)
                pred_correctness = 1.0 if pred == refanswer else 0.0
                total_hits += pred_correctness
                total_tokens += 1
        return torch.tensor(total_hits/total_tokens*100).to(inputs["input_ids"].device)