# Standard library
import os
import math
import warnings
from enum import Enum, auto
from tqdm import tqdm
from functools import reduce
import contextlib
from typing import Union, List, Tuple, Optional, Dict, Callable
from collections import OrderedDict
from abc import ABC, abstractmethod

# Pytorch
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.utils.data import DataLoader
import torch.nn.utils.parametrize as parametrize

# Transformers
from transformers import AutoModel

# Custom imports
from FairLangProc.algorithms.output import CustomOutput



# SOURCE:
# https://github.com/CPJKU/ModularizedDebiasing/tree/main




#=====================================================================
#                           UTILS
#=====================================================================


def dict_to_device(d: dict, device: Union[str, torch.device]) -> dict:
    return {k:v.to(device) for k,v in d.items()}


def get_param_from_name(
    model: torch.nn.Module,
    param_name: str
):
    return reduce(lambda a,b: getattr(a,b), [model] + param_name.split("."))


def concrete_stretched(
    alpha: torch.Tensor,
    l: Union[float, int] = -1.5,
    r: Union[float, int] = 1.5,
    deterministic: bool = False
) -> torch.Tensor:
    if not deterministic:
        u = torch.zeros_like(alpha).uniform_().clamp_(0.001, 0.999)
        u_term = u.log() - (1-u).log()
    else:
        u_term = 0.
    s = (torch.sigmoid(u_term + alpha))
    s_stretched = s*(r-l) + l
    z = s_stretched.clamp(0, 1000).clamp(-1000, 1)
    return z



#=====================================================================
#                       PARAMETRIZATION CLASSES
#=====================================================================



class DiffWeightFinetune(nn.Module):

    def __init__(
        self,
        weight: nn.Parameter,
        alpha_init: float,
        concrete_lower: float,
        concrete_upper: float,
        structured: bool
    ):
        super().__init__()
        self.concrete_lower = concrete_lower
        self.concrete_upper = concrete_upper
        self.structured = structured

        self.register_parameter("finetune", Parameter(torch.clone(weight)))
        self.register_parameter("alpha", Parameter(torch.zeros_like(weight) + alpha_init))

        if structured:
            self.register_parameter("alpha_group", Parameter(torch.zeros((1,), device=weight.device) + alpha_init))

        self.active = True

    def forward(self, X):
        if not self.active: return X
        diff = (self.finetune - X).detach()
        return (self.finetune - diff) + self.diff_weight(X)

    def diff_weight(self, X):
        return self.z * (self.finetune - X)

    @property
    def z(self) -> Parameter:
        z = self.dist(self.alpha)
        if self.structured:
            z *= self.dist(self.alpha_group)
        return z

    @property
    def alpha_weights(self) -> list:
        alpha = [self.alpha]
        if self.structured:
            alpha.append(self.alpha_group)
        return alpha

    def dist(self, alpha) -> torch.Tensor:
        return concrete_stretched(
            alpha,
            l=self.concrete_lower,
            r=self.concrete_upper,
            deterministic=(not self.training)
        )

    def set_frozen(self, frozen: bool) -> None:
        self.finetune.requires_grad = not frozen
        self.alpha.requires_grad = not frozen
        if self.structured:
            self.alpha_group.requires_grad = not frozen
        if frozen:
            self.eval()
        else:
            self.train()



class DiffWeightFixmask(nn.Module):

    def __init__(self, diff_weight: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self.register_parameter("diff_weight", Parameter(diff_weight * mask))
        self.register_parameter("mask", Parameter(mask, requires_grad=False))
        self.active = True

    def forward(self, X):
        if not self.active: return X
        return X + self.mask * self.diff_weight

    def set_frozen(self, frozen: bool) -> None:
        self.diff_weight.requires_grad = not frozen


#=====================================================================
#                           MODEL CLASS
#=====================================================================

class ModelState(Enum):
    INIT = auto()
    FINETUNING = auto()
    FIXMASK = auto()


class BaseModel(nn.Module, ABC):

    @property
    def encoder_module(self) -> torch.nn.Module:
        if isinstance(self.encoder, torch.nn.DataParallel):
            return self.encoder.module
        else:
            return self.encoder


    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device


    @property
    def model_type(self) -> str:
        return self.encoder_module.config.model_type


    @property
    def model_name(self) -> str:
        return self.encoder_module.config._name_or_path


    @property
    def hidden_size(self) -> int:
        return self.encoder_module.embeddings.word_embeddings.embedding_dim


    @property
    def total_layers(self) -> int:
        possible_keys = ["num_hidden_layers", "n_layer"]
        cfg = self.encoder_module.config
        for k in possible_keys:
            if k in cfg.__dict__:
                return getattr(cfg, k) + 1 # +1 for embedding layer and last layer
        raise Exception("number of layers of pre trained model could not be determined")


    def __init__(self, encoder: nn.Module = None, encoder_state_dict: OrderedDict = None):
        super().__init__()
        self.encoder = encoder

        if encoder_state_dict is not None:
            self.encoder.load_state_dict(encoder_state_dict)
            self.state_dict_init = True
        else:
            self.state_dict_init = False

    @abstractmethod
    def _forward(self, **x) -> torch.Tensor:
        pass


    def get_layer_idx_from_module(self, module_name: str) -> int:
        # get layer index based on module name
        if self.model_type == "xlnet":
            search_str_emb = "word_embedding"
            search_str_hidden = "layer"
        else:
            search_str_emb = "embeddings"
            search_str_hidden = "encoder.layer"

        if search_str_emb in module_name:
            return 0
        elif search_str_hidden in module_name:
            return int(module_name.split(search_str_hidden + ".")[1].split(".")[0]) + 1
        elif 'pooler.dense' in module_name:
            return self.total_layers - 1
        else:
            warnings.warn(f"layer idx could not be determined for module_name {module_name}")


    def to(self, device: Union[list, Union[str, torch.device]], *args, **kwargs) -> None:
        self._remove_parallel()
        if isinstance(device, list):
            super().to(device[0])
            if len(device)>1:
                asssert_fn = lambda x: x=="cuda" if isinstance(x, str) else x.type=="cuda"
                assert all([asssert_fn(d) for d in device]), "if list of devices is given, all must be of type 'cuda'"
                self.encoder = torch.nn.DataParallel(self.encoder, device_ids=device)
        else:
            super().to(device)


    def cpu(self):
        self._remove_parallel()
        super().cpu()


    def cuda(self, *args, **kwargs) -> None:
        self._remove_parallel()
        super().cuda(*args, **kwargs)


    def _remove_parallel(self) -> None:
        if isinstance(self.encoder, torch.nn.DataParallel):
            self.encoder = self.encoder.module




class BasePruningModel(BaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_state = ModelState.INIT
        self.fixmask_pct = None


    @property
    def parametrized(self) -> bool:
        return (self.model_state == ModelState.FINETUNING or self.model_state == ModelState.FIXMASK)


    @property
    def fixmask_state(self) -> bool:
        return self.model_state == ModelState.FIXMASK


    @property
    def finetune_state(self) -> bool:
        return self.model_state == ModelState.FINETUNING


    @property
    def n_parametrizations(self) -> int:
        return len(list(self.get_encoder_base_modules()[0].parametrizations.values())[0])


    @staticmethod
    def get_log_ratio(concrete_lower: float, concrete_upper: float) -> int:
        return 0 if (concrete_lower == 0) else math.log(-concrete_lower / concrete_upper)


    @staticmethod
    def get_l0_norm_term(alpha: torch.Tensor, log_ratio: float) -> torch.Tensor:
        return torch.sigmoid(alpha - log_ratio).sum()


    def get_encoder_base_modules(self, return_names: bool = False):
        if self.parametrized:
            check_fn = lambda m: hasattr(m, "parametrizations")
        else:
            check_fn = lambda m: len(m._parameters)>0
        return [(n,m) if return_names else m for n,m in self.encoder.named_modules() if check_fn(m)]


    def get_sparsity_pen(self, sparsity_pen: Union[float, List[float], Tuple[float]]) -> None:
        if isinstance(sparsity_pen, (list, tuple)):
            assert len(sparsity_pen) == self.total_layers,  "invalid sparsity penalty per layer: # of layers mismatch"
            return sparsity_pen
        else:
            return [sparsity_pen] * self.total_layers


    def _get_sparsity_loss(self, log_ratio: float, sparsity_pen: list, idx: int) -> torch.Tensor:
        assert self.finetune_state, "model needs to be in finetuning state"
        l0_pen = 0.
        for module_name, base_module in self.get_encoder_base_modules(return_names=True):
            layer_idx = self.get_layer_idx_from_module(module_name)
            module_pen = 0.
            for n, par_list in list(base_module.parametrizations.items()):
                for a in par_list[idx].alpha_weights:
                    module_pen += self.get_l0_norm_term(a, log_ratio)
            l0_pen += (module_pen * sparsity_pen[layer_idx])
        return l0_pen


    @torch.no_grad()
    def _count_non_zero_params(self, *args, **kwargs) -> Tuple[int, int, int]:
        assert self.parametrized, "Function only implemented for diff pruning"

        l = [self._count_non_zero_params_for_module(m, *args, **kwargs) for m in self.get_encoder_base_modules()]
        return [sum(x) for x in list(zip(*l))]


    @torch.no_grad()
    def _count_non_zero_params_per_layer(self, *args, **kwargs) -> Dict[int, Tuple[int, int, int]]:
        assert self.parametrized, "Function only implemented for diff pruning"

        t = torch.zeros((self.total_layers, 3), dtype=int)
        for module_name, base_module in self.get_encoder_base_modules(return_names=True):
            layer_idx = self.get_layer_idx_from_module(module_name)
            counts = self._count_non_zero_params_for_module(base_module, *args, **kwargs)
            t[layer_idx] += torch.tensor(counts)
        return {i:v.tolist() for i,v in enumerate(t)}


    @torch.no_grad()
    def _count_non_zero_params_for_module(self, m: torch.nn.Module, idx: Optional[int] = None, merged: bool = False) -> Tuple[int, int, int]:

        def count_fn(p, binary: bool):
            if binary:
                p = p.bool()
                n_p = p.numel()
                n_p_zero = (~p).sum()
                n_p_one = (n_p - n_p_zero)
            else:
                n_p = p.numel()
                n_p_zero = (p == 0.).sum()
                n_p_one = (p == 1.).sum()
            return torch.tensor([n_p, n_p_zero, n_p_one])

        assert hasattr(m, "parametrizations"), "module has no parametrizations"
        p_counts = torch.zeros((3,), dtype=int)
        with self.deterministic():
            for n, par_list in list(m.parametrizations.items()):
                if merged:
                    if isinstance(par_list[0], DiffWeightFixmask):
                        p = torch.stack([x.mask for x in par_list]).sum(0)
                    else:
                        p = torch.stack([(x.z != 0.) for x in par_list]).sum(0)
                    p_counts += count_fn(p, True)
                else:
                    if idx is not None: par_list = [par_list[idx]]
                    for par in par_list:
                        p = par.mask if isinstance(par, DiffWeightFixmask) else par.z
                        p_counts += count_fn(p, p.dtype==torch.bool)

        return p_counts.tolist()


    def _remove_parametrizations(self, leave_parametrized: bool = True) -> None:
        self._freeze_parametrizations(True)
        for module in self.get_encoder_base_modules():
            try:
                for n in list(module.parametrizations):
                    parametrize.remove_parametrizations(module, n, leave_parametrized=leave_parametrized)
            except AttributeError:
                pass
        self.model_state = ModelState.INIT


    def _add_diff_parametrizations(self, n_parametrizations: int = 1, p_requires_grad: bool = False, fixmask_init: bool = False, **kwargs) -> None:
        assert not self.parametrized, "cannot add diff parametrizations because of existing parametrizations in the model"
        for base_module in self.get_encoder_base_modules():
            for n,p in list(base_module.named_parameters()):
                p.requires_grad = p_requires_grad
                for _ in range(n_parametrizations): # number of diff networks to add
                    if fixmask_init:
                        # in case of fixmask init, can only initalize with dummy values
                        parametrize.register_parametrization(base_module, n, DiffWeightFixmask(
                            torch.zeros_like(p), torch.ones_like(p, dtype=bool)
                        ))
                    else:
                        parametrize.register_parametrization(base_module, n, DiffWeightFinetune(p, **kwargs))
        if fixmask_init:
            self.model_state = ModelState.FIXMASK
        else:
            self.model_state = ModelState.FINETUNING


    def _parametrizations_fn(self, fn: Callable, idx: Optional[int] = None):
        for base_module in self.get_encoder_base_modules():
            try:
                for par_list in base_module.parametrizations.values():
                    if idx is not None:
                        try:
                            fn(par_list[idx])
                        except IndexError:
                            pass
                    else:
                        for par in par_list:
                            fn(par)
            except AttributeError:
                pass


    def _parametrizations_set_attr(self, attr: str, v, idx: Optional[int] = None):
        fn = lambda x: setattr(x, attr, v)
        self._parametrizations_fn(fn, idx)


    def _activate_parametrizations(self, active: bool, idx: int):
        fn = lambda x: setattr(x, "active", active)
        self._parametrizations_fn(fn, idx)


    def _freeze_parametrizations(self, frozen: bool, idx: Optional[int] = None):
        fn = lambda x: x.set_frozen(frozen)
        self._parametrizations_fn(fn, idx)


    def _freeze_original_parameters(self, frozen: bool):
        for base_module in self.get_encoder_base_modules():
            for par_list in base_module.parametrizations.values():
                par_list.original.requires_grad = not frozen


    @torch.no_grad()
    def _finetune_to_fixmask(
        self,
        pct: Optional[float] = None,
        sequential: Union[bool, list, tuple] = True,
        merged_cutoff: bool = False,
        merged_min_pct: float = 0.01
    ) -> None:

        if isinstance(sequential, (list, tuple)):
            assert len(sequential) == self.n_parametrizations, "if sequential is list, needs to equal self.n_parametrizations"
        else:
            sequential = [sequential] * self.n_parametrizations

        def _get_cutoff(values, pct, abs = True):
            k = int(round(len(values) * pct, 0))
            if abs: values = torch.abs(values)
            return torch.topk(values, k+1, largest=True, sorted=True)[0][-1]

        assert self.model_state == ModelState.FINETUNING, "model needs to be in finetuning state"

        with self.deterministic():

            if pct is not None:

                diff_weights_abs = [torch.tensor([])] * self.n_parametrizations
                for base_module in self.get_encoder_base_modules():
                    for n, par_list in list(base_module.parametrizations.items()):
                        w = par_list.original.detach()
                        for idx, seq in enumerate(sequential):
                            diff_weight = par_list[idx].diff_weight(w)
                            diff_weights_abs[idx] = torch.cat([diff_weights_abs[idx], torch.abs(diff_weight.flatten().cpu())])
                            if seq: w = diff_weight + w

                if merged_cutoff and (self.n_parametrizations > 1):
                    min_cutoffs = [_get_cutoff(x, merged_min_pct, abs=False) for x in diff_weights_abs]
                    if merged_min_pct >= pct:
                        print(f"merged_min_pct >= pct, using target sparsity merged_min_pct={merged_min_pct}")
                        cutoffs = min_cutoffs
                    else:
                        remaining = torch.cat([x[x<c] for x,c in zip(diff_weights_abs, min_cutoffs)])
                        remaining_cutoff = _get_cutoff(remaining, pct - merged_min_pct)
                        cutoffs = [min(remaining_cutoff, c) for c in min_cutoffs]
                else:
                    cutoffs = [_get_cutoff(x, pct, abs=False) for x in diff_weights_abs]

            for base_module in self.get_encoder_base_modules():
                for n, par_list in list(base_module.parametrizations.items()):
                    diff_weights = []
                    w = par_list.original
                    for idx, seq in enumerate(sequential):
                        diff_weight = par_list[idx].diff_weight(w)
                        if pct is not None:
                            i = 0 if merged_cutoff else idx
                            diff_mask = (torch.abs(diff_weight) > cutoffs[i])
                        else:
                            diff_mask = ~torch.isclose(diff_weight, torch.tensor(0.), rtol=1e-8)
                        diff_weights.append((diff_weight, diff_mask))
                        if seq: w = diff_weight + w

                    parametrize.remove_parametrizations(base_module, n, leave_parametrized=False)
                    for (diff_weight, diff_mask) in diff_weights:
                        parametrize.register_parametrization(base_module, n, DiffWeightFixmask(diff_weight, diff_mask))

        self.model_state = ModelState.FIXMASK
        self.fixmask_pct = pct


    def _get_diff_param_groups(
        self,
        learning_rate: float,
        weight_decay: float = 0.0,
        learning_rate_alpha: Optional[float] = None,
        idx: Optional[int] = None,
    ) -> list:

        if idx is None:
            idx_len = 0
            idx = ""
        else:
            idx_len = len(str(idx))

        if self.model_state == ModelState.FIXMASK:
            return [
                {
                    "params": [p for n,p in self.encoder.named_parameters() if n[-(12+idx_len):] == f"{idx}.diff_weight"],
                    "weight_decay": weight_decay,
                    "lr": learning_rate
                }
            ]
        else:
            return [
                {
                    "params": [p for n,p in self.encoder.named_parameters() if n[-(9+idx_len):] == f"{idx}.finetune"],
                    "weight_decay": weight_decay,
                    "lr": learning_rate
                },
                {
                    "params": [p for n,p in self.encoder.named_parameters() if n[-(6+idx_len):]==f"{idx}.alpha" or n[-(12+idx_len):]==f"{idx}.alpha_group"],
                    "lr": learning_rate_alpha
                }
            ]


    @contextlib.contextmanager
    def deterministic(self):
        tmp_state = self.training
        if tmp_state: self.eval()
        yield
        if tmp_state: self.train()


    def _as_module(self, named_parameter_list: list):
        new_model = AutoModel.from_pretrained(self.model_name)
        with torch.no_grad():
            for p_name, p in named_parameter_list:
                _p = reduce(lambda a,b: getattr(a,b), [new_model] + p_name.split("."))
                _p.copy_(p)
        return new_model


    def get_diff_weights(self, idx: int, as_module: bool = False):
        res = []
        p_names = [n[:-9] for n, _ in self.encoder.named_parameters() if n[-9:]==".original"]
        with torch.no_grad():
            for p_name in p_names:
                par_list = reduce(lambda a,b: getattr(a,b), [self.encoder] + p_name.split("."))
                par = par_list[idx]
                if isinstance(par, DiffWeightFixmask):
                    diff_weight = par.mask * par.diff_weight
                elif isinstance(par, DiffWeightFinetune):
                    w = par_list.original.detach()
                    diff_weight = par.diff_weight(w)
                res.append((p_name.replace(".parametrizations", ""), diff_weight))

        if as_module:
            return self._as_module(res)
        else:
            return res


    def get_base_weights(self, as_module: bool = False):
        if self.parametrized:
            res = [(n[:-9].replace(".parametrizations", ""), p) for n,p in self.encoder.named_parameters() if n[-9:]==".original"]
        else:
            res = list(self.encoder.named_parameters())

        if as_module:
            return self._as_module(res)
        else:
            return res


    @torch.no_grad()
    def load_state_dict_to_parametrizations(
        self,
        encoder_state_dict: OrderedDict,
        idx: int = 0
    ):
        assert not any([("parametrizations" in k) for k in encoder_state_dict.keys()]), "cant use parametrized state dict"

        for k,v in encoder_state_dict.items():
            k_parts = k.split(".")
            try:
                par_list = get_param_from_name(self.encoder, ".".join(k_parts[:-1] + ["parametrizations"]))
            except:
                continue
            par = get_param_from_name(par_list, ".".join([k_parts[-1], str(idx)]))
            if isinstance(par, DiffWeightFinetune):
                par.finetune.copy_(v)
            elif isinstance(par, DiffWeightFixmask):
                diff = v - par_list.original
                par.diff_weight.copy_(par.mask * diff)




class DiffPrunedDebiasing(BasePruningModel, ABC):
    """
    Implements differ pruning for bias mitigation in pretrained models.
    
    Args:
        model (nn.Module):          Pretrained model (e.g., BERT, GPT-2).
        input_ids_A (torch.Tensor): Tensor with ids of text with demographic information of group A.
        input_ids_B (torch.Tensor): Tensor with ids of text with demographic information of group B.
        lambda_sparse (float):      Weight for sparsity loss.
        lambda_bias (float):        Weight for bias mitigation loss.
        bias_kernel (Callable):     Kernel for the embeddings of the bias loss. If None, defaults to the identity.
        upper (float):              Parameter for concrete relaxation (has to be > 1).
        lower (float):              Parameter for concrete relaxation (has to be < 0).
        temp (float):               Temperature for concrete relaxation loss.
    """

    def __init__(
        self,
        model: nn.Module,
        encoder: nn.Module,
        input_ids_A: torch.Tensor,
        input_ids_B: torch.Tensor,
        lambda_sparse: float = 1.0,
        lambda_bias: float = 1.0,
        bias_kernel: Callable = None,
        fixmask_init: bool = False,
        alpha_init: Optional[Union[int, float]] = 5,
        structured_diff_prunning: Optional[bool] = False,
        upper: float = 1.1,
        lower: float = -0.1
    ):
        super().__init__(encoder = encoder)
        self.model = model
        self.encoder = encoder

        self.lambda_sparse = lambda_sparse
        self.lambda_bias = lambda_bias
        self.sparsity_pen = self.get_sparsity_pen(self.lambda_bias)      

        self.upper = upper
        self.lower = lower
        self.log_ratio = math.log(-self.upper/self.lower)

        self.kernel = bias_kernel
        self.inputs_A = input_ids_A
        self.inputs_B = input_ids_B

        super()._add_diff_parametrizations(
            n_parametrizations = 1,
            p_requires_grad = False,
            fixmask_init = fixmask_init,
            alpha_init = alpha_init,
            concrete_lower = self.lower,
            concrete_upper = self.upper,
            structured = structured_diff_prunning
        )


    def _get_bias_loss(self):
        """
        Compute debias loss as the difference of the kernel of the counterfactual pairs
        """ 
        # Get hidden states from last layer
        group_a = self._forward(**self.inputs_A)
        group_b = self._forward(**self.inputs_B)
        
        group_a_mean = group_a.mean(dim=0)
        group_b_mean = group_b.mean(dim=0)
        
        return F.mse_loss(
            group_a_mean.requires_grad_(True), 
            group_b_mean.requires_grad_(True)
        )
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
            )
        
        if labels is not None:
            task_loss = outputs.loss
            bias_loss = self.lambda_bias*self._get_bias_loss()
            sparse_loss = self._get_sparse_loss(log_ratio = self.log_ratio, sparsity_pen = self.sparsity_pen, idx = 0)
            loss = task_loss + bias_loss + sparse_loss
        else:
            loss = None

        return CustomOutput(loss = loss, logits = outputs.logits)


class DiffPrunningBERT(DiffPrunedDebiasing):

    def _get_encoder(self):
        return self.model.bert

    def _forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
            )
        return outputs.last_hidden_state[:,0,:]

# class DiffPrunedDebiasing(nn.Module, ABC):
#     """
#     Implements differ pruning for bias mitigation in pretrained models.
#     
#     Args:
#         model (nn.Module):          Pretrained model (e.g., BERT, GPT-2).
#         input_ids_A (torch.Tensor): Tensor with ids of text with demographic information of group A.
#         input_ids_B (torch.Tensor): Tensor with ids of text with demographic information of group B.
#         lambda_sparse (float):      Weight for sparsity loss.
#         lambda_bias (float):        Weight for bias mitigation loss.
#         bias_kernel (Callable):     Kernel for the embeddings of the bias loss. If None, defaults to the identity.
#         upper (float):              Parameter for concrete relaxation (has to be > 1).
#         lower (float):              Parameter for concrete relaxation (has to be < 0).
#         temp (float):               Temperature for concrete relaxation loss.
#     """
# 
#     def __init__(
#         self,
#         model: nn.Module,
#         input_ids_A: torch.Tensor,
#         input_ids_B: torch.Tensor,
#         lambda_sparse: float = 1.0,
#         lambda_bias: float = 1.0,
#         bias_kernel: Callable = None,
#         fixmask_init: bool = False,
#         alpha_init: Optional[Union[int, float]] = 5,
#         structured_diff_prunning: Optional[bool] = False,
#         upper: float = 1.1,
#         lower: float = -0.1
#     ):
#         super().__init__()
#         self.model = model
#         self.encoder = self._get_encoder()
# 
#         self.lambda_sparse = lambda_sparse
#         self.lambda_bias = lambda_bias
#         self.sparsity_pen = self.get_sparsity_pen(self.lambda_bias)      
# 
# 
#         self.upper = upper
#         self.lower = lower
#         self.log_ratio = math.log(-self.upper/self.lower)
# 
#         self.kernel = bias_kernel
#         self.inputs_A = input_ids_A
#         self.inputs_B = input_ids_B
# 
#         self.model_state = ModelState.INIT
#         self.fixmask_pct = None
# 
#         self._add_diff_parametrizations(
#             n_parametrizations = 1,
#             p_requires_grad = False,
#             fixmask_init = fixmask_init,
#             alpha_init = alpha_init,
#             concrete_lower = self.lower,
#             concrete_upper = self.upper,
#             structured = structured_diff_prunning
#             )
# 
#     def forward(self, input_ids, attention_mask=None, labels=None):
#         outputs = self.model(
#             input_ids = input_ids,
#             attention_mask = attention_mask,
#             labels = labels
#             )
#         loss = None
#         if labels is not None:
# 
#             loss_main = outputs.loss
#             loss_sparsity = self._get_sparsity_loss(self.log_ratio, self.sparsity_pen, 0)
#             loss_bias = self._get_bias_loss()
# 
#             loss = loss_main + self.lambda_bias*loss_sparsity + self.lambda_bias*loss_bias
#         
# 
#         return CustomOutput(
#             logits = outputs.logits,
#             loss = loss
#         )
#     
#     @property
#     def encoder_module(self) -> torch.nn.Module:
#         if isinstance(self.encoder, torch.nn.DataParallel):
#             return self.encoder.module
#         else:
#             return self.encoder
# 
#     @property
#     def device(self) -> torch.device:
#         return next(self.encoder.parameters()).device
# 
# 
#     @property
#     def model_type(self) -> str:
#         return self.encoder_module.config.model_type
# 
# 
#     @property
#     def model_name(self) -> str:
#         return self.encoder_module.config._name_or_path
# 
# 
#     @property
#     def hidden_size(self) -> int:
#         return self.encoder_module.embeddings.word_embeddings.embedding_dim
# 
#     @property
#     def parametrized(self) -> bool:
#         return (self.model_state == ModelState.FINETUNING or self.model_state == ModelState.FIXMASK)
# 
# 
#     @property
#     def fixmask_state(self) -> bool:
#         return self.model_state == ModelState.FIXMASK
# 
# 
#     @property
#     def finetune_state(self) -> bool:
#         return self.model_state == ModelState.FINETUNING
# 
#     def get_layer_idx_from_module(self, module_name: str) -> int:
#         # get layer index based on module name
#         if self.model_type == "xlnet":
#             search_str_emb = "word_embedding"
#             search_str_hidden = "layer"
#         else:
#             search_str_emb = "embeddings"
#             search_str_hidden = "encoder.layer"
# 
#         if search_str_emb in module_name:
#             return 0
#         elif search_str_hidden in module_name:
#             return int(module_name.split(search_str_hidden + ".")[1].split(".")[0]) + 1
#         elif 'pooler.dense' in module_name:
#             return self.total_layers - 1
#         else:
#             warnings.warn(f"layer idx could not be determined for module_name {module_name}")
# 
# 
#     def get_encoder_base_modules(self, return_names: bool = False):
#         if self.parametrized:
#             check_fn = lambda m: hasattr(m, "parametrizations")
#         else:
#             check_fn = lambda m: len(m._parameters)>0
#         return [(n,m) if return_names else m for n,m in self.encoder.named_modules() if check_fn(m)]
# 
# 
#     def get_sparsity_pen(self, sparsity_pen: Union[float, List[float], Tuple[float]]) -> None:
#         if isinstance(sparsity_pen, (list, tuple)):
#             assert len(sparsity_pen) == self.total_layers,  "invalid sparsity penalty per layer: # of layers mismatch"
#             return sparsity_pen
#         else:
#             return [sparsity_pen] * self.total_layers
#         
# 
#     def _add_diff_parametrizations(self, n_parametrizations: int = 1, p_requires_grad: bool = False, fixmask_init: bool = False, **kwargs) -> None:
#         assert not self.parametrized, "cannot add diff parametrizations because of existing parametrizations in the model"
#         for base_module in self.get_encoder_base_modules():
#             for n,p in list(base_module.named_parameters()):
#                 p.requires_grad = p_requires_grad
#                 for _ in range(n_parametrizations): # number of diff networks to add
#                     if fixmask_init:
#                         # in case of fixmask init, can only initalize with dummy values
#                         parametrize.register_parametrization(base_module, n, DiffWeightFixmask(
#                             torch.zeros_like(p), torch.ones_like(p, dtype=bool)
#                         ))
#                     else:
#                         parametrize.register_parametrization(base_module, n, DiffWeightFinetune(p, **kwargs))
#         if fixmask_init:
#             self.model_state = ModelState.FIXMASK
#         else:
#             self.model_state = ModelState.FINETUNING
# 
#     
#     def _remove_parametrizations(self, leave_parametrized: bool = True) -> None:
#         self._freeze_parametrizations(True)
#         for module in self.get_encoder_base_modules():
#             try:
#                 for n in list(module.parametrizations):
#                     parametrize.remove_parametrizations(module, n, leave_parametrized=leave_parametrized)
#             except AttributeError:
#                 pass
#         self.model_state = ModelState.INIT
# 
#     @staticmethod
#     def get_l0_norm_term(alpha: torch.Tensor, log_ratio: float) -> torch.Tensor:
#         return torch.sigmoid(alpha - log_ratio).sum()
#     
#     def _get_sparsity_loss(self, log_ratio: float, sparsity_pen: list, idx: int) -> torch.Tensor:
#         assert self.finetune_state, "model needs to be in finetuning state"
#         l0_pen = 0.
#         for module_name, base_module in self.get_encoder_base_modules(return_names=True):
#             layer_idx = self.get_layer_idx_from_module(module_name)
#             module_pen = 0.
#             for n, par_list in list(base_module.parametrizations.items()):
#                 for a in par_list[idx].alpha_weights:
#                     module_pen += self.get_l0_norm_term(a, log_ratio)
#             l0_pen += (module_pen * sparsity_pen[layer_idx])
#         return l0_pen
#     
#     def _get_bias_loss(self):
#         """
#         Compute debias loss as the difference of the kernel of the counterfactual pairs
#         """ 
#         # Get hidden states from last layer
#         group_a = self.encoder(**self.inputs_A)
#         group_b = self.encoder(**self.inputs_B)
#         
#         group_a_mean = group_a.mean(dim=0)
#         group_b_mean = group_b.mean(dim=0)
#         
#         return F.mse_loss(
#             group_a_mean.requires_grad_(True), 
#             group_b_mean.requires_grad_(True)
#             )
# 
# 
# class DiffPrunningBERT(DiffPrunedDebiasing):
# 
#     def _get_encoder(self):
#         return self.model.bert
# 
#     def _get_head(self):
#         return self.model.classifier