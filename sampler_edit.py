from __future__ import annotations
import torch
from functools import partial
import collections
import math
import logging
import numpy

import node_helpers
from comfy.ldm.modules.diffusionmodules.util import make_beta_schedule
import comfy.samplers
import comfy.sampler_helpers
import comfy.model_patcher
import comfy.patcher_extension
import comfy.hooks
import comfy.patcher_extension
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
    from comfy.model_base import BaseModel
    from comfy.controlnet import ControlBase

def editcond_set_values_with_timestep_range(conditioning, values={}, start_percent=0.0, end_percent=1.0):

    if start_percent > end_percent:
        logging.warning(f"start_percent ({start_percent}) must be <= end_percent ({end_percent})")
        return conditioning

    EPS = 1e-5 
    c = []
    for t in conditioning:
        cond_start = t[1].get("start_percent", 0.0)
        cond_end   = t[1].get("end_percent",   1.0)
        intersect_start = max(start_percent, cond_start)
        intersect_end   = min(end_percent,   cond_end)

        if intersect_start >= intersect_end: # no overlap: emit unchanged
            c.append(t)
            continue

        if intersect_start > cond_start: # part before the requested range
            c.extend(node_helpers.conditioning_set_values([t], {"start_percent": cond_start, "end_percent": intersect_start - EPS}))

        c.extend(node_helpers.conditioning_set_values([t], {**values, "start_percent": intersect_start, "end_percent": intersect_end}))

        if intersect_end < cond_end: # part after the requested range
            c.extend(node_helpers.conditioning_set_values([t], {"start_percent": intersect_end + EPS, "end_percent": cond_end}))
    return c


def get_sigma_factory(steps=20, denoise=1.0, scheduler="linear", device="cpu"):
    try:
        if denoise < 1.0:
            new_steps = int(steps / denoise)
            sigmas = comfy.samplers.calculate_sigmas(
                comfy.model_sampling.ModelSamplingDiscrete(model_config=None),
                scheduler,
                new_steps
            ).to(device)
            sigmas = sigmas[-(steps + 1):]
        else:
            sigmas = comfy.samplers.calculate_sigmas(
                comfy.model_sampling.ModelSamplingDiscrete(model_config=None),
                scheduler,
                steps
            ).to(device)
    except Exception as e:
        print(f"[EasySigmaCalculator] base sigmas error: {e}")
        sigmas = torch.linspace(14.6, 0.0, steps).to(device)

    return sigmas

def update_params_from_sigmas(sigmas, model_type):
    params = {
        "step_start": 0,
        "step_end": len(sigmas),
        "shift": 1.15,
        "cosine_s": 8e-3,
        "sigma_min": 0.002,
        "sigma_max": 120.0,
        "sigmas": sigmas,
        "multiplier": 1000
    }
    
    params["sigmas"] = sigmas
    valid_sigmas = sigmas[sigmas > 0]
    if len(valid_sigmas) > 0:
        params["sigma_min"] = float(valid_sigmas.min().item())
    else:
        params["sigma_min"] = 0.002 # fallback
    params["sigma_max"] = float(sigmas.max().item())
    params["step_start"] = 0
    params["step_end"] = len(sigmas)

    if model_type == "ModelSamplingFlux":
        params["shift"] = 1.15
    elif model_type == "StableCascadeSampling":
        params["cosine_s"] = 8e-3
    elif model_type == "ModelSamplingDiscreteFlow":
        params["multiplier"] = 1000

    return params

def sig_discrete(timesteps=1000, beta_schedule="linear", linear_start=0.00085, linear_end=0.012, zsnr=False):
    betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    if zsnr:
        sigmas = rescale_zero_terminal_snr_sigmas(sigmas)
    return sigmas

def sig_d_edm(timesteps=1000):
    # Discrete EDM  :  sigma=exp(log_sigma)=
    sigmas = torch.linspace(-10, 10, timesteps).exp()  # example
    return sigmas

def sig_c_edm(sigma_min=0.002, sigma_max=120.0, steps=1000):
    sigmas = torch.linspace(math.log(sigma_min), math.log(sigma_max), steps).exp()
    return sigmas

def sig_c_v(sigma_min=0.002, sigma_max=120.0, steps=1000):
    # ContinuousEDM base
    sigmas = torch.linspace(math.log(sigma_min), math.log(sigma_max), steps).exp()
    return sigmas

def time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)

def sig_flow(shift=1.0, timesteps=1000, multiplier=1000):
    ts = (torch.arange(1, timesteps + 1, 1) / timesteps) * multiplier
    sigmas = time_snr_shift(shift, ts)
    return sigmas

def sig_cascade(timesteps=10000, shift=1.0, cosine_s=8e-3):
    _init_alpha_cumprod = torch.cos(cosine_s / (1 + cosine_s) * torch.pi * 0.5) ** 2
    sigmas = torch.empty((timesteps), dtype=torch.float32)
    for x in range(timesteps):
        t = (x + 1) / timesteps
        alpha_cumprod = (torch.cos((t + cosine_s) / (1 + cosine_s) * torch.pi * 0.5) ** 2 / _init_alpha_cumprod)
        sigmas[x] = ((1 - alpha_cumprod) / alpha_cumprod) ** 0.5
    return sigmas

def sig_flux(timesteps=10000, shift=1.15):
    ts = (torch.arange(1, timesteps + 1, 1) / timesteps)
    sigmas = torch.tensor([flux_time_shift(shift, 1.0, t.item()) for t in ts])
    return sigmas

def sig_cosmos(sigma_min=0.002, sigma_max=120.0, steps=1000):
    sigmas = torch.linspace(math.log(sigma_min), math.log(sigma_max), steps).exp()
    return sigmas

SAMPLING_CLASS_MAP = {
    "ModelSamplingDiscrete": sig_discrete,
    "ModelSamplingDiscreteEDM": sig_d_edm,
    "ModelSamplingContinuousEDM": sig_c_edm,
    "ModelSamplingContinuousV": sig_c_v,
    "ModelSamplingDiscreteFlow": sig_flow,
    "StableCascadeSampling": sig_cascade,
    "ModelSamplingFlux": sig_flux,
    "ModelSamplingCosmosRFlow": sig_cosmos,
}

def get_sampling_by_type(model_type_key):

    model_typename = SAMPLING_CLASS_MAP.get(model_type_key, "ModelSamplingDiscrete")
    
    return model_typename

def calculate_sigmas_by_sampling(model_type_key, steps, denoise, scheduler, device, params):


    sig_func = SAMPLING_CLASS_MAP.get(model_type_key, sig_discrete)

    if sig_func:
        if model_type_key == "ModelSamplingDiscrete":
            sigmas = sig_func(timesteps=params["step_end"])
        elif model_type_key == "ModelSamplingDiscreteEDM":
            sigmas = sig_func(timesteps=params["step_end"])
        elif model_type_key in ["ModelSamplingContinuousEDM", "ModelSamplingContinuousV", "ModelSamplingCosmosRFlow"]:
            sigmas = sig_func(sigma_min=params["sigma_min"], sigma_max=params["sigma_max"], steps=params["step_end"])
        elif model_type_key == "ModelSamplingDiscreteFlow":
            sigmas = sig_func(shift=params["shift"], timesteps=params["step_end"], multiplier=params["multiplier"])
        elif model_type_key == "StableCascadeSampling":
            sigmas = sig_func(timesteps=params["step_end"], shift=params["shift"], cosine_s=params["cosine_s"])
        elif model_type_key == "ModelSamplingFlux":
            sigmas = sig_func(timesteps=params["step_end"], shift=params["shift"])
        else:
            sigmas = sig_func(timesteps=params["step_end"])
    else:
        sigmas = params.get("sigmas")
        if sigmas is None:
            sigmas = torch.linspace(14.6, 0.0, steps).to(device)
    if denoise < 1.0:
        start_idx = int(steps * (1.0 - denoise))
        sigmas = sigmas[start_idx:]

    if len(sigmas) > steps:
        sigmas = torch.nn.functional.interpolate(
            sigmas.view(1, 1, -1),
            size=steps,
            mode="linear",
            align_corners=True
        ).view(-1)

    if sigmas[-1] > 0:
        sigmas = torch.cat([sigmas, torch.tensor([0.0], device=sigmas.device)])

    return 

def edit_mask_upscale(mask, width, height, upscale_method, crop):

    if mask.dim() == 2:
        mask_4d = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask_4d = mask.unsqueeze(0)
    else:
        mask_4d = mask
    orig_shape = mask_4d.shape

    out_4d = torch.nn.functional.interpolate(mask_4d, size=(height, width), mode='bilinear', align_corners=False)

    return out_4d, out_4d


def edit_add_area_dims(area, num_dims):
    while (len(area) // 2) < num_dims:
        area = [2147483648] + area[:len(area) // 2] + [0] + area[len(area) // 2:]
    return area

def edit_get_area_and_mult(conds, x_in, timestep_in):
    dims = tuple(x_in.shape[2:])
    area = None
    strength = 1.0

    if 'timestep_start' in conds:
        timestep_start = conds['timestep_start']
        if timestep_in[0] > timestep_start:
            return None
    if 'timestep_end' in conds:
        timestep_end = conds['timestep_end']
        if timestep_in[0] < timestep_end:
            return None
    if 'area' in conds:
        area = list(conds['area'])
        area = edit_add_area_dims(area, len(dims))
        if (len(area) // 2) > len(dims):
            area = area[:len(dims)] + area[len(area) // 2:(len(area) // 2) + len(dims)]

    if 'strength' in conds:
        strength = conds['strength']

    input_x = x_in
    if area is not None:
        for i in range(len(dims)):
            area[i] = min(input_x.shape[i + 2] - area[len(dims) + i], area[i])
            input_x = input_x.narrow(i + 2, area[len(dims) + i], area[i])

    if 'mask' in conds:
        # Scale the mask to the size of the input
        # The mask should have been resized as we began the sampling process
        mask_strength = 1.0
        if "mask_strength" in conds:
            mask_strength = conds["mask_strength"]
        mask = conds['mask']
        # assert (mask.shape[1:] == x_in.shape[2:])

        mask = mask[:input_x.shape[0]]
        if area is not None:
            for i in range(len(dims)):
                mask = mask.narrow(i + 1, area[len(dims) + i], area[i])

        mask = mask * mask_strength
        mask = mask.unsqueeze(1).repeat((input_x.shape[0] // mask.shape[0], input_x.shape[1]) + (1, ) * (mask.ndim - 1))
    else:
        mask = torch.ones_like(input_x)
    mult = mask * strength

    if 'mask' not in conds and area is not None:
        fuzz = 8
        for i in range(len(dims)):
            rr = min(fuzz, mult.shape[2 + i] // 4)
            if area[len(dims) + i] != 0:
                for t in range(rr):
                    m = mult.narrow(i + 2, t, 1)
                    m *= ((1.0 / rr) * (t + 1))
            if (area[i] + area[len(dims) + i]) < x_in.shape[i + 2]:
                for t in range(rr):
                    m = mult.narrow(i + 2, area[i] - 1 - t, 1)
                    m *= ((1.0 / rr) * (t + 1))

    conditioning = {}
    model_conds = conds["model_conds"]
    for c in model_conds:
        conditioning[c] = model_conds[c].process_cond(batch_size=x_in.shape[0], area=area)

    hooks = conds.get('hooks', None)
    control = conds.get('control', None)

    patches = None
    if 'gligen' in conds:
        gligen = conds['gligen']
        patches = {}
        gligen_type = gligen[0]
        gligen_model = gligen[1]
        if gligen_type == "position":
            gligen_patch = gligen_model.model.set_position(input_x.shape, gligen[2], input_x.device)
        else:
            gligen_patch = gligen_model.model.set_empty(input_x.shape, input_x.device)

        patches['middle_patch'] = [gligen_patch]

    cond_obj = collections.namedtuple('cond_obj', ['input_x', 'mult', 'conditioning', 'area', 'control', 'patches', 'uuid', 'hooks'])

    return cond_obj(input_x, mult, conditioning, area, control, patches, conds['uuid'], hooks)


def edit_finalize_default_conds(model: 'BaseModel', hooked_to_run: dict[comfy.hooks.HookGroup,list[tuple[tuple,int]]], default_conds: list[list[dict]], x_in, timestep, model_options):
    # need to figure out remaining unmasked area for conds
    default_mults = []
    for _ in default_conds:
        default_mults.append(torch.ones_like(x_in))
    # look through each finalized cond in hooked_to_run for 'mult' and subtract it from each cond
    for lora_hooks, to_run in hooked_to_run.items():
        for cond_obj, i in to_run:
            # if no default_cond for cond_type, do nothing
            if len(default_conds[i]) == 0:
                continue
            area: list[int] = cond_obj.area
            if area is not None:
                curr_default_mult: torch.Tensor = default_mults[i]
                dims = len(area) // 2
                for i in range(dims):
                    curr_default_mult = curr_default_mult.narrow(i + 2, area[i + dims], area[i])
                curr_default_mult -= cond_obj.mult
            else:
                default_mults[i] -= cond_obj.mult
    # for each default_mult, ReLU to make negatives=0, and then check for any nonzeros
    for i, mult in enumerate(default_mults):
        # if no default_cond for cond type, do nothing
        if len(default_conds[i]) == 0:
            continue
        torch.nn.functional.relu(mult, inplace=True)
        # if mult is all zeros, then don't add default_cond
        if torch.max(mult) == 0.0:
            continue

        cond = default_conds[i]
        for x in cond:
            # do edit_get_area_and_mult to get all the expected values
            p = edit_get_area_and_mult(x, x_in, timestep)
            if p is None:
                continue
            # replace p's mult with calculated mult
            p = p._replace(mult=mult)
            if p.hooks is not None:
                model.current_patcher.prepare_hook_patches_current_keyframe(timestep, p.hooks, model_options)
            hooked_to_run.setdefault(p.hooks, list())
            hooked_to_run[p.hooks] += [(p, i)]

def edit_finalize_default_maskconds(model: 'BaseModel', hooked_to_run: dict[comfy.hooks.HookGroup,list[tuple[tuple,int]]], default_conds: list[list[dict]], x_in, timestep, model_options):
    # need to figure out remaining unmasked area for conds
    default_mults = []
    for _ in default_conds:
        default_mults.append(torch.ones_like(x_in))
    
    # look through each finalized cond in hooked_to_run for 'mult' and subtract it from each cond
    for lora_hooks, to_run in hooked_to_run.items():
        for cond_obj, i in to_run:
            # if no default_cond for cond_type, do nothing
            if len(default_conds[i]) == 0:
                continue
            area: list[int] = cond_obj.area
            if area is not None:
                curr_default_mult: torch.Tensor = default_mults[i]
                dims = len(area) // 2
                for d in range(dims):  # indexerror guard
                    curr_default_mult = curr_default_mult.narrow(d + 2, area[d + dims], area[d])
                curr_default_mult -= cond_obj.mult
            else:
                default_mults[i] -= cond_obj.mult
                
    # for each default_mult, ReLU to make negatives=0, and then check for any nonzeros
    for i, mult in enumerate(default_mults):
        # if no default_cond for cond type, do nothing
        if len(default_conds[i]) == 0:
            continue
        torch.nn.functional.relu(mult, inplace=True)
        # if mult is all zeros, then don't add default_cond
        if torch.max(mult) == 0.0:
            continue

        cond = default_conds[i]
        for x in cond:
            # get global_mask_c(default 1.0 or ex_strength)
            global_strength = x.get("mask_strength", 1.0)
            
            # do edit_get_area_and_mult to get all the expected values
            p = edit_get_area_and_mult(x, x_in, timestep)
            if p is None:
                continue

            final_mult = mult * global_strength
            
            # replace p's mult with calculated mult
            p = p._replace(mult=final_mult)
            if p.hooks is not None:
                model.current_patcher.prepare_hook_patches_current_keyframe(timestep, p.hooks, model_options)
            hooked_to_run.setdefault(p.hooks, list())
            hooked_to_run[p.hooks] += [(p, i)]

def edit_calc_cond_batch(model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep, model_options: dict[str]):
    handler: comfy.context_windows.ContextHandlerABC = model_options.get("context_handler", None)
    if handler is None or not handler.should_use_context(model, conds, x_in, timestep, model_options):
        return _edit_calc_cond_batch_outer(model, conds, x_in, timestep, model_options)
    return handler.execute(_edit_calc_cond_batch_outer, model, conds, x_in, timestep, model_options)

def _edit_calc_cond_batch_outer(model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep, model_options):
    executor = comfy.patcher_extension.WrapperExecutor.new_executor(
        _edit_calc_cond_batch,
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.CALC_COND_BATCH, model_options, is_model_options=True)
    )
    return executor.execute(model, conds, x_in, timestep, model_options)

def _edit_calc_cond_batch(model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep, model_options):
    out_conds = []
    out_counts = []
    # separate conds by matching hooks
    hooked_to_run: dict[comfy.hooks.HookGroup,list[tuple[tuple,int]]] = {}
    default_conds = []
    default_maskconds = []
    has_default_conds = False
    has_default_maskconds = False

    for i in range(len(conds)):
        out_conds.append(torch.zeros_like(x_in))
        out_counts.append(torch.ones_like(x_in) * 1e-37)

        cond = conds[i]
        default_c = []
        default_mask_c = []
        if cond is not None:
            for x in cond:
                if 'default' in x:
                    default_c.append(x)
                    has_default_conds = True
                    continue
                elif 'global_mask_c' in x:
                    default_mask_c.append(x)
                    has_default_maskconds = True
                    continue
                p = edit_get_area_and_mult(x, x_in, timestep)
                if p is None:
                    continue
                if p.hooks is not None:
                    model.current_patcher.prepare_hook_patches_current_keyframe(timestep, p.hooks, model_options)
                hooked_to_run.setdefault(p.hooks, list())
                hooked_to_run[p.hooks] += [(p, i)]
        default_conds.append(default_c)
        default_maskconds.append(default_mask_c)

    if has_default_conds:
        edit_finalize_default_conds(model, hooked_to_run, default_conds, x_in, timestep, model_options)
    if has_default_maskconds:
        edit_finalize_default_maskconds(model, hooked_to_run, default_maskconds, x_in, timestep, model_options)
    model.current_patcher.prepare_state(timestep, model_options)

    # run every hooked_to_run separately
    for hooks, to_run in hooked_to_run.items():
        while len(to_run) > 0:
            first = to_run[0]
            first_shape = first[0][0].shape
            to_batch_temp = []
            for x in range(len(to_run)):
                if comfy.samplers.can_concat_cond(to_run[x][0], first[0]):
                    to_batch_temp += [x]

            to_batch_temp.reverse()
            to_batch = to_batch_temp[:1]

            free_memory = model.current_patcher.get_free_memory(x_in.device)
            for i in range(1, len(to_batch_temp) + 1):
                batch_amount = to_batch_temp[:len(to_batch_temp)//i]
                input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
                cond_shapes = collections.defaultdict(list)
                for tt in batch_amount:
                    cond = {k: v.size() for k, v in to_run[tt][0].conditioning.items()}
                    for k, v in to_run[tt][0].conditioning.items():
                        cond_shapes[k].append(v.size())

                if model.memory_required(input_shape, cond_shapes=cond_shapes) * 1.5 < free_memory:
                    to_batch = batch_amount
                    break

            input_x = []
            mult = []
            c = []
            cond_or_uncond = []
            uuids = []
            area = []
            control = None
            patches = None
            for x in to_batch:
                o = to_run.pop(x)
                p = o[0]
                input_x.append(p.input_x)
                mult.append(p.mult)
                c.append(p.conditioning)
                area.append(p.area)
                cond_or_uncond.append(o[1])
                uuids.append(p.uuid)
                control = p.control
                patches = p.patches

            batch_chunks = len(cond_or_uncond)
            input_x = torch.cat(input_x)
            c = comfy.samplers.cond_cat(c)
            timestep_ = torch.cat([timestep] * batch_chunks)

            transformer_options = model.current_patcher.apply_hooks(hooks=hooks)
            if 'transformer_options' in model_options:
                transformer_options = comfy.patcher_extension.merge_nested_dicts(transformer_options,
                                                                                 model_options['transformer_options'],
                                                                                 copy_dict1=False)

            if patches is not None:
                transformer_options["patches"] = comfy.patcher_extension.merge_nested_dicts(
                    transformer_options.get("patches", {}),
                    patches
                )

            transformer_options["cond_or_uncond"] = cond_or_uncond[:]
            transformer_options["uuids"] = uuids[:]
            transformer_options["sigmas"] = timestep

            c['transformer_options'] = transformer_options

            if control is not None:
                c['control'] = control.get_control(input_x, timestep_, c, len(cond_or_uncond), transformer_options)

            if 'model_function_wrapper' in model_options:
                output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
            else:
                output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)

            for o in range(batch_chunks):
                cond_index = cond_or_uncond[o]
                a = area[o]
                current_mult = mult[o]
                
                if a is None:
                    out_conds[cond_index] += output[o] * mult[o]
                    out_counts[cond_index] += mult[o]
                else:
                    out_c = out_conds[cond_index]
                    out_cts = out_counts[cond_index]
                    dims = len(a) // 2
                    for i in range(dims):
                        out_c = out_c.narrow(i + 2, a[i + dims], a[i])
                        out_cts = out_cts.narrow(i + 2, a[i + dims], a[i])
                    out_c += output[o] * mult[o]
                    out_cts += mult[o]

    for i in range(len(out_conds)):
        out_conds[i] /= out_counts[i]
    
    return out_conds


def edit_cfg_func(model, cond_pred, uncond_pred, cond_scale, x, timestep, model_options={}, cond=None, uncond=None):
    if "sampler_cfg_function" in model_options:
        args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options, "input_cond": cond, "input_uncond": uncond}
        cfg_result = x - model_options["sampler_cfg_function"](args)
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "cond_scale": cond_scale, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                "sigma": timestep, "model_options": model_options, "input": x}
        cfg_result = fn(args)

    return cfg_result

#The main sampling function shared by all the samplers
#Returns denoised
def edit_sampling_func(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    conds = [cond, uncond_]
    if "sampler_calc_cond_batch_function" in model_options:
        args = {"conds": conds, "input": x, "sigma": timestep, "model": model, "model_options": model_options}
        out = model_options["sampler_calc_cond_batch_function"](args)
    else:
        out = edit_calc_cond_batch(model, conds, x, timestep, model_options)

    for fn in model_options.get("sampler_pre_cfg_function", []):
        args = {"conds":conds, "conds_out": out, "cond_scale": cond_scale, "timestep": timestep,
                "input": x, "sigma": timestep, "model": model, "model_options": model_options}
        out = fn(args)

    return edit_cfg_func(model, out[0], out[1], cond_scale, x, timestep, model_options=model_options, cond=cond, uncond=uncond_)

#The main sampling function shared by all the samplers
#Returns denoised

def edit_get_mask_aabb(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.int)

    b = masks.shape[0]

    bounding_boxes = torch.zeros((b, 4), device=masks.device, dtype=torch.int)
    is_empty = torch.zeros((b), device=masks.device, dtype=torch.bool)
    for i in range(b):
        mask = masks[i]
        if mask.numel() == 0:
            continue
        if torch.max(mask != 0) == False:
            is_empty[i] = True
            continue
        mask_2d = masks[i].view(-1, masks[i].shape[-2], masks[i].shape[-1]).max(dim=0).values
        if torch.any(mask > 0):
            y, x = torch.where(mask_2d > 0)
            bounding_boxes[i, 0] = torch.min(x)
            bounding_boxes[i, 1] = torch.min(y)
            bounding_boxes[i, 2] = torch.max(x)
            bounding_boxes[i, 3] = torch.max(y)
            is_empty[i] = False
        else:
            is_empty[i] = True

    return bounding_boxes, is_empty

def edit_resolve_areas_and_cond_masks_multidim(conditions, dims, device):
    # We need to decide on an area outside the sampling loop in order to properly generate opposite areas of equal sizes.
    # While we're doing this, we can also resolve the mask device and scaling for performance reasons.
    h_lat, w_lat = dims[-2], dims[-1]
    for i in range(len(conditions)):
        c = conditions[i]
        modified = c.copy()

        if 'area' in modified:
            area = modified['area']
            if area and area[0] == "percentage":
                a = area[1:]
                a_len = len(a) // 2
                area = ()
                for d in range(len(dims)):
                    area += (max(1, round(a[d] * dims[d])),)
                for d in range(len(dims)):
                    area += (round(a[d + a_len] * dims[d]),)
                modified['area'] = area

        if 'mask' in modified:
            mask = modified['mask']
            mask = mask.to(device=device)
            if len(mask.shape) == len(dims):
                mask = mask.unsqueeze(0)
            if mask.shape[1:] != dims:
                if mask.ndim < 3:
                    mask, taget_mask = edit_mask_upscale(mask.unsqueeze(1), dims[-1], dims[-2], 'bilinear', 'none')
                else:
                    mask, taget_mask = edit_mask_upscale(mask, dims[-1], dims[-2], 'bilinear', 'none')
            else:
                taget_mask = mask

            if modified.get("set_area_to_bounds", False): #TODO: handle dim != 2
                bounds = torch.max(torch.abs(taget_mask), dim=1, keepdim=True).values
                boxes, is_empty = edit_get_mask_aabb(bounds)
                if is_empty[0]:
                    modified['area'] = (int(h_lat), int(w_lat), 0, 0)
                else:
                    box = boxes[0]
                    H, W, Y, X = (box[3] - box[1] + 1, box[2] - box[0] + 1, box[1], box[0])
                    H = max(8, min(int(H), h_lat))
                    W = max(8, min(int(W), w_lat))
                    Y = max(0, min(int(Y), h_lat - 1))
                    X = max(0, min(int(X), w_lat - 1))
                    modified['area'] = (H, W, Y, X)
            else:
                if 'area' not in modified or modified['area'] is None:
                    modified['area'] = (int(h_lat), int(w_lat), 0, 0)
                else:
                    area = modified['area']
                    if len(area) == 4:
                        h_a, w_a, y_a, x_a = area
                        modified['area'] = (
                            max(8, min(int(h_a), h_lat)),
                            max(8, min(int(w_a), w_lat)),
                            max(0, min(int(y_a), h_lat - 1)),
                            max(0, min(int(x_a), w_lat - 1))
                        )
            
            modified['mask'] = mask
        conditions[i] = modified

def edit_create_cond_with_same_area_if_none(conds, c):
    if 'area' not in c:
        return

    def area_inside(a, area_cmp):
        a = edit_add_area_dims(a, len(area_cmp) // 2)
        area_cmp = edit_add_area_dims(area_cmp, len(a) // 2)

        a_l = len(a) // 2
        area_cmp_l = len(area_cmp) // 2
        for i in range(min(a_l, area_cmp_l)):
            if a[a_l + i] < area_cmp[area_cmp_l + i]:
                return False
        for i in range(min(a_l, area_cmp_l)):
            if (a[i] + a[a_l + i]) > (area_cmp[i] + area_cmp[area_cmp_l + i]):
                return False
        return True

    c_area = c['area']
    smallest = None
    for x in conds:
        if 'area' in x:
            a = x['area']
            if area_inside(c_area, a):
                if smallest is None:
                    smallest = x
                elif 'area' not in smallest:
                    smallest = x
                else:
                    if math.prod(smallest['area'][:len(smallest['area']) // 2]) > math.prod(a[:len(a) // 2]):
                        smallest = x
        else:
            if smallest is None:
                smallest = x
    if smallest is None:
        return
    if 'area' in smallest:
        if smallest['area'] == c_area:
            return

    out = c.copy()
    out['model_conds'] = smallest['model_conds'].copy() #TODO: which fields should be copied?
    conds += [out]


def edit_apply_empty_x_to_equal_area(conds, uncond, name, uncond_fill_func):
    cond_cnets = []
    cond_other = []
    uncond_cnets = []
    uncond_other = []
    for t in range(len(conds)):
        x = conds[t]
        if 'area' not in x:
            if name in x and x[name] is not None:
                cond_cnets.append(x[name])
            else:
                cond_other.append((x, t))
    for t in range(len(uncond)):
        x = uncond[t]
        if 'area' not in x:
            if name in x and x[name] is not None:
                uncond_cnets.append(x[name])
            else:
                uncond_other.append((x, t))

    if len(uncond_cnets) > 0:
        return

    for x in range(len(cond_cnets)):
        temp = uncond_other[x % len(uncond_other)]
        o = temp[0]
        if name in o and o[name] is not None:
            n = o.copy()
            n[name] = uncond_fill_func(cond_cnets, x)
            uncond += [n]
        else:
            n = o.copy()
            n[name] = uncond_fill_func(cond_cnets, x)
            uncond[temp[1]] = n

def edit_encode_model_conds(model_function, conds, noise, device, prompt_type, **kwargs):
    for t in range(len(conds)):
        x = conds[t]
        params = x.copy()
        params["device"] = device
        params["noise"] = noise
        default_width = None
        if len(noise.shape) >= 4: #TODO: 8 multiple should be set by the model
            default_width = noise.shape[3] * 8
        params["width"] = params.get("width", default_width)
        params["height"] = params.get("height", noise.shape[2] * 8)
        params["prompt_type"] = params.get("prompt_type", prompt_type)
        for k in kwargs:
            if k not in params:
                params[k] = kwargs[k]

        out = model_function(**params)
        x = x.copy()
        model_conds = x['model_conds'].copy()
        for k in out:
            model_conds[k] = out[k]
        x['model_conds'] = model_conds
        conds[t] = x
    return conds


def edit_process_conds(model, noise, conds, device, latent_image=None, denoise_mask=None, seed=None, latent_shapes=None):
    for k in conds:
        conds[k] = conds[k][:]
        edit_resolve_areas_and_cond_masks_multidim(conds[k], noise.shape[2:], device)

    for k in conds:
        comfy.samplers.calculate_start_end_timesteps(model, conds[k])

    if hasattr(model, 'extra_conds'):
        for k in conds:
            conds[k] = edit_encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed, latent_shapes=latent_shapes)

    #make sure each cond area has an opposite one with the same area
    for k in conds:
        for c in conds[k]:
            for kk in conds:
                if k != kk:
                    edit_create_cond_with_same_area_if_none(conds[kk], c)

    for k in conds:
        for c in conds[k]:
            if 'hooks' in c:
                for hook in c['hooks'].hooks:
                    hook.initialize_timesteps(model)

    for k in conds:
        comfy.samplers.pre_run_control(model, conds[k])

    if "positive" in conds:
        positive = conds["positive"]
        for k in conds:
            if k != "positive":
                edit_apply_empty_x_to_equal_area(list(filter(lambda c: c.get('control_apply_to_uncond', False) == True, positive)), conds[k], 'control', lambda cond_cnets, x: cond_cnets[x])
                edit_apply_empty_x_to_equal_area(positive, conds[k], 'gligen', lambda cond_cnets, x: cond_cnets[x])

    return conds


class edit_CFGGuider:
    def __init__(self, model_patcher: ModelPatcher):
        self.model_patcher = model_patcher
        self.model_options = model_patcher.model_options
        self.original_conds = {}
        self.cfg = 1.0

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg):
        self.cfg = cfg

    def inner_set_conds(self, conds):
        for k in conds:
            if self.model_patcher.is_dynamic() and comfy.sampler_helpers.cond_has_hooks(conds[k]):
                self.model_patcher = self.model_patcher.get_non_dynamic_delegate()
            self.original_conds[k] = comfy.sampler_helpers.convert_cond(conds[k])

    def __call__(self, *args, **kwargs):
        return self.outer_predict_noise(*args, **kwargs)

    def outer_predict_noise(self, x, timestep, model_options={}, seed=None):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self.predict_noise,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.PREDICT_NOISE, self.model_options, is_model_options=True)
        ).execute(x, timestep, model_options, seed)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        return edit_sampling_func(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        if latent_image is not None and torch.count_nonzero(latent_image) > 0: #Don't shift the empty latent image.
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = edit_process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)

        extra_model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
        extra_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
        extra_args = {"model_options": extra_model_options, "seed": seed}

        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            sampler.sample,
            sampler,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, extra_args["model_options"], is_model_options=True)
        )
        samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
        return self.inner_model.process_latent_out(samples.to(torch.float32))

    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None, latent_shapes=None):
        self.inner_model, self.conds, self.loaded_models = comfy.sampler_helpers.prepare_sampling(self.model_patcher, noise.shape, self.conds, self.model_options)
        device = self.model_patcher.load_device

        noise = noise.to(device=device, dtype=torch.float32)
        latent_image = latent_image.to(device=device, dtype=torch.float32)
        sigmas = sigmas.to(device)
        comfy.samplers.cast_to_load_options(self.model_options, device=device, dtype=self.model_patcher.model_dtype())

        try:
            self.model_patcher.pre_run()
            output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
        finally:
            self.model_patcher.cleanup()

        comfy.sampler_helpers.cleanup_models(self.conds, self.loaded_models)
        del self.inner_model
        del self.loaded_models
        return output

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        if sigmas.shape[-1] == 0:
            return latent_image

        if latent_image.is_nested:
            latent_image, latent_shapes = comfy.utils.pack_latents(latent_image.unbind())
            noise, _ = comfy.utils.pack_latents(noise.unbind())
        else:
            latent_shapes = [latent_image.shape]

        if denoise_mask is not None:
            if denoise_mask.is_nested:
                denoise_masks = denoise_mask.unbind()
                denoise_masks = denoise_masks[:len(latent_shapes)]
            else:
                denoise_masks = [denoise_mask]

            for i in range(len(denoise_masks), len(latent_shapes)):
                denoise_masks.append(torch.ones(latent_shapes[i]))

            for i in range(len(denoise_masks)):
                denoise_masks[i] = comfy.sampler_helpers.prepare_mask(denoise_masks[i], latent_shapes[i], self.model_patcher.load_device)

            if len(denoise_masks) > 1:
                denoise_mask, _ = comfy.utils.pack_latents(denoise_masks)
            else:
                denoise_mask = denoise_masks[0]
            denoise_mask = denoise_mask.float()

        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))
        comfy.samplers.preprocess_conds_hooks(self.conds)

        try:
            orig_model_options = self.model_options
            self.model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
            # if one hook type (or just None), then don't bother caching weights for hooks (will never change after first step)
            orig_hook_mode = self.model_patcher.hook_mode
            if comfy.samplers.get_total_hook_groups_in_conds(self.conds) <= 1:
                self.model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
            comfy.sampler_helpers.prepare_model_patcher(self.model_patcher, self.conds, self.model_options)
            comfy.samplers.filter_registered_hooks_on_conds(self.conds, self.model_options)
            executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
                self.outer_sample,
                self,
                comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, self.model_options, is_model_options=True)
            )
            output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=latent_shapes)
        finally:
            comfy.samplers.cast_to_load_options(self.model_options, device=self.model_patcher.offload_device)
            self.model_options = orig_model_options
            self.model_patcher.hook_mode = orig_hook_mode
            self.model_patcher.restore_hook_patches()

        del self.conds

        if len(latent_shapes) > 1:
            output = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(output, latent_shapes))
        return output


def edit_sample(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options={}, latent_image=None, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    cfg_guider = edit_CFGGuider(model)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg)
    return cfg_guider.sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)


def sample_custom_edit(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=None, callback=None, disable_pbar=False, seed=None):
    samples = edit_sample(model, noise, positive, negative, cfg, model.load_device, sampler, sigmas, model_options=model.model_options, latent_image=latent_image, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
    return samples