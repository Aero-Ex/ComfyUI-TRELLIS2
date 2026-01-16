import torch
import numpy as np
from unittest.mock import MagicMock
import functools

# Mocking the logic from trellis2_image_to_3d.py
def inject_sampler_multi_image_logic(sampler, num_images, num_steps, mode='stochastic'):
    sampler._old_inference_model = sampler._inference_model
    SAMPLER_KWARGS = {'neg_cond', 'guidance_strength', 'guidance_interval', 'guidance_rescale'}

    if mode == 'stochastic':
        cond_indices = (np.arange(num_steps) % num_images).tolist()
        step_cache = {}

        def _new_inference_model(self, model, x_t, t, cond, **kwargs):
            t_key = float(t)
            if t_key not in step_cache:
                step_cache[t_key] = cond_indices.pop(0)
            cond_idx = step_cache[t_key]
            cond_i = cond[cond_idx]
            return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)

    elif mode == 'multidiffusion':
        def _new_inference_model(self, model, x_t, t, cond, neg_cond, guidance_strength, **kwargs):
            preds = []
            for i in range(len(cond)):
                pred = self._old_inference_model(model, x_t, t, cond[i], neg_cond=neg_cond, guidance_strength=1, **kwargs)
                preds.append(pred)
            avg_pred_pos = sum(preds) / len(preds)
            pred_neg = self._old_inference_model(model, x_t, t, cond[0], neg_cond=neg_cond, guidance_strength=0, **kwargs)
            return guidance_strength * avg_pred_pos + (1 - guidance_strength) * pred_neg
    
    sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

def test_logic():
    sampler = MagicMock()
    sampler._inference_model = MagicMock(side_effect=lambda model, x_t, t, cond, **kwargs: f"result_{kwargs.get('guidance_strength', 'none')}_{id(cond)}")
    
    cond = [object() for _ in range(3)]
    neg_cond = object()
    
    # Test Stochastic
    print("\n--- Testing Stochastic ---")
    inject_sampler_multi_image_logic(sampler, 3, 6, mode='stochastic')
    # t=1.0
    res1 = sampler._inference_model(None, None, 1.0, cond)
    res2 = sampler._inference_model(None, None, 1.0, cond)
    # t=0.8
    res3 = sampler._inference_model(None, None, 0.8, cond)
    
    print(f"t=1.0 call 1: {res1}")
    print(f"t=1.0 call 2: {res2}")
    print(f"t=0.8 call 1: {res3}")
    
    assert res1 == res2
    assert res1 != res3
    print("Stochastic consistency OK")

    # Test Multidiffusion
    print("\n--- Testing Multidiffusion ---")
    sampler._inference_model = MagicMock(side_effect=lambda model, x_t, t, cond, **kwargs: 1.0 if kwargs.get('guidance_strength') == 1 else 0.0)
    inject_sampler_multi_image_logic(sampler, 3, 6, mode='multidiffusion')
    
    # guidance_strength = 7.5
    # pos calls return 1.0, neg call returns 0.0
    # result = 7.5 * 1.0 + (1 - 7.5) * 0.0 = 7.5
    res = sampler._inference_model(None, None, 1.0, cond, neg_cond, 7.5)
    print(f"Multidiffusion result (expected 7.5): {res}")
    assert res == 7.5
    
    # Check calls
    calls = sampler._old_inference_model.call_args_list
    gs_calls = [c.kwargs['guidance_strength'] for c in calls]
    print(f"Guidance strength calls: {gs_calls}")
    assert gs_calls == [1, 1, 1, 0]
    print("Multidiffusion CFG logic OK")

if __name__ == "__main__":
    test_logic()
