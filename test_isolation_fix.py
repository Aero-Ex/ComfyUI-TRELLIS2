import os
import sys
from unittest.mock import MagicMock

# Mock comfy_env
sys.modules['comfy_env'] = MagicMock()
import comfy_env

def mock_isolated(*args, **kwargs):
    def decorator(cls):
        # Mock the proxy class that comfy_env.isolated would return
        class ProxyClass(cls):
            def get_conditioning(self, *args, **kwargs):
                # Simulate the proxy behavior (calling worker.call_method)
                # In real comfy_env, this would NOT have the 'local' arg unless injected
                print(f"Proxy called with kwargs: {list(kwargs.keys())}")
                return "proxy_result"
        ProxyClass.FUNCTION = cls.FUNCTION
        return ProxyClass
    return decorator

comfy_env.isolated = mock_isolated

# Now import our isolation utility
sys.path.append('d:/@home/aero/comfy/ComfyUI/custom_nodes/ComfyUI-TRELLIS2/nodes/utils')
from isolation import smart_isolated

# Define a dummy config class
class Config:
    def __init__(self, direct_mode=False):
        self.direct_mode = direct_mode

# Define a dummy node class
@smart_isolated(env="test")
class DummyNode:
    FUNCTION = "get_conditioning"
    def get_conditioning(self, model_config, images=None):
        print(f"Original method called with direct_mode={model_config.direct_mode}")
        return "original_result"

# Test 1: Isolated mode (direct_mode=False)
print("--- Test 1: Isolated mode ---")
node = DummyNode()
config_isolated = Config(direct_mode=False)
result_isolated = node.get_conditioning(config_isolated)
print(f"Result: {result_isolated}")

# Test 2: Direct mode (direct_mode=True)
print("\n--- Test 2: Direct mode ---")
config_direct = Config(direct_mode=True)
result_direct = node.get_conditioning(config_direct)
print(f"Result: {result_direct}")
