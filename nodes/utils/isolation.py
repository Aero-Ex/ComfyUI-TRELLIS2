import os
import functools

def smart_isolated(*args, **kwargs):
    """
    A smart version of the @isolated decorator that can be bypassed.
    
    Bypass triggers:
    1. TRELLIS2_DIRECT_MODE environment variable is set to "1"
    2. comfy_env is not installed
    3. direct_mode=True is passed in the model_config (handled at runtime)
    """
    try:
        from comfy_env import isolated
        base_isolated = isolated(*args, **kwargs)
        HAS_COMFY_ENV = True
    except ImportError:
        def base_isolated(cls): return cls
        HAS_COMFY_ENV = False

    def decorator(cls):
        # If environment variable is set, skip isolation entirely
        if os.environ.get("TRELLIS2_DIRECT_MODE") == "1":
            return cls
            
        if not HAS_COMFY_ENV:
            return cls
            
        # Capture the original method before it gets proxied
        func_name = getattr(cls, "FUNCTION", None)
        original_method = None
        if func_name and hasattr(cls, func_name):
            original_method = getattr(cls, func_name)

        # Apply the base isolated decorator (returns a class with proxies)
        isolated_cls = base_isolated(cls)
        
        # If we have comfy-env and a method to wrap, handle direct_mode at runtime
        if HAS_COMFY_ENV and original_method:
            proxy_method = getattr(isolated_cls, func_name)
            
            @functools.wraps(original_method)
            def wrapped_method(self, *m_args, **m_kwargs):
                # Check if first non-self argument is a config with direct_mode
                config = None
                if len(m_args) > 0:
                    config = m_args[0]
                elif 'model_config' in m_kwargs:
                    config = m_kwargs['model_config']
                
                # If direct_mode is True, bypass the proxy and call the original method directly
                if config and hasattr(config, 'direct_mode') and config.direct_mode:
                    return original_method(self, *m_args, **m_kwargs)
                
                # Otherwise, call the proxy method (which runs in the isolated venv)
                return proxy_method(self, *m_args, **m_kwargs)
            
            setattr(isolated_cls, func_name, wrapped_method)
                    
        return isolated_cls
        
    return decorator
