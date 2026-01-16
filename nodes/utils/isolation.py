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
            
        # Apply the base isolated decorator
        isolated_cls = base_isolated(cls)
        
        # If we have comfy-env, we want to wrap the execution method to check for direct_mode at runtime
        if HAS_COMFY_ENV:
            func_name = getattr(isolated_cls, "FUNCTION", None)
            if func_name and hasattr(isolated_cls, func_name):
                original_method = getattr(isolated_cls, func_name)
                
                @functools.wraps(original_method)
                def wrapped_method(self, *m_args, **m_kwargs):
                    # Check if first non-self argument is a config with direct_mode
                    # In TRELLIS2 nodes, the first arg after 'self' is usually model_config
                    config = None
                    if len(m_args) > 0:
                        config = m_args[0]
                    elif 'model_config' in m_kwargs:
                        config = m_kwargs['model_config']
                    
                    if config and hasattr(config, 'direct_mode') and config.direct_mode:
                        m_kwargs['local'] = True
                        
                    return original_method(self, *m_args, **m_kwargs)
                
                setattr(isolated_cls, func_name, wrapped_method)
                    
        return isolated_cls
        
    return decorator
