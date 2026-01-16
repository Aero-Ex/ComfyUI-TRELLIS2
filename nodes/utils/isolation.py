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
        
        # If we have comfy-env, we want to wrap the proxy to check for direct_mode at runtime
        if HAS_COMFY_ENV:
            original_init = isolated_cls.__init__
            
            # We need to wrap the methods to check for direct_mode in the first argument (usually model_config)
            # This is complex because comfy-env replaces methods with proxies.
            # However, comfy-env's proxy checks for a 'local' keyword argument.
            
            # We can wrap the methods of the isolated class
            for attr_name in dir(isolated_cls):
                if not attr_name.startswith("_") and callable(getattr(isolated_cls, attr_name)):
                    method = getattr(isolated_cls, attr_name)
                    
                    @functools.wraps(method)
                    def wrapped_method(*m_args, **m_kwargs):
                        # Check if first non-self argument is a config with direct_mode
                        # In ComfyUI, the first arg after 'self' is usually the model_config
                        if len(m_args) > 1:
                            config = m_args[1]
                            if hasattr(config, 'direct_mode') and config.direct_mode:
                                m_kwargs['local'] = True
                        elif 'model_config' in m_kwargs:
                            config = m_kwargs['model_config']
                            if hasattr(config, 'direct_mode') and config.direct_mode:
                                m_kwargs['local'] = True
                                
                        return method(*m_args, **m_kwargs)
                    
                    setattr(isolated_cls, attr_name, wrapped_method)
                    
        return isolated_cls
        
    return decorator
