
def register():

    return "vllm_hymeta.platform.HymetaCudaPlatform"

def register_model():
    

    from .models import register_model
    register_model()