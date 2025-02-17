import importlib.util
import os

os.environ["MODELS_CONFIG_YAML_PATH"] = "./omni_router.yaml"
os.environ["ENABLE_BILLING"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///./omni_router.db"
os.environ["RECORD_CONTENT"] = "30"
os.environ["DEFAULT_USER_BALANCE"] = "10"
os.environ["BASE_COST_CURRENCY_UNIT"] = "$"
os.environ["ACTUAL_COST_CURRENCY_UNIT"] = "$"

module_name = "omni_router_manifold_pipeline"
module_path = "omni_router_manifold_pipeline.py"
spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(f"Loaded module: {module.__name__}")
if hasattr(module, "Pipeline"):
    pipeline = module.Pipeline()
    print(pipeline)
    print(pipeline.get_pipelines())
