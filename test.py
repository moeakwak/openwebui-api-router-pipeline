from api_router_manifold_pipeline import Pipeline
import os

os.environ["MODELS_CONFIG_YAML_PATH"] = "api_router.yaml"
os.environ["DATABASE_URL"] = "sqlite:///api_router.db"

p = Pipeline()
print(p.models)
