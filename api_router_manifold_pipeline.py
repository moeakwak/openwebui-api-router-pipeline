import json
import re
from typing import Literal, Optional, Union, Generator, Iterator
import os
from pydantic import BaseModel
import requests
from sqlalchemy import Engine, create_engine
from sqlmodel import Field, SQLModel, Session, select
from datetime import datetime
import yaml
from pathlib import Path

# Schemas


class Model(BaseModel):
    provider: str
    code: str
    human_name: str
    prompt_price: float  # $ per 1M tokens
    completion_price: float  # $ per 1M tokens


class Provider(BaseModel):
    key: str
    format: Optional[Literal["openai"]] = Field(default="openai")
    url: str
    api_key: str
    models: list[Model] | Literal["auto"] = Field(default="auto")  # "auto" means get models from /models endpoint


class ModelsConfig(BaseModel):
    providers: list[Provider]


class OpenAICompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# DB


class User(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    openwebui_id: str = Field(index=True, unique=True)
    email: Optional[str] = Field(index=True, unique=True, default=None)
    name: Optional[str] = Field(default=None)
    balance: float = Field(default=0, description="The balance of the user in USD. May be negative if the user has an outstanding debt.")
    role: str = Field(default="user")
    created_at: datetime = Field(default_factory=datetime.now)


class UsageLog(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    price: float = Field(description="Computed price of the usage in USD.")
    cost: float = Field(description="Actual cost of the usage in USD. 0 if not applicable.")
    content: Optional[str] = Field(default=None, description="The content of the message. Only applicable if RECORD_CONTENT is true.")
    created_at: datetime = Field(default_factory=datetime.now)


class Pipeline:
    class Valves(BaseModel):
        MODELS_CONFIG_YAML_PATH: str = "/app/pipelines/api_router.yaml"
        DATABASE_URL: str = "sqlite:////app/pipelines/api_router.db"
        ENABLE_BILLING: bool = True
        RECORD_CONTENT: int = Field(default=30, description="Record the first N characters of the content. Set to 0 to disable recording content.")
        DEFAULT_USER_BALANCE: float = Field(default=10, description="Default balance of the user in USD.")
        DISPLAY_COST_AFTER_MESSAGE: bool = Field(default=True, description="If true, display the cost of the usage after the message.")

    def __init__(self):
        self.type = "manifold"
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        self.id = "api_router"
        self.name = "api: "

        self.valves = self.Valves(
            **{
                "MODELS_CONFIG_YAML_PATH": os.getenv("MODELS_CONFIG_YAML_PATH", "/app/pipelines/api_router.yaml"),
                "ENABLE_BILLING": True if os.getenv("ENABLE_BILLING", "true").lower() == "true" else False,
                "DATABASE_URL": os.getenv("DATABASE_URL", "sqlite:////app/pipelines/api_router.db"),
                "RECORD_CONTENT": int(os.getenv("RECORD_CONTENT", 30)),
                "DEFAULT_USER_BALANCE": float(os.getenv("DEFAULT_USER_BALANCE", 10)),
                "DISPLAY_COST_AFTER_MESSAGE": True if os.getenv("DISPLAY_COST_AFTER_MESSAGE", "true").lower() == "true" else False,
            }
        )
        self.config = self.load_config()
        self.models: dict[str, Model] = self.load_models()
        self.pipelines = self.get_pipeline_names()
        self.engine = self.setup_db()

    def setup_db(self) -> Engine:
        engine = create_engine(self.valves.DATABASE_URL)
        SQLModel.metadata.create_all(engine)
        return engine

    # async def on_startup(self):
    #     # This function is called when the server is started.
    #     print(f"on_startup:{__name__}")
    #     pass

    # async def on_shutdown(self):
    #     # This function is called when the server is stopped.
    #     print(f"on_shutdown:{__name__}")
    #     pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        print(f"on_valves_updated:{__name__}")
        self.config = self.load_config()
        self.models = self.load_models()
        self.pipelines = self.get_pipeline_names()
        self.engine = self.setup_db()

    def load_config(self):
        try:
            path = Path(self.valves.MODELS_CONFIG_YAML_PATH)
            print(f"Loading config from {path.absolute()}")
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")

            with open(path, "r") as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            return ModelsConfig.model_validate(data)
        except Exception as e:
            print(f"Error loading config: {e}")
            return ModelsConfig(providers=[])

    def load_models(self) -> dict[str, Model]:
        models = {}

        if not self.config.providers:
            print("No providers found in config.yaml")
            return models

        for provider in self.config.providers:
            if isinstance(provider.models, list):
                models.update({f"{provider.key}/{model.code}": model for model in provider.models})
            elif provider.models == "auto":
                try:
                    new_models = {}
                    headers = {}
                    headers["Authorization"] = f"Bearer {provider.api_key}"
                    headers["Content-Type"] = "application/json"

                    r = requests.get(f"{provider.url}/models", headers=headers)

                    new_models.update(
                        {
                            f"{provider.key}/{model['id']}": Model(
                                provider=provider.key,
                                code=model["id"],
                                human_name=model["name"] if "name" in model else model["id"],
                                prompt_price=model["prompt_price"] if "prompt_price" in model else 0,
                                completion_price=model["completion_price"] if "completion_price" in model else 0,
                            )
                            for model in r.json()["data"]
                            if model["id"].startswith(("gpt", "o1", "claude", "gemini", "llama", "mixtral"))
                        }
                    )
                    print(f"Loaded {len(new_models)} models for provider {provider.key}")
                    models.update(new_models)

                except requests.exceptions.RequestException as e:
                    print(f"Could not fetch models for provider {provider.key}. Status code: {r.status_code}, Error: {e}, Response: {r.text}")

                except Exception as e:
                    print(f"Could not fetch models for provider {provider.key}. Error: {e}")

        return models

    def get_pipeline_names(self) -> list[str]:
        return list(self.models.keys())

    def get_model_and_provider_by_id(self, model_id: str) -> Optional[tuple[Model, Provider]]:
        if not model_id:
            return None

        model = self.models.get(model_id)
        if not model:
            return None

        provider = next((p for p in self.config.providers if p.key == model.provider), None)
        if not provider:
            return None

        return model, provider

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")

        body["user_info"] = user  # name, id, email, role

        return body

    def remove_usage_cost_in_messages(self, messages: list[dict]) -> list[dict]:
        for message in messages:
            if "content" in message:
                message["content"] = re.sub(r'\n*<span class="usage-cost-tip-ignore-this"[^>]*>.*?</span>', "", message["content"])
        return messages

    def generate_usage_cost_message(self, price: float, cost: float, user: User) -> str:
        return f'\n\n<span class="usage-cost-tip-ignore-this" style="font-size: 12px; color: gray;">Price of this message: ${price:.6f}, cost: ${cost:.6f}. Remaining balance: ${user.balance - cost:.6f}</span>'

    def pipe(self, user_message: str, model_id: str, messages: list[dict], body: dict) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        print(messages)
        print(user_message)

        model, provider = self.get_model_and_provider_by_id(model_id)
        if not model:
            raise ValueError(f"Model {body.get('model')} not found")

        user_info = body.get("user_info")
        if not user_info:
            raise ValueError("User info not found")

        # create user if not exist
        email = user_info.get("email")
        if not email:
            raise ValueError("Email not found")

        with Session(self.engine) as session:
            user = session.exec(select(User).where(User.email == email)).first()
            if not user:
                user = User(email=email, openwebui_id=user_info.get("id"), role=user_info.get("role", "user"))
                session.add(user)
                session.commit()

        headers = {}
        headers["Authorization"] = f"Bearer {provider.api_key}"
        headers["Content-Type"] = "application/json"

        payload = {**body, "messages": self.remove_usage_cost_in_messages(messages), "model": model.code, "stream_options": {"include_usage": True}}

        if "user" in payload:
            del payload["user"]
        if "user_info" in payload:
            del payload["user_info"]
        if "chat_id" in payload:
            del payload["chat_id"]
        if "title" in payload:
            del payload["title"]

        print(payload)

        try:
            r = requests.post(
                url=f"{provider.url}/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()

            if body["stream"]:
                content = ""
                usage = None
                price = None
                cost = None
                last_chunk: dict | None = None
                stop_chunk: dict | None = None

                for line in r.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8").strip("data: ")

                    chunk = json.loads(line)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        content_delta = chunk["choices"][0].get("delta", {}).get("content", "")
                        content += content_delta
                        # prevent displaying stop chunk
                        if chunk.get("choices", [{}])[0].get("finish_reason") == "stop":
                            stop_chunk = chunk
                        else:
                            last_chunk = chunk
                            yield "data: " + line + "\n\n"

                    elif "usage" in chunk:
                        usage = OpenAICompletionUsage(**chunk["usage"])
                        price = (usage.prompt_tokens * model.prompt_price + usage.completion_tokens * model.completion_price) / 1e6
                        cost = price if user.role != "admin" else 0
                        if self.valves.DISPLAY_COST_AFTER_MESSAGE:
                            if last_chunk:
                                new_chunk = last_chunk.copy()
                                new_chunk["choices"][0]["delta"]["content"] = self.generate_usage_cost_message(price, cost, user)
                                yield "data: " + json.dumps(new_chunk) + "\n\n"
                            else:
                                print("Error displaying usage cost: last_chunk is None")
                            if stop_chunk:
                                yield "data: " + json.dumps(stop_chunk) + "\n\n"
                                stop_chunk = None
                        yield "data: " + line + "\n\n"

                if stop_chunk:
                    yield "data: " + json.dumps(stop_chunk) + "\n\n"

                self.add_usage_log(user.id, model.code, usage, price, cost, content)

            else:
                response = r.json()
                usage = OpenAICompletionUsage(**response["usage"])
                content = response["choices"][0]["message"]["content"]

                price = (usage.prompt_tokens * model.prompt_price + usage.completion_tokens * model.completion_price) / 1e6
                cost = price if user.role != "admin" else 0
                self.add_usage_log(user.id, model.code, usage, price, cost, content)

                if self.valves.DISPLAY_COST_AFTER_MESSAGE:
                    response["choices"][0]["message"]["content"] += self.generate_usage_cost_message(price, cost, user)

                return response

        except Exception as e:
            raise ValueError(f"Error: {e}")

    def add_usage_log(self, user_id: int, model: str, usage: OpenAICompletionUsage, price: float, cost: float, content: str):
        with Session(self.engine) as session:
            usage_log = UsageLog(
                user_id=user_id,
                model=model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                price=price,
                cost=cost,
                content=(
                    content[: self.valves.RECORD_CONTENT]
                    if isinstance(self.valves.RECORD_CONTENT, int)
                    else (content if self.valves.RECORD_CONTENT else None)
                ),
            )
            session.add(usage_log)
            session.commit()

    def handle_bot_model(self, message: str, user: User):
        pass
