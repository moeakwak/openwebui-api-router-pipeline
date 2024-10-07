"""
title: API Router Manifold Pipeline
author: Moeakwak
date: 2024-10-12
version: 0.1.0
license: MIT
description: A pipeline for routing OpenAI models, track user usages, etc.
requirements: sqlmodel, sqlalchemy, requests, pathlib, tabulate
"""

import json
import re
from typing import Literal, Optional, Union, Generator, Iterator
import os
from pydantic import BaseModel
import requests
from sqlalchemy import Engine, create_engine
from sqlmodel import Field, SQLModel, Session, select
from datetime import datetime, timedelta
from tabulate import tabulate
import yaml
from pathlib import Path
from sqlalchemy import func
from typing import Optional

# Schemas


class Model(BaseModel):
    provider: str
    code: str
    human_name: Optional[str] = Field(default=None)
    prompt_price: float  # $ per 1M tokens
    completion_price: float  # $ per 1M tokens
    no_system_prompt: Optional[bool] = Field(default=False, description="If true, remove the system prompt. Useful for o1 models.")
    no_stream: Optional[bool] = Field(default=False, description="If true, do not stream the response. Useful for o1 models.")


class Provider(BaseModel):
    key: str
    format: Optional[Literal["openai"]] = Field(default="openai")
    url: str
    api_key: str
    price_ratio: Optional[float] = Field(
        default=1,
        description="The price ratio of the provider. For example, if the price ratio is 0.5, then the actual cost of the provider is half of the original cost.",
    )


class Config(BaseModel):
    providers: list[Provider]
    models: list[Model]


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
    updated_at: datetime = Field(default_factory=datetime.now)


class UsageLog(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float = Field(description="Computed cost of the usage.")
    actual_cost: float = Field(description="Actual cost of the usage, after applying the price ratio of the provider.")
    content: Optional[str] = Field(default=None, description="The content of the message. Only applicable if RECORD_CONTENT is true.")
    created_at: datetime = Field(default_factory=datetime.now)


def escape_pipeline_id(pipeline_id: str) -> str:
    return pipeline_id.replace("/", "_")


class Pipeline:
    class Valves(BaseModel):
        MODELS_CONFIG_YAML_PATH: str = "/app/pipelines/api_router.yaml"
        DATABASE_URL: str = "sqlite:////app/pipelines/api_router.db"
        ENABLE_BILLING: bool = True
        RECORD_CONTENT: int = Field(
            default=30, description="Record the first N characters of the content. Set to 0 to disable recording content, -1 to record all content."
        )
        DEFAULT_USER_BALANCE: float = Field(default=10, description="Default balance of the user in USD.")
        DISPLAY_COST_AFTER_MESSAGE: bool = Field(default=True, description="If true, display the cost of the usage after the message.")
        BASE_COST_CURRENCY_UNIT: str = Field(default="$", description="The currency unit of the base cost.")
        ACTUAL_COST_CURRENCY_UNIT: str = Field(
            default="$", description="The currency unit of the actual cost, also the currency unit of the user balance."
        )

    def __init__(self):
        self.type = "manifold"
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        self.id = "api_router"
        self.name = "💬 "

        self.valves = self.Valves(
            **{
                "MODELS_CONFIG_YAML_PATH": os.getenv("MODELS_CONFIG_YAML_PATH", "/app/pipelines/api_router.yaml"),
                "ENABLE_BILLING": True if os.getenv("ENABLE_BILLING", "true").lower() == "true" else False,
                "DATABASE_URL": os.getenv("DATABASE_URL", "sqlite:////app/pipelines/api_router.db"),
                "RECORD_CONTENT": int(os.getenv("RECORD_CONTENT", 30)),
                "DEFAULT_USER_BALANCE": float(os.getenv("DEFAULT_USER_BALANCE", 10)),
                "DISPLAY_COST_AFTER_MESSAGE": True if os.getenv("DISPLAY_COST_AFTER_MESSAGE", "true").lower() == "true" else False,
                "BASE_COST_CURRENCY_UNIT": os.getenv("BASE_COST_CURRENCY_UNIT", "$"),
                "ACTUAL_COST_CURRENCY_UNIT": os.getenv("ACTUAL_COST_CURRENCY_UNIT", "$"),
            }
        )
        self.config = self.load_config()
        self.models: dict[str, Model] = self.load_models()
        self.pipelines = self.get_pipelines()
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
        # print(f"on_valves_updated:{__name__}")
        self.config = self.load_config()
        self.models = self.load_models()
        self.pipelines = self.get_pipelines()
        self.engine = self.setup_db()

    def load_config(self):
        try:
            path = Path(self.valves.MODELS_CONFIG_YAML_PATH)
            print(f"Loading config from {path.absolute()}")
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")

            with open(path, "r") as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            return Config.model_validate(data)
        except Exception as e:
            print(f"Error loading config: {e}")
            return Config(providers=[])

    def load_models(self) -> dict[str, Model]:
        models = {}
        provider_keys = {p.key for p in self.config.providers}

        if not self.config.models:
            print("No models found in config.yaml")
            return models

        for model in self.config.models:
            if model.provider not in provider_keys:
                print(f"Provider {model.provider} not found in config.yaml")
                continue
            models[escape_pipeline_id(f"{model.provider}/{model.code}")] = model

        return models

    def get_pipelines(self) -> list[dict]:
        pipelines = [{"id": model_id, "name": model.human_name or model.code} for model_id, model in self.models.items()]
        pipelines.append({"id": "service_bot", "name": "AAA 🤖 Service Bot"})
        return pipelines

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
        # print(f"inlet:{__name__}")

        if not "email" in user or not "id" in user or not "role" in user:
            raise Exception("User info not found")

        with Session(self.engine) as session:
            db_user = session.exec(select(User).where(User.email == user["email"])).first()
            if not db_user:
                db_user = User(email=user["email"], openwebui_id=user["id"], role=user["role"], balance=self.valves.DEFAULT_USER_BALANCE)
                session.add(db_user)
                session.commit()
            if self.valves.ENABLE_BILLING and db_user.balance <= 0 and db_user.role != "admin":
                raise Exception("Your account has insufficient balance. Please top up your account.")

        body["user_info"] = db_user.model_dump()  # name, id, email, role

        return body

    def remove_usage_cost_in_messages(self, messages: list[dict]) -> list[dict]:
        for message in messages:
            if "content" in message:
                message["content"] = re.sub(r"\n*\*\(📊[^)]*\)\*$", "", message["content"], flags=re.MULTILINE)
        return messages

    def generate_usage_cost_message(self, usage: OpenAICompletionUsage, base_cost: float, actual_cost: float, user: User) -> str:
        return f"\n\n*(📊 Cost {self.valves.BASE_COST_CURRENCY_UNIT}{base_cost:.6f} | Actual Cost {self.valves.ACTUAL_COST_CURRENCY_UNIT}{actual_cost:.6f})*"

    def compute_price(self, model: Model, provider: Provider, usage: OpenAICompletionUsage) -> tuple[float, float]:
        base_cost = (usage.prompt_tokens * model.prompt_price + usage.completion_tokens * model.completion_price) / 1e6
        actual_cost = base_cost * provider.price_ratio
        return base_cost, actual_cost

    def pipe(self, user_message: str, model_id: str, messages: list[dict], body: dict) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        # print(f"pipe:{__name__}")

        if model_id == "service_bot":
            return self.handle_bot(user_message, body["user_info"])

        model, provider = self.get_model_and_provider_by_id(model_id)
        if not model:
            raise Exception(f"Model {body.get('model')} not found")

        user_info = body.get("user_info")
        if not user_info:
            raise Exception("User info not found")

        user = User(**user_info)

        headers = {}
        headers["Authorization"] = f"Bearer {provider.api_key}"
        headers["Content-Type"] = "application/json"

        payload = {**body, "messages": self.remove_usage_cost_in_messages(messages), "model": model.code, "stream_options": {"include_usage": True}}

        fake_stream = False

        if model.no_system_prompt:
            payload["messages"] = [message for message in messages if message["role"] != "system"]

        if model.no_stream and body["stream"]:
            fake_stream = True
            payload["stream"] = False

        if "user" in payload:
            del payload["user"]
        if "user_info" in payload:
            del payload["user_info"]
        if "chat_id" in payload:
            del payload["chat_id"]
        if "title" in payload:
            del payload["title"]

        # print("payload: ", payload)

        try:
            r = requests.post(
                url=f"{provider.url}/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()

            if not fake_stream and body["stream"]:
                return self.stream_response(r, model, provider, user)
            elif fake_stream:
                return self.fake_stream_response(r, model, provider, user)
            else:
                return self.non_stream_response(r, model, provider, user)

        except Exception as e:
            raise e

    def stream_response(self, r, model, provider, user):
        def generate():
            content = ""
            usage = None
            last_chunk: dict | None = None
            stop_chunk: dict | None = None

            for line in r.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8").strip("data: ")

                if line == "[DONE]":
                    yield "data: [DONE]\n\n"
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {line}")
                    continue

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
                    base_cost, actual_cost = self.compute_price(model, provider, usage)
                    if self.valves.DISPLAY_COST_AFTER_MESSAGE:
                        if last_chunk:
                            new_chunk = last_chunk.copy()
                            new_chunk["choices"][0]["delta"]["content"] = self.generate_usage_cost_message(usage, base_cost, actual_cost, user)
                            yield "data: " + json.dumps(new_chunk) + "\n\n"
                        else:
                            print("Error displaying usage cost: last_chunk is None")
                        if stop_chunk:
                            yield "data: " + json.dumps(stop_chunk) + "\n\n"
                            stop_chunk = None
                    yield "data: " + line + "\n\n"

            if stop_chunk:
                yield "data: " + json.dumps(stop_chunk) + "\n\n"

            self.add_usage_log(user.id, model.code, usage, base_cost, actual_cost, content)

        return generate()

    def fake_stream_response(self, r, model, provider, user):
        def generate():
            response = r.json()
            usage = OpenAICompletionUsage(**response["usage"])
            content = response["choices"][0]["message"]["content"]
            logprobs = response["choices"][0].get("logprobs", None)
            finish_reason = response["choices"][0].get("finish_reason", None)

            base_cost, actual_cost = self.compute_price(model, provider, usage)
            self.add_usage_log(user.id, model.code, usage, base_cost, actual_cost, content)

            if self.valves.DISPLAY_COST_AFTER_MESSAGE:
                content += self.generate_usage_cost_message(usage, base_cost, actual_cost, user)

            chunk = {
                **response,
                "usage": None,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": content}, "logprobs": None, "finish_reason": None}],
            }
            stop_chunk = {
                **chunk,
                "choices": [{"index": 0, "delta": {}, "logprobs": logprobs, "finish_reason": finish_reason}],
            }
            usage_chunk = {
                **chunk,
                "choices": None,
                "usage": usage.model_dump(),
            }

            yield "data: " + json.dumps(chunk) + "\n\n"
            yield "data: " + json.dumps(stop_chunk) + "\n\n"
            yield "data: " + json.dumps(usage_chunk) + "\n\n"
            yield "data: [DONE]"

        return generate()

    def non_stream_response(self, r, model, provider, user):
        response = r.json()
        usage = OpenAICompletionUsage(**response["usage"])
        content = response["choices"][0]["message"]["content"]

        base_cost, actual_cost = self.compute_price(model, provider, usage)
        self.add_usage_log(user.id, model.code, usage, base_cost, actual_cost, content)

        return response

    def add_usage_log(self, user_id: int, model: str, usage: OpenAICompletionUsage, base_cost: float, actual_cost: float, content: str):
        with Session(self.engine) as session:
            try:
                with session.begin():
                    user = session.exec(select(User).where(User.id == user_id)).first()
                    if not user:
                        raise Exception("User not found")

                    if self.valves.RECORD_CONTENT > 0:
                        short_content = content[: self.valves.RECORD_CONTENT]
                        if len(content) > len(short_content):
                            short_content += "..."
                        content = short_content
                    elif self.valves.RECORD_CONTENT == 0:
                        content = None

                    usage_log = UsageLog(
                        user_id=user.id,
                        model=model,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        total_tokens=usage.total_tokens,
                        cost=base_cost,
                        actual_cost=actual_cost,
                        content=content,
                    )
                    user.balance -= actual_cost
                    user.updated_at = datetime.now()
                    session.add(usage_log)
                    session.commit()
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                session.rollback()
                raise

    def handle_bot(self, message: str, user: dict):
        bot = ServiceBot(self)
        return bot.handle_command(message, user)


class ServiceBot:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.commands = {
            "help": self.help,
            "info": self.info,
            "me": self.me,
            "topup": self.topup,
            "set_balance": self.set_balance,
            "users": self.users,
            "stats": self.stats,
            "gstats": self.global_stats,
        }

    def parse_command(self, message: str) -> tuple[str, list[str]]:
        parts = message.strip().split()
        command = parts[0].lower()
        args = parts[1:]
        return command, args

    def handle_command(self, message: str, user: dict) -> str:
        command, args = self.parse_command(message)

        if command not in self.commands:
            return "Unknown command. Type 'help' for a list of available commands."

        if command in ["topup", "set_balance", "users", "gstats"] and user["role"] != "admin":
            return "You don't have permission to use this command."

        return self.commands[command](args, user)

    def help(self, args: list[str], user: dict) -> str:
        help_text = """
Available commands:
- **help**: Show this help message
- **info**: List all model information
- **me**: Show your user information
- **stats** [d/w/m]: Show your usage statistics (d: daily, w: weekly, m: monthly, default: daily)
"""
        if user["role"] == "admin":
            help_text += """

Admin commands:
- **topup** [user id/email] [amount]: Top up a user's balance
- **set_balance** [user id/email] [amount]: Set a user's balance
- **users**: List all users
- **gstats** [d/w/m]: Show global usage statistics (d: daily, w: weekly, m: monthly, default: daily)
"""
        return help_text.strip()

    def info(self, args: list[str], user: dict) -> str:
        headers = ["Model", "Prompt Price", "Completion Price", "Ratio"]
        price_ratio_map = {}
        for provider in self.pipeline.config.providers:
            price_ratio_map[provider.key] = provider.price_ratio
        data = [
            [
                model.human_name or model.code,
                f"{self.pipeline.valves.BASE_COST_CURRENCY_UNIT}{model.prompt_price:.2f} / {self.pipeline.valves.ACTUAL_COST_CURRENCY_UNIT}{model.prompt_price * price_ratio_map[model.provider]:.2f}",
                f"{self.pipeline.valves.BASE_COST_CURRENCY_UNIT}{model.completion_price:.2f} / {self.pipeline.valves.ACTUAL_COST_CURRENCY_UNIT}{model.completion_price * price_ratio_map[model.provider]:.2f}",
                f"{price_ratio_map[model.provider] or 1:.2f}",
            ]
            for model in self.pipeline.models.values()
        ]
        return f"{tabulate(data, headers=headers, tablefmt='pipe', colalign=('left',))}"

    def me(self, args: list[str], user: dict) -> str:
        with Session(self.pipeline.engine) as session:
            db_user = session.exec(select(User).where(User.email == user["email"])).first()
            if not db_user:
                return "User not found."
            return f"""
Your information:
- ID: {db_user.id}
- Email: {db_user.email}
- Name: {db_user.name or 'N/A'}
- Balance: {self.pipeline.valves.ACTUAL_COST_CURRENCY_UNIT}{db_user.balance:.2f}
- Role: {db_user.role}
- Created at: {db_user.created_at}
""".strip()

    def topup(self, args: list[str], user: dict) -> str:
        if len(args) != 2:
            return "Usage: topup [user id/email] [amount]"

        user_id_or_email, amount = args
        try:
            amount = float(amount)
        except ValueError:
            return "Invalid amount. Please provide a numeric value."

        with Session(self.pipeline.engine) as session:
            db_user = session.exec(select(User).where((User.id == user_id_or_email) | (User.email == user_id_or_email))).first()
            if not db_user:
                return "User not found."
            db_user.balance += amount
            db_user.updated_at = datetime.now()  # 更新updated_at字段
            session.commit()
            return f"Successfully topped up {db_user.email}'s balance by ${amount:.2f}. New balance: ${db_user.balance:.2f}"

    def set_balance(self, args: list[str], user: dict) -> str:
        if len(args) != 2:
            return "Usage: set_balance [user id/email] [amount]"

        user_id_or_email, amount = args
        try:
            amount = float(amount)
        except ValueError:
            return "Invalid amount. Please provide a numeric value."

        with Session(self.pipeline.engine) as session:
            db_user = session.exec(select(User).where((User.id == user_id_or_email) | (User.email == user_id_or_email))).first()
            if not db_user:
                return "User not found."
            db_user.balance = amount
            db_user.updated_at = datetime.now()  # 更新updated_at字段
            session.commit()
            return f"Successfully set {db_user.email}'s balance to ${amount:.2f}"

    def users(self, args: list[str], user: dict) -> str:
        with Session(self.pipeline.engine) as session:
            users = session.exec(select(User)).all()
            headers = ["ID", "Name", "Email", "Balance", "Role", "Updated At", "Created At"]
            data = [
                [
                    u.id,
                    u.name or "N/A",
                    u.email,
                    f"{self.pipeline.valves.ACTUAL_COST_CURRENCY_UNIT}{u.balance:.3f}",
                    u.role,
                    u.updated_at,
                    u.created_at,
                ]
                for u in users
            ]
            return f"{tabulate(data, headers=headers, tablefmt='pipe', colalign=('left',))}"

    def stats(self, args: list[str], user: dict) -> str:
        period = "d" if not args else args[0]
        return self._get_stats(period, user["id"])

    def global_stats(self, args: list[str], user: dict) -> str:
        if user["role"] != "admin":
            return "You don't have permission to use this command."
        period = "d" if not args else args[0]
        return self._get_stats(period)

    def _get_stats(self, period: str, user_id: Optional[int] = None) -> str:
        time_delta = {
            "d": timedelta(days=1),
            "w": timedelta(weeks=1),
            "m": timedelta(days=30),
        }.get(period.lower(), timedelta(days=1))

        with Session(self.pipeline.engine) as session:
            query = select(
                UsageLog.model,
                func.sum(UsageLog.prompt_tokens).label("total_prompt_tokens"),
                func.sum(UsageLog.completion_tokens).label("total_completion_tokens"),
                func.sum(UsageLog.total_tokens).label("total_tokens"),
                func.sum(UsageLog.cost).label("total_cost"),
                func.sum(UsageLog.actual_cost).label("total_actual_cost"),
                func.count().label("count"),
            ).where(UsageLog.created_at >= datetime.now() - time_delta)

            if user_id:
                query = query.where(UsageLog.user_id == user_id)

            query = query.group_by(UsageLog.model)

            results = session.exec(query).all()

            if not results:
                return "No usage records found for the specified time period."

            headers = ["Model", "Prompt Tokens", "Completion Tokens", "Total Tokens", "Base Cost", "Actual Cost", "Usage Count"]
            data = [
                [
                    r.model,
                    r.total_prompt_tokens,
                    r.total_completion_tokens,
                    r.total_tokens,
                    f"{self.pipeline.valves.BASE_COST_CURRENCY_UNIT}{r.total_cost:.4f}",
                    f"{self.pipeline.valves.ACTUAL_COST_CURRENCY_UNIT}{r.total_actual_cost:.4f}",
                    r.count,
                ]
                for r in results
            ]

            if not user_id:
                unique_users = session.exec(
                    select(func.count(func.distinct(UsageLog.user_id))).where(UsageLog.created_at >= datetime.now() - time_delta)
                ).first()
                headers.append("Unique Users")
                for row in data:
                    row.append(unique_users)

            table = tabulate(data, headers=headers, tablefmt="pipe", colalign=('left',))
            period_str = {"d": "Daily", "w": "Weekly", "m": "Monthly"}.get(period.lower(), "Daily")
            return f"{period_str} Statistics:\n\n{table}"
