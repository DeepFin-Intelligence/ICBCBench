"""Generation-only backends. Judge clients and credentials are kept separate.

An explicitly selected Python adapter is trusted local code. It must implement
its own network timeouts and cancellation; the toolkit cannot interrupt an
arbitrary Python callable safely.
"""

import argparse
import hashlib
import importlib
from importlib.machinery import PathFinder
import inspect
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

from openai import OpenAI

from evaluation_toolkit.model_clients import load_project_env, resolve_openrouter_model
from evaluation_toolkit.result_import import _complete_record


def positive_timeout(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def add_generation_arguments(parser):
    parser.add_argument("--model_backend", choices=("builtin", "openai", "agent"), default="builtin",
                        help="Tested-system backend; does not change the judge backend")
    parser.add_argument("--model_base_url", help="OpenAI-compatible chat API base URL, including /v1 where required")
    parser.add_argument("--model_api_key_env", default="ICBCBENCH_MODEL_API_KEY",
                        help="Environment variable containing the tested model's key (never the key itself)")
    parser.add_argument("--model_timeout", type=positive_timeout, default=120.0,
                        help="Custom OpenAI-compatible request timeout in seconds; one retry maximum")
    parser.add_argument("--agent", help="Trusted synchronous Python adapter as module:callable")
    parser.add_argument("--attachment_mode", choices=("image", "url"), default="image",
                        help="Render referenced PDF pages, or pass only their URLs and page descriptions")


def model_backend(options):
    return getattr(options, "model_backend", "builtin")


def generation_model_name(options):
    return resolve_openrouter_model(options.model) if model_backend(options) == "builtin" else options.model


def _validated_url(value):
    # Reject credentials before returning/logging a configuration. In particular,
    # do not echo the rejected value in exceptions.
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        raise ValueError("--model_base_url must be a nonempty HTTP(S) URL without whitespace")
    try:
        parsed = urlsplit(value)
        parsed.port  # Validate malformed/out-of-range ports without echoing the URL.
    except ValueError:
        raise ValueError("--model_base_url is invalid") from None
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or "?" in value or "#" in value):
        raise ValueError("--model_base_url must use HTTP(S), with no userinfo, query, or fragment")
    try:
        local = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        local = parsed.hostname.lower() == "localhost"
    if parsed.scheme == "http" and not local:
        raise ValueError("Remote --model_base_url must use HTTPS; HTTP is allowed only for localhost/loopback")
    return value.rstrip("/"), local


def _agent_source_identity(specification):
    """Locate source without importing/executing an adapter or its packages."""
    if not isinstance(specification, str) or not re.fullmatch(
        r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", specification
    ):
        raise ValueError("--agent must be module:callable, for example examples.custom_agent:generate")
    module, attribute = specification.split(":")
    search_path = sys.path
    parts = module.split(".")
    sources = []
    for index in range(len(parts)):
        name = ".".join(parts[:index + 1])
        spec = PathFinder.find_spec(name, search_path)
        if spec is None:
            raise ValueError(f"Agent module {module!r} cannot be found on the Python import path")
        if spec.origin and Path(spec.origin).is_file():
            sources.append({"module": name, "sha256": hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()})
        search_path = spec.submodule_search_locations
        if index < len(parts) - 1 and search_path is None:
            raise ValueError(f"Agent module {name!r} is not a package")
    return {"entrypoint": specification, "source_files": sources}


def generation_config(options):
    """Return a validated, JSON-serializable identity containing no API keys."""
    backend = model_backend(options)
    if backend not in {"builtin", "openai", "agent"}:
        raise ValueError("Unknown model backend")
    mode = getattr(options, "attachment_mode", "image")
    if mode not in {"image", "url"}:
        raise ValueError("attachment_mode must be image or url")
    environment = getattr(options, "model_api_key_env", "ICBCBENCH_MODEL_API_KEY")
    if not isinstance(environment, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", environment):
        raise ValueError("--model_api_key_env must name an environment variable, not contain a key")
    try:
        timeout = positive_timeout(getattr(options, "model_timeout", 120.0))
    except (ValueError, TypeError, argparse.ArgumentTypeError):
        raise ValueError("--model_timeout must be a finite positive number") from None
    config = {"model_backend": backend, "model": generation_model_name(options), "attachment_mode": mode,
              "model_base_url": None, "model_api_key_env": environment, "model_timeout": timeout, "agent": None}
    base_url = getattr(options, "model_base_url", None)
    adapter = getattr(options, "agent", None)
    if backend != "openai" and base_url is not None:
        raise ValueError("--model_base_url requires --model_backend openai")
    if backend != "agent" and adapter is not None:
        raise ValueError("--agent requires --model_backend agent")
    if backend == "openai":
        url, _ = _validated_url(base_url)
        config.update(model_base_url=url, max_retries=1)
    elif backend == "agent":
        identity = _agent_source_identity(adapter)
        config.update(agent=adapter, agent_source_files=identity["source_files"])
    return config


def validate_generation_credentials(options):
    """Validate custom configuration without initializing clients or adapters.

    Built-in provider credentials remain the existing runner's responsibility.
    A local OpenAI-compatible server may omit authentication.
    """
    config = generation_config(options)
    load_project_env()
    if config["model_backend"] == "openai":
        _, local = _validated_url(config["model_base_url"])
        if not local and not os.getenv(config["model_api_key_env"], "").strip():
            raise ValueError(f"Missing {config['model_api_key_env']} for the custom remote model endpoint")
    return config


def _usage(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Generation usage must be a JSON object or null")
    for key in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens", "total_tokens"):
        count = value.get(key)
        if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
            raise ValueError(f"Generation usage {key} must be a nonnegative integer or null")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("Generation usage must contain finite JSON values") from None
    return value


def _validated_result(result):
    if not isinstance(result, dict):
        raise ValueError("Generation must return an object with nonempty string content")
    content = result.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Generation returned empty or non-string content")
    _complete_record(dict(result, response=content), "custom generation")
    citations = result.get("citations", [])
    if not isinstance(citations, list):
        raise ValueError("Generation citations must be a JSON array")
    try:
        json.dumps(citations, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError("Generation citations must contain finite JSON values") from None
    return {"content": content, "usage": _usage(result.get("usage")), "citations": citations}


def generate_custom(options, *, task_id, track, language, messages):
    """Generate using only sanitized prompt data; never pass answers or rubrics.

    Adapter contract: generate({id, track, language, model, messages}) ->
    {content: str, usage?: dict, citations?: list}. Include inline source URLs in
    content for citation scoring; citations are additionally saved as metadata.
    """
    config = validate_generation_credentials(options)
    if config["model_backend"] == "openai":
        key = os.getenv(config["model_api_key_env"], "").strip() or "icbcbench-local-no-key"
        kwargs = {"model": options.model, "messages": messages, "stream": False}
        if getattr(options, "temperature", None) is not None:
            kwargs["temperature"] = options.temperature
        if getattr(options, "max_completion_tokens", None) is not None:
            kwargs["max_tokens"] = options.max_completion_tokens
        # Do not inherit OPENROUTER_API_KEY, aliases, or provider-only extras.
        with OpenAI(api_key=key, base_url=config["model_base_url"], timeout=config["model_timeout"], max_retries=1) as client:
            response = client.chat.completions.create(**kwargs)
        if not response.choices or response.choices[0].finish_reason != "stop":
            raise ValueError("Custom model response did not finish normally (expected finish_reason=stop)")
        return _validated_result({"content": response.choices[0].message.content,
                                  "usage": response.usage.model_dump() if response.usage else None})
    if config["model_backend"] == "agent":
        module, attribute = config["agent"].split(":")
        adapter = getattr(importlib.import_module(module), attribute, None)
        if not callable(adapter):
            raise ValueError("Agent entrypoint must identify a callable")
        # Copy the allowed inputs so a local adapter cannot mutate runner prompts.
        request = json.loads(json.dumps({"id": str(task_id), "track": track, "language": language,
                                        "model": options.model, "messages": messages}, allow_nan=False))
        result = adapter(request)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise ValueError("Agent entrypoint must be synchronous; wrap async work in a synchronous function")
        return _validated_result(result)
    raise ValueError("generate_custom requires the openai or agent backend")
