from __future__ import annotations

import importlib
from typing import Any

_PUBLIC_FACADE_NAMES = {
    "_copy_loop_globals_to_public",
    "_apply_chat_effective_mode",
    "_handle_chat_command",
    "_handle_chat_command_impl",
    "_handle_forge_chat_command",
    "_handle_forge_chat_command_impl",
    "_run_plan_mode_approval_loop",
    "_run_plan_mode_approval_loop_impl",
    "_sync_cli_globals",
    "_sync_loop_globals_from_public",
    "chat_impl",
    "run_impl",
}


def _loop_module() -> Any:
    return importlib.import_module(f"{__name__}.loop")


def _commands_module() -> Any:
    return importlib.import_module(f"{__name__}.commands")


def _copy_loop_globals_to_public(*, overwrite: bool = False) -> None:
    module_globals = globals()
    for name, value in _loop_module().__dict__.items():
        if name.startswith("__") or name in _PUBLIC_FACADE_NAMES:
            continue
        if overwrite or name not in module_globals:
            module_globals[name] = value


def _sync_loop_globals_from_public() -> None:
    loop_globals = _loop_module().__dict__
    for name, value in globals().items():
        if name.startswith("__") or name in _PUBLIC_FACADE_NAMES:
            continue
        loop_globals[name] = value


def _sync_cli_globals(cli_mod: Any) -> None:
    loop = _loop_module()
    loop._sync_cli_globals(cli_mod)
    _copy_loop_globals_to_public(overwrite=True)


def _handle_chat_command(*args: Any, **kwargs: Any) -> Any:
    _commands_module()._sync_command_globals(globals())
    return _commands_module()._handle_chat_command(*args, **kwargs)


def _handle_forge_chat_command(*args: Any, **kwargs: Any) -> Any:
    _commands_module()._sync_command_globals(globals())
    return _commands_module()._handle_forge_chat_command(*args, **kwargs)


def _run_plan_mode_approval_loop(*args: Any, **kwargs: Any) -> Any:
    _sync_loop_globals_from_public()
    return _loop_module()._run_plan_mode_approval_loop(*args, **kwargs)


def _apply_chat_effective_mode(*args: Any, **kwargs: Any) -> Any:
    _sync_loop_globals_from_public()
    return _loop_module()._apply_chat_effective_mode(*args, **kwargs)


def _handle_chat_command_impl(cli_mod: Any, *args: Any, **kwargs: Any) -> Any:
    _sync_cli_globals(cli_mod)
    return _handle_chat_command(*args, **kwargs)


def _handle_forge_chat_command_impl(cli_mod: Any, *args: Any, **kwargs: Any) -> Any:
    _sync_cli_globals(cli_mod)
    return _handle_forge_chat_command(*args, **kwargs)


def _run_plan_mode_approval_loop_impl(cli_mod: Any, *args: Any, **kwargs: Any) -> Any:
    _sync_cli_globals(cli_mod)
    return _run_plan_mode_approval_loop(*args, **kwargs)


def _print_chat_context_impl(cli_mod: Any, *args: Any, **kwargs: Any) -> Any:
    _sync_cli_globals(cli_mod)
    _sync_loop_globals_from_public()
    return _loop_module()._print_chat_context(*args, **kwargs)


def chat_impl(cli_mod: Any, *args: Any, **kwargs: Any) -> Any:
    _sync_cli_globals(cli_mod)
    _sync_loop_globals_from_public()
    return _loop_module().chat(*args, **kwargs)


def run_impl(cli_mod: Any, *args: Any, **kwargs: Any) -> Any:
    _sync_cli_globals(cli_mod)
    _sync_loop_globals_from_public()
    return _loop_module().run(*args, **kwargs)


def __getattr__(name: str) -> Any:
    _copy_loop_globals_to_public()
    if name in globals():
        return globals()[name]
    value = getattr(_loop_module(), name)
    globals()[name] = value
    return value


def _exported_names() -> set[str]:
    _copy_loop_globals_to_public()
    return {name for name in globals() if not name.startswith("__") or name == "__version__"}


def __dir__() -> list[str]:
    return sorted(_exported_names())


__all__ = tuple(sorted(_exported_names()))
