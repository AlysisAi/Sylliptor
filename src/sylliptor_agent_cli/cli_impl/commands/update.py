from __future__ import annotations

import json
from typing import Any

import typer

from ... import __version__
from ...config import AppConfig, ConfigError, load_config
from ...updates import (
    InstallerPlan,
    UpdateStatus,
    check_for_updates,
    detect_installer_plan,
    maybe_refresh_update_cache_in_background,
    passive_update_notice,
    resolve_update_check_enabled,
    run_installer_plan,
    status_from_cache,
)
from . import _patchable
from ._shared import _console

update_app = typer.Typer(add_completion=False, help="Check for and apply Sylliptor updates.")
_BACKGROUND_UPDATE_SUBCOMMANDS = {"chat", "run", "forge"}


def _cached_update_notice() -> str | None:
    try:
        return passive_update_notice(
            current_version=__version__, cfg=_patchable("load_config", load_config)()
        )
    except Exception:  # noqa: BLE001 - update notices must never block startup/status rendering
        return None


def _start_background_update_check() -> None:
    try:
        maybe_refresh_update_cache_in_background(
            current_version=__version__,
            cfg=_patchable("load_config", load_config)(),
        )
    except Exception:  # noqa: BLE001 - update checks must never block normal CLI startup
        return


def _print_update_status(console: Any, status: UpdateStatus) -> None:
    if status.source == "disabled":
        console.print("Update checks are disabled in config.")
        return
    if status.update_available and status.latest_version:
        console.print(
            f"[yellow]Sylliptor {status.latest_version} is available.[/yellow] "
            f"You have {status.current_version}."
        )
        if status.url:
            console.print(f"[dim]Release:[/dim] {status.url}")
        console.print("[dim]Run `sylliptor update` to apply it with confirmation.[/dim]")
        return
    if status.up_to_date:
        console.print(f"Sylliptor is up to date ({status.current_version}).")
        return
    if status.error:
        console.print(f"[red]Update check failed:[/red] {status.error}")
        return
    console.print("No update check has been recorded yet. Run `sylliptor update check`.")


def _update_status_or_exit(
    *,
    console: Any,
    cfg: AppConfig,
    cached: bool,
) -> UpdateStatus:
    if cached:
        status = _patchable("status_from_cache", status_from_cache)(
            current_version=__version__,
            cfg=cfg,
        )
    else:
        status = _patchable("check_for_updates", check_for_updates)(
            current_version=__version__,
            cfg=cfg,
            force=True,
        )
    if status.error:
        _print_update_status(console, status)
        raise typer.Exit(code=1)
    return status


def _print_installer_plan(console: Any, plan: InstallerPlan) -> None:
    console.print(f"[dim]Installer:[/dim] {plan.method}")
    if plan.reason:
        console.print(f"[dim]Reason:[/dim] {plan.reason}")
    if plan.display_command:
        console.print(f"[dim]Command:[/dim] {plan.display_command}")


def _run_update_flow(*, yes: bool, dry_run: bool, cached: bool) -> None:
    console = _console()
    cfg = _patchable("load_config", load_config)()
    try:
        status = _update_status_or_exit(console=console, cfg=cfg, cached=cached)
    except ConfigError as exc:
        console.print(f"[red]Update config error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    _print_update_status(console, status)
    if not status.update_available:
        return

    plan = _patchable("detect_installer_plan", detect_installer_plan)()
    _print_installer_plan(console, plan)
    if not plan.supported:
        console.print(
            "[yellow]Automatic command selection is not available for this install.[/yellow]"
        )
        console.print("Update manually using the installer that created this environment.")
        raise typer.Exit(code=1)
    if dry_run:
        console.print("[dim]Dry run only; no command executed.[/dim]")
        return
    if not yes and not typer.confirm("Run this update command now?", default=False):
        console.print("Update cancelled.")
        return
    exit_code = _patchable("run_installer_plan", run_installer_plan)(plan)
    if exit_code != 0:
        console.print(f"[red]Update command failed with exit code {exit_code}.[/red]")
        raise typer.Exit(code=exit_code)
    console.print("[green]Update command completed.[/green] Restart Sylliptor to use it.")


@update_app.callback(invoke_without_command=True)
def update_main(
    ctx: typer.Context,
    yes: bool = typer.Option(False, "--yes", "-y", help="Run the detected update command."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show the detected update command without running it.",
    ),
    cached: bool = typer.Option(
        False,
        "--cached",
        help="Use cached version information instead of checking PyPI first.",
    ),
) -> None:
    """Check for a newer Sylliptor release and apply it only after confirmation."""
    if ctx.invoked_subcommand is not None:
        return
    _run_update_flow(yes=yes, dry_run=dry_run, cached=cached)


@update_app.command("check")
def update_check(
    cached: bool = typer.Option(
        False,
        "--cached",
        help="Use cached version information instead of checking PyPI.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Check whether a newer Sylliptor package is available."""
    console = _console()
    cfg = _patchable("load_config", load_config)()
    try:
        if cached:
            status = _patchable("status_from_cache", status_from_cache)(
                current_version=__version__,
                cfg=cfg,
            )
        else:
            status = _patchable("check_for_updates", check_for_updates)(
                current_version=__version__,
                cfg=cfg,
                force=True,
            )
    except ConfigError as exc:
        console.print(f"[red]Update config error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    if as_json:
        console.print_json(json.dumps(status.to_json()))
        if status.error:
            raise typer.Exit(code=1)
        return
    _print_update_status(console, status)
    if status.error:
        raise typer.Exit(code=1)


def _update_status_summary(status: UpdateStatus, cfg: AppConfig | None) -> str:
    if not resolve_update_check_enabled(cfg):
        return "disabled"
    if status.update_available and status.latest_version:
        return f"available {status.latest_version} (run: sylliptor update)"
    if status.up_to_date and status.latest_version:
        return f"current ({status.current_version})"
    if status.error:
        return f"last check failed ({status.error})"
    return "not checked (run: sylliptor update check)"


def _cached_update_status_summary(cfg: AppConfig | None) -> str:
    try:
        return _update_status_summary(
            _patchable("status_from_cache", status_from_cache)(
                current_version=__version__,
                cfg=cfg,
            ),
            cfg,
        )
    except ConfigError as exc:
        return f"config error ({exc})"
    except Exception:
        return "unknown"
