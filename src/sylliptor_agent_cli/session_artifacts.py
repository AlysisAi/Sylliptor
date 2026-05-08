from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


def normalize_artifact_parts(parts: tuple[str, ...]) -> list[str]:
    normalized: list[str] = []
    for raw_part in parts:
        text = str(raw_part).strip().replace("\\", "/")
        if not text:
            raise ValueError("artifact path part cannot be empty")
        candidate = PurePosixPath(text)
        if candidate.is_absolute():
            raise ValueError("artifact path part must be relative")
        for segment in candidate.parts:
            if segment in {"", "."}:
                continue
            if segment == "..":
                raise ValueError("artifact path cannot traverse upward")
            normalized.append(segment)
    if not normalized:
        raise ValueError("artifact path cannot be empty")
    return normalized


def _path_is_under_root(*, path: Path, root: Path | None) -> bool:
    if root is None:
        return False
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class ModelArtifactReference:
    locator: str
    artifact_readable_via_fs: bool
    artifact_location: str


@dataclass(frozen=True)
class SessionArtifactLayout:
    filesystem_root: Path
    locator_prefix: str = "session_artifacts"

    def artifact_fs_path(self, *parts: str) -> Path:
        return self.filesystem_root.joinpath(*normalize_artifact_parts(parts))

    def artifact_locator(self, *parts: str) -> str:
        locator_parts = normalize_artifact_parts((self.locator_prefix, *parts))
        return "/".join(locator_parts)

    def locator_for_path(self, artifact_path: Path) -> str:
        rel = artifact_path.resolve().relative_to(self.filesystem_root.resolve()).as_posix()
        return self.artifact_locator(rel)

    def model_reference_for_path(
        self,
        *,
        artifact_path: Path,
        workspace_root: Path | None,
    ) -> ModelArtifactReference:
        readable_via_fs = _path_is_under_root(path=artifact_path, root=workspace_root)
        return ModelArtifactReference(
            locator=self.locator_for_path(artifact_path),
            artifact_readable_via_fs=readable_via_fs,
            artifact_location="workspace_root" if readable_via_fs else "external_session_store",
        )

    def display_reference_for_path(
        self,
        *,
        artifact_path: Path,
        workspace_root: Path | None,
    ) -> str:
        model_reference = self.model_reference_for_path(
            artifact_path=artifact_path,
            workspace_root=workspace_root,
        )
        if model_reference.artifact_readable_via_fs and workspace_root is not None:
            return artifact_path.resolve().relative_to(workspace_root.resolve()).as_posix()
        return model_reference.locator
