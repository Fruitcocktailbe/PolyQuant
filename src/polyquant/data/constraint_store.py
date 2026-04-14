"""
ConstraintStore - Persistence layer for validated market constraints.

The ConstraintStore is the bridge between the Map Maker (offline analysis)
and the Navigator (real-time trading). It persists the validated constraint
matrices and dependencies so they can be loaded instantly at trade time.

ARCHITECTURE:
-------------
Map Maker --(writes)--> ConstraintStore --(reads)--> Navigator

WHY FILE-BASED:
---------------
1. Simplicity: No database server required.
2. Portability: Easy to version control and share.
3. Atomicity: File writes are atomic on most systems.
4. Speed: JSON parsing is <1ms for typical constraint sizes.

USAGE:
------
    # In Map Maker (write)
    store = ConstraintStore()
    await store.save_constraints(cluster_id, constraints, dependencies)
    
    # In Navigator (read)
    store = ConstraintStore()
    constraints, dependencies = await store.load_constraints(cluster_id)
"""

import asyncio  # Week 3: For parallel manifest loading
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from polyquant.utils import config, get_logger

logger = get_logger(__name__)


# Bumped from "1.0" → "1.1" when MapMaker stopped filtering constraints by
# snapshot-time liquidity/spread. v1.0 manifests are missing structurally
# valid constraints and are auto-quarantined on load.
CURRENT_MANIFEST_VERSION = "1.1"

# Manifests older than this are deleted on startup. Stale manifests reference
# markets that have likely resolved or moved, so keeping them around just
# pollutes the solver's constraint set. Re-running `polyquant map` regenerates
# whatever is still relevant.
MANIFEST_TTL_DAYS = 30


def _parse_version(version: str) -> tuple[int, ...]:
    """
    Parse a dotted version string into an integer tuple for semantic comparison.

    String comparison is lexicographic: "1.10" < "1.2" is True, which would
    incorrectly quarantine a newer manifest as legacy. Tuple comparison
    ("1.10" → (1, 10)) gives the expected ordering.
    """
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return (0,)


class StoredConstraint(BaseModel):
    """A single constraint stored in the ConstraintStore."""
    constraint_id: str
    description: str
    coefficients: dict[str, float]
    rhs: float
    confidence: float
    reasoning: str
    source_markets: list[str] = Field(default_factory=list)

    @field_validator("coefficients")
    @classmethod
    def _validate_token_id_types(cls, v: dict) -> dict:
        for tid in v.keys():
            if not isinstance(tid, str):
                raise TypeError(
                    f"Token ID must be str, got {type(tid).__name__}: {tid!r}"
                )
            if not tid:
                raise ValueError("Token ID cannot be empty")
        return v


class StoredDependency(BaseModel):
    """A single dependency stored in the ConstraintStore."""
    source_market_id: str
    source_outcome: str
    target_market_id: str
    target_outcome: str
    relationship: str  # implies, excludes, partition
    confidence: float


class ConstraintManifest(BaseModel):
    """
    The full constraint manifest for a market cluster.
    
    This is what gets persisted to disk and loaded by the Navigator.
    """
    cluster_id: str
    topic: str = ""
    market_ids: list[str] = Field(default_factory=list)
    market_exchanges: dict[str, str] = Field(default_factory=dict)  # market_id -> exchange
    market_titles: dict[str, str] = Field(default_factory=dict)  # market_id -> human-readable title
    constraints: list[StoredConstraint] = Field(default_factory=list)
    dependencies: list[StoredDependency] = Field(default_factory=list)
    correlations: list[Any] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = CURRENT_MANIFEST_VERSION
    
    @property
    def constraint_count(self) -> int:
        return len(self.constraints)
    
    @property
    def dependency_count(self) -> int:
        return len(self.dependencies)


class ConstraintStore:
    """
    Persistence layer for market constraints.
    
    Stores validated constraints as JSON files for fast loading
    by the Navigator at trade time.
    
    Storage Layout:
        .polyquant/
            constraints/
                {cluster_id}.json
                {cluster_id}.json
                ...
                _index.json  (list of all cluster IDs)
    """
    
    def __init__(self, base_path: Path | None = None):
        """
        Initialize the ConstraintStore.

        Args:
            base_path: Base directory for storage.
                       Defaults to .polyquant/constraints in project root.
        """
        if base_path is None:
            # Use project root / .polyquant / constraints
            self.base_path = Path.cwd() / ".polyquant" / "constraints"
        else:
            self.base_path = Path(base_path)

        # Ensure directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Sweep manifests older than the TTL. Cluster IDs are content-addressed,
        # so fresh runs always overwrite still-relevant clusters; anything not
        # touched in MANIFEST_TTL_DAYS references markets that have almost
        # certainly resolved and should not feed the solver.
        removed = self._sweep_stale_manifests()
        if removed:
            logger.info("Swept stale manifests", removed=removed, ttl_days=MANIFEST_TTL_DAYS)

        logger.debug("ConstraintStore initialized", path=str(self.base_path))

    def _sweep_stale_manifests(self) -> int:
        """Delete manifest files older than MANIFEST_TTL_DAYS by mtime."""
        cutoff = (datetime.utcnow() - timedelta(days=MANIFEST_TTL_DAYS)).timestamp()
        removed_ids: list[str] = []
        for path in self.base_path.glob("*.json"):
            if path.name == "_index.json":
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed_ids.append(path.stem)
            except OSError as e:
                logger.warning("Failed to sweep manifest", file=path.name, error=str(e))
        if removed_ids:
            index_path = self.base_path / "_index.json"
            if index_path.exists():
                try:
                    idx_data = json.loads(index_path.read_text())
                    idx_data["clusters"] = [
                        cid for cid in idx_data.get("clusters", []) if cid not in removed_ids
                    ]
                    idx_data["updated_at"] = datetime.utcnow().isoformat()
                    temp_path = index_path.with_suffix(".tmp")
                    temp_path.write_text(json.dumps(idx_data, indent=2))
                    temp_path.replace(index_path)
                except Exception as e:
                    logger.warning("Failed to update index after sweep", error=str(e))
        return len(removed_ids)
    
    async def save_manifest(self, manifest: ConstraintManifest) -> None:
        """
        Save a constraint manifest to disk.
        
        Args:
            manifest: The ConstraintManifest to save.
        """
        file_path = self.base_path / f"{manifest.cluster_id}.json"
        
        # Serialize to JSON
        data = manifest.model_dump(mode="json")
        
        # Write atomically (write to temp file, then rename)
        temp_path = file_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(data, indent=2, default=str))
        temp_path.replace(file_path)
        
        # Update index
        await self._update_index(manifest.cluster_id)
        
        logger.info(
            "Saved constraint manifest",
            cluster_id=manifest.cluster_id,
            constraints=manifest.constraint_count,
            dependencies=manifest.dependency_count,
        )
    
    async def load_manifest(self, cluster_id: str) -> ConstraintManifest | None:
        """
        Load a constraint manifest from disk.
        
        Args:
            cluster_id: The cluster ID to load.
            
        Returns:
            ConstraintManifest if found, None otherwise.
        """
        file_path = self.base_path / f"{cluster_id}.json"
        
        if not file_path.exists():
            logger.debug("Manifest not found", cluster_id=cluster_id)
            return None
        
        try:
            data = json.loads(file_path.read_text())
            manifest = ConstraintManifest.model_validate(data)
            
            logger.debug(
                "Loaded constraint manifest",
                cluster_id=cluster_id,
                constraints=manifest.constraint_count,
            )
            return manifest
            
        except Exception as e:
            logger.error("Failed to load manifest", cluster_id=cluster_id, error=str(e))
            return None
    
    def quarantine_legacy_manifests(self) -> int:
        """
        Move pre-CURRENT_MANIFEST_VERSION manifests to _legacy/.

        Pre-1.1 manifests were generated under MapMaker logic that filtered
        constraints by snapshot-time liquidity, so they're missing structurally
        valid constraints. Forcing a re-map yields correct manifests.

        Returns:
            Count of manifests moved.
        """
        legacy_dir = self.base_path / "_legacy"
        moved: list[str] = []

        for path in self.base_path.glob("*.json"):
            if path.name == "_index.json":
                continue
            try:
                data = json.loads(path.read_text())
            except Exception as e:
                logger.warning("Could not inspect manifest for quarantine", file=path.name, error=str(e))
                continue
            version = data.get("version", "1.0")
            if _parse_version(version) < _parse_version(CURRENT_MANIFEST_VERSION):
                legacy_dir.mkdir(exist_ok=True)
                path.rename(legacy_dir / path.name)
                moved.append(path.stem)

        if moved:
            # Drop quarantined cluster_ids from the index so list_clusters() stays consistent.
            index_path = self.base_path / "_index.json"
            if index_path.exists():
                try:
                    idx_data = json.loads(index_path.read_text())
                    idx_data["clusters"] = [
                        cid for cid in idx_data.get("clusters", []) if cid not in moved
                    ]
                    idx_data["updated_at"] = datetime.utcnow().isoformat()
                    temp_path = index_path.with_suffix(".tmp")
                    temp_path.write_text(json.dumps(idx_data, indent=2))
                    temp_path.replace(index_path)
                except Exception as e:
                    logger.warning("Could not update index after quarantine", error=str(e))

            logger.warning(
                f"Quarantined {len(moved)} pre-{CURRENT_MANIFEST_VERSION} manifests to _legacy/. "
                "These were generated under MapMaker logic that filtered constraints by "
                "snapshot liquidity and are missing structurally valid constraints. "
                "Run `python -m polyquant.map_maker` to regenerate fresh manifests."
            )

        return len(moved)

    async def load_all_manifests(self) -> list[ConstraintManifest]:
        """
        Load all stored constraint manifests.

        Week 3 Enhancement: Async file I/O with parallel loading for ~50ms speedup.

        Returns:
            List of all stored ConstraintManifests.
        """
        import aiofiles

        # First-pass quarantine: move pre-1.1 manifests aside before loading.
        self.quarantine_legacy_manifests()

        # Get all manifest files (excluding index)
        manifest_files = [
            f for f in self.base_path.glob("*.json")
            if f.name != "_index.json"
        ]

        async def load_one(file_path) -> ConstraintManifest | None:
            """Load a single manifest file asynchronously."""
            try:
                # Week 3: Async file I/O instead of blocking read_text()
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()

                data = json.loads(content)
                manifest = ConstraintManifest.model_validate(data)
                return manifest

            except Exception as e:
                logger.warning(
                    "Skipping invalid manifest",
                    file=file_path.name,
                    error=str(e),
                )
                return None

        # Week 3: Parallel loading with asyncio.gather (~50ms improvement for 10+ files)
        results = await asyncio.gather(*[load_one(f) for f in manifest_files])

        # Filter out None values (failed loads)
        manifests = [m for m in results if m is not None]

        logger.info("Loaded all manifests", count=len(manifests))
        return manifests
    
    async def list_clusters(self) -> list[str]:
        """
        List all stored cluster IDs.

        Returns:
            List of cluster IDs.
        """
        index_path = self.base_path / "_index.json"

        if not index_path.exists():
            return []

        try:
            data = json.loads(index_path.read_text())
            return data.get("clusters", [])
        except Exception:
            return []

    def list_clusters_with_mtimes(self) -> dict[str, float]:
        """
        Return cluster_id → on-disk mtime for every manifest file.

        Used by the hot-reload loop to detect content updates: if MapMaker
        rewrites an existing manifest (e.g. NegRisk clusters reuse stable
        cluster_ids across runs), the mtime advances and the navigator can
        re-inject the updated constraints.
        """
        result: dict[str, float] = {}
        for path in self.base_path.glob("*.json"):
            if path.name == "_index.json":
                continue
            try:
                result[path.stem] = path.stat().st_mtime
            except OSError:
                continue
        return result
    
    async def _update_index(self, cluster_id: str) -> None:
        """Update the index file with a new cluster ID."""
        index_path = self.base_path / "_index.json"
        
        # Load existing index
        if index_path.exists():
            data = json.loads(index_path.read_text())
        else:
            data = {"clusters": [], "updated_at": None}
        
        # Add if not already present
        clusters = data.get("clusters")
        if clusters is None:
            clusters = []
            data["clusters"] = clusters
            
        if cluster_id not in clusters:
            clusters.append(cluster_id)
        
        data["updated_at"] = datetime.utcnow().isoformat()
        
        # Write atomically
        temp_path = index_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(data, indent=2))
        temp_path.replace(index_path)
    
