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
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from polyquant.utils import config, get_logger

logger = get_logger(__name__)


class StoredConstraint(BaseModel):
    """A single constraint stored in the ConstraintStore."""
    constraint_id: str
    description: str
    coefficients: dict[str, float]
    rhs: float
    confidence: float
    reasoning: str
    source_markets: list[str] = Field(default_factory=list)


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
    constraints: list[StoredConstraint] = Field(default_factory=list)
    dependencies: list[StoredDependency] = Field(default_factory=list)
    correlations: list[Any] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0"
    
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
        
        logger.debug("ConstraintStore initialized", path=str(self.base_path))
    
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
    
    async def load_all_manifests(self) -> list[ConstraintManifest]:
        """
        Load all stored constraint manifests.

        Week 3 Enhancement: Async file I/O with parallel loading for ~50ms speedup.

        Returns:
            List of all stored ConstraintManifests.
        """
        import aiofiles

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
    
    async def delete_manifest(self, cluster_id: str) -> bool:
        """
        Delete a constraint manifest.
        
        Args:
            cluster_id: The cluster ID to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        file_path = self.base_path / f"{cluster_id}.json"
        
        if not file_path.exists():
            return False
        
        file_path.unlink()
        
        # Update index
        await self._remove_from_index(cluster_id)
        
        logger.info("Deleted constraint manifest", cluster_id=cluster_id)
        return True
    
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
    
    async def _remove_from_index(self, cluster_id: str) -> None:
        """Remove a cluster ID from the index."""
        index_path = self.base_path / "_index.json"
        
        if not index_path.exists():
            return
        
        data = json.loads(index_path.read_text())
        
        clusters = data.get("clusters", [])
        if cluster_id in clusters:
            clusters.remove(cluster_id)
            data["updated_at"] = datetime.utcnow().isoformat()
            
            temp_path = index_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(data, indent=2))
            temp_path.replace(index_path)
