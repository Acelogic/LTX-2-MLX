"""Weight registry for caching loaded weights in LTX-2 MLX.

The registry provides a mechanism to cache loaded weights across different
model components, avoiding redundant loading of the same weights.
"""

import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import mlx.core as mx


# Type alias for state dict
StateDict = Dict[str, mx.array]


class Registry(Protocol):
    """
    Protocol for managing state dictionaries in a registry.

    Implementations must provide:
    - add: Add a state dictionary to the registry
    - pop: Remove a state dictionary from the registry
    - get: Retrieve a state dictionary from the registry
    - clear: Clear all state dictionaries from the registry
    """

    def add(
        self,
        paths: List[str],
        op_name: Optional[str],
        state_dict: StateDict,
    ) -> str:
        """
        Add a state dictionary to the registry.

        Args:
            paths: List of source file paths.
            op_name: Optional operation/transformation name.
            state_dict: Dictionary mapping weight names to arrays.

        Returns:
            The unique ID for this state dict.
        """
        ...

    def pop(
        self,
        paths: List[str],
        op_name: Optional[str],
    ) -> Optional[StateDict]:
        """
        Remove and return a state dictionary from the registry.

        Args:
            paths: List of source file paths.
            op_name: Optional operation/transformation name.

        Returns:
            The state dict if found, None otherwise.
        """
        ...

    def get(
        self,
        paths: List[str],
        op_name: Optional[str],
    ) -> Optional[StateDict]:
        """
        Retrieve a state dictionary from the registry without removing it.

        Args:
            paths: List of source file paths.
            op_name: Optional operation/transformation name.

        Returns:
            The state dict if found, None otherwise.
        """
        ...

    def clear(self) -> None:
        """Clear all state dictionaries from the registry."""
        ...


class DummyRegistry:
    """
    Dummy registry that does not cache state dictionaries.

    Use this when you don't want caching behavior.
    """

    def add(
        self,
        _paths: List[str],
        _op_name: Optional[str],
        _state_dict: StateDict,
    ) -> str:
        """No-op add - returns empty string."""
        return ""

    def pop(
        self,
        _paths: List[str],
        _op_name: Optional[str],
    ) -> Optional[StateDict]:
        """No-op pop - always returns None."""
        return None

    def get(
        self,
        _paths: List[str],
        _op_name: Optional[str],
    ) -> Optional[StateDict]:
        """No-op get - always returns None."""
        return None

    def clear(self) -> None:
        """No-op clear."""
        pass


@dataclass
class StateDictRegistry:
    """
    Registry that caches state dictionaries for reuse.

    Thread-safe implementation using a lock for concurrent access.

    Attributes:
        _state_dicts: Internal cache mapping IDs to state dicts.
        _lock: Threading lock for thread safety.
    """

    _state_dicts: Dict[str, StateDict] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _generate_id(self, paths: List[str], op_name: Optional[str]) -> str:
        """Generate a unique ID from paths and operation name."""
        m = hashlib.sha256()
        parts = [str(Path(p).resolve()) for p in paths]
        if op_name is not None:
            parts.append(op_name)
        m.update("\0".join(parts).encode("utf-8"))
        return m.hexdigest()

    def add(
        self,
        paths: List[str],
        op_name: Optional[str],
        state_dict: StateDict,
    ) -> str:
        """
        Add a state dictionary to the registry.

        Args:
            paths: List of source file paths.
            op_name: Optional operation/transformation name.
            state_dict: Dictionary mapping weight names to arrays.

        Returns:
            The unique ID for this state dict.

        Raises:
            ValueError: If a state dict with the same ID already exists.
        """
        sd_id = self._generate_id(paths, op_name)
        with self._lock:
            if sd_id in self._state_dicts:
                raise ValueError(
                    f"State dict retrieved from {paths} with {op_name} already added. "
                    f"Check with get() first."
                )
            self._state_dicts[sd_id] = state_dict
        return sd_id

    def pop(
        self,
        paths: List[str],
        op_name: Optional[str],
    ) -> Optional[StateDict]:
        """
        Remove and return a state dictionary from the registry.

        Args:
            paths: List of source file paths.
            op_name: Optional operation/transformation name.

        Returns:
            The state dict if found, None otherwise.
        """
        with self._lock:
            return self._state_dicts.pop(self._generate_id(paths, op_name), None)

    def get(
        self,
        paths: List[str],
        op_name: Optional[str],
    ) -> Optional[StateDict]:
        """
        Retrieve a state dictionary without removing it.

        Args:
            paths: List of source file paths.
            op_name: Optional operation/transformation name.

        Returns:
            The state dict if found, None otherwise.
        """
        with self._lock:
            return self._state_dicts.get(self._generate_id(paths, op_name), None)

    def clear(self) -> None:
        """Clear all cached state dictionaries."""
        with self._lock:
            self._state_dicts.clear()

    def __len__(self) -> int:
        """Return the number of cached state dicts."""
        with self._lock:
            return len(self._state_dicts)

    def keys(self) -> List[str]:
        """Return list of cached state dict IDs."""
        with self._lock:
            return list(self._state_dicts.keys())
