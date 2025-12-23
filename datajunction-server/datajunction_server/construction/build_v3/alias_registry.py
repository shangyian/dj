"""
Alias Registry: Clean SQL naming without _DOT_ pollution.

This module provides the AliasRegistry class which maps semantic names
(like 'orders.customer.country') to clean SQL aliases (like 'country').

The registry ensures:
- Aliases are unique within a query
- Shortest possible name is used when unique
- Progressively qualifies when conflicts exist
- Falls back to numeric suffixes as last resort
"""

from __future__ import annotations

import re


class AliasRegistry:
    """
    Maps semantic names to clean, unique SQL aliases.
    
    The registry tries to produce the shortest unique alias:
    - 'orders.customer.country' → 'country' (if unique)
    - 'orders.customer.country' → 'customer_country' (if 'country' taken)
    - 'orders.customer.country' → 'orders_customer_country' (if both taken)
    - 'orders.customer.country' → 'country_1' (fallback with numeric suffix)
    
    Usage:
        registry = AliasRegistry()
        
        alias1 = registry.register('orders.customer.country')  # → 'country'
        alias2 = registry.register('users.country')  # → 'users_country' (conflict)
        
        # Later, look up the alias
        alias = registry.get_alias('orders.customer.country')  # → 'country'
        semantic = registry.get_semantic('country')  # → 'orders.customer.country'
    """
    
    # Characters that are not valid in SQL identifiers (will be replaced with _)
    INVALID_CHARS = re.compile(r'[^a-zA-Z0-9_]')
    
    def __init__(self):
        self._semantic_to_alias: dict[str, str] = {}
        self._alias_to_semantic: dict[str, str] = {}
        self._used_aliases: set[str] = set()
    
    def register(self, semantic_name: str) -> str:
        """
        Register a semantic name and return its SQL alias.
        
        If already registered, returns the existing alias.
        Otherwise, generates a new unique alias.
        
        Args:
            semantic_name: Full semantic name (e.g., 'orders.customer.country')
            
        Returns:
            Clean SQL alias (e.g., 'country')
        """
        # Return existing alias if already registered
        if semantic_name in self._semantic_to_alias:
            return self._semantic_to_alias[semantic_name]
        
        # Generate a new unique alias
        alias = self._generate_alias(semantic_name)
        self._register(semantic_name, alias)
        return alias
    
    def get_alias(self, semantic_name: str) -> str | None:
        """Get the alias for a semantic name, or None if not registered."""
        return self._semantic_to_alias.get(semantic_name)
    
    def get_semantic(self, alias: str) -> str | None:
        """Get the semantic name for an alias, or None if not found."""
        return self._alias_to_semantic.get(alias)
    
    def is_registered(self, semantic_name: str) -> bool:
        """Check if a semantic name is already registered."""
        return semantic_name in self._semantic_to_alias
    
    def _register(self, semantic_name: str, alias: str) -> None:
        """Internal: register a semantic name → alias mapping."""
        self._semantic_to_alias[semantic_name] = alias
        self._alias_to_semantic[alias] = semantic_name
        self._used_aliases.add(alias)
    
    def _generate_alias(self, semantic_name: str) -> str:
        """
        Generate a unique alias for a semantic name.
        
        Strategy:
        1. Try the last part: 'orders.customer.country' → 'country'
        2. Try progressively longer suffixes: 'customer_country', 'orders_customer_country'
        3. Fall back to numeric suffix: 'country_1', 'country_2', ...
        """
        # Split on dots and clean each part
        parts = [self._clean_part(p) for p in semantic_name.split('.')]
        parts = [p for p in parts if p]  # Remove empty parts
        
        if not parts:
            # Edge case: empty or all-invalid name
            return self._generate_fallback_alias('col')
        
        # Strategy 1 & 2: Try progressively longer suffixes
        for i in range(len(parts)):
            candidate = '_'.join(parts[-(i + 1):])
            if candidate and candidate not in self._used_aliases:
                return candidate
        
        # Strategy 3: Numeric suffix fallback
        base = parts[-1] if parts[-1] else 'col'
        return self._generate_fallback_alias(base)
    
    def _generate_fallback_alias(self, base: str) -> str:
        """Generate an alias with numeric suffix."""
        counter = 1
        while f"{base}_{counter}" in self._used_aliases:
            counter += 1
        return f"{base}_{counter}"
    
    def _clean_part(self, part: str) -> str:
        """
        Clean a name part to be a valid SQL identifier.
        
        - Replace invalid characters with underscores
        - Remove leading/trailing underscores
        - Collapse multiple underscores
        """
        # Replace invalid chars with underscore
        cleaned = self.INVALID_CHARS.sub('_', part)
        
        # Collapse multiple underscores
        while '__' in cleaned:
            cleaned = cleaned.replace('__', '_')
        
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        
        return cleaned
    
    def all_mappings(self) -> dict[str, str]:
        """Return all semantic → alias mappings."""
        return dict(self._semantic_to_alias)
    
    def clear(self) -> None:
        """Clear all registrations."""
        self._semantic_to_alias.clear()
        self._alias_to_semantic.clear()
        self._used_aliases.clear()
    
    def __len__(self) -> int:
        """Number of registered aliases."""
        return len(self._semantic_to_alias)
    
    def __contains__(self, semantic_name: str) -> bool:
        """Check if a semantic name is registered."""
        return semantic_name in self._semantic_to_alias


class ScopedAliasRegistry(AliasRegistry):
    """
    An alias registry with scope support for nested queries.
    
    Useful when generating CTEs where inner queries might have
    different alias requirements than outer queries.
    
    Usage:
        registry = ScopedAliasRegistry()
        
        # Register some aliases
        registry.register('orders.total')  # → 'total'
        
        # Enter a new scope (e.g., for a subquery)
        registry.push_scope()
        registry.register('users.total')  # → 'total' (different scope)
        
        # Exit scope
        registry.pop_scope()
        # Back to original state
    """
    
    def __init__(self):
        super().__init__()
        self._scope_stack: list[tuple[dict, dict, set]] = []
    
    def push_scope(self) -> None:
        """Push a new scope onto the stack."""
        # Save current state
        self._scope_stack.append((
            dict(self._semantic_to_alias),
            dict(self._alias_to_semantic),
            set(self._used_aliases),
        ))
    
    def pop_scope(self) -> None:
        """Pop the current scope and restore the previous state."""
        if not self._scope_stack:
            raise ValueError("Cannot pop scope: no scope to pop")
        
        (
            self._semantic_to_alias,
            self._alias_to_semantic,
            self._used_aliases,
        ) = self._scope_stack.pop()
    
    @property
    def scope_depth(self) -> int:
        """Current scope depth (0 = root scope)."""
        return len(self._scope_stack)

