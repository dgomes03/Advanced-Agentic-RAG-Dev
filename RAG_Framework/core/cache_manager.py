"""Utility for managing KV cache checkpoints to exclude tool results from cache."""


class CacheManager:
    """
    Manages KV cache checkpoints to support selective caching.

    This allows saving the cache state before processing tool results,
    then trimming back after generation to exclude bulky tool results
    from the cache, reducing RAM usage significantly.
    """

    @staticmethod
    def get_checkpoint(cache):
        """
        Record current cache position for later restoration.

        Args:
            cache: MLX-LM KVCache object (list of cache layers)

        Returns:
            List of offsets for each cache layer, or None if cache is None
        """
        if cache is None:
            return None
        return [layer.offset for layer in cache]

    @staticmethod
    def restore_checkpoint(cache, checkpoint):
        """
        Trim cache back to a saved checkpoint.

        This removes tokens added after the checkpoint was taken,
        effectively excluding tool results from the persistent cache.

        Args:
            cache: MLX-LM KVCache object (list of cache layers)
            checkpoint: List of offsets from get_checkpoint()
        """
        if cache is None or checkpoint is None:
            return

        for layer, saved_offset in zip(cache, checkpoint):
            current_offset = layer.offset
            trim_amount = current_offset - saved_offset
            if trim_amount > 0:
                layer.trim(trim_amount)

    @staticmethod
    def get_cache_size(cache):
        """
        Get total tokens currently in cache.

        Args:
            cache: MLX-LM KVCache object (list of cache layers)

        Returns:
            Number of tokens in cache, or 0 if cache is None/empty
        """
        if cache and len(cache) > 0:
            return cache[0].offset
        return 0

    @staticmethod
    def log_cache_stats(cache, label="Cache"):
        """
        Log current cache statistics for debugging/monitoring.

        Args:
            cache: MLX-LM KVCache object
            label: Label for the log message
        """
        size = CacheManager.get_cache_size(cache)
        print(f"[CacheManager] {label}: {size} tokens")
        return size
