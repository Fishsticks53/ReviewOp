export function filterGraphSuggestions(options = [], query = "", limit = 8) {
  const cleaned = Array.isArray(options)
    ? options
        .map((item, index) => ({ value: String(item || "").trim(), index }))
        .filter((item) => Boolean(item.value))
    : [];

  const needle = String(query || "").trim().toLowerCase();
  const ranked = needle
    ? cleaned
        .filter((item) => item.value.toLowerCase().includes(needle))
        .sort((a, b) => {
          const aStarts = a.value.toLowerCase().startsWith(needle) ? 0 : 1;
          const bStarts = b.value.toLowerCase().startsWith(needle) ? 0 : 1;
          if (aStarts !== bStarts) return aStarts - bStarts;
          return a.index - b.index;
        })
    : [...cleaned].sort((a, b) => a.value.localeCompare(b.value));

  return ranked.slice(0, Math.max(1, limit)).map((item) => item.value);
}

export function resetGraphFilters(defaultFilters) {
  return {
    ...(defaultFilters || {}),
  };
}
