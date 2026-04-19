export const DEFAULT_SEARCH_SORT = "most_recent";
export const DEFAULT_SEARCH_MIN_RATING = 0;

const VALID_SORTS = new Set(["most_recent", "most_helpful", "highest_rated", "lowest_rated"]);

export function normalizeSort(value, fallback = DEFAULT_SEARCH_SORT) {
  const raw = String(value || "");
  return VALID_SORTS.has(raw) ? raw : fallback;
}

export function normalizeMinRating(
  value,
  fallback = DEFAULT_SEARCH_MIN_RATING,
  min = DEFAULT_SEARCH_MIN_RATING,
  max = 5
) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  const clamped = Math.min(max, Math.max(min, Math.floor(n)));
  return clamped;
}

export function getSearchState(params) {
  return {
    q: params.get("q") || "",
    minRating: normalizeMinRating(params.get("min_rating")),
    sort: normalizeSort(params.get("sort")),
  };
}

export function hasSearchFilters({ q = "", minRating = DEFAULT_SEARCH_MIN_RATING, sort = DEFAULT_SEARCH_SORT } = {}) {
  return Boolean(q || Number(minRating) > DEFAULT_SEARCH_MIN_RATING || sort !== DEFAULT_SEARCH_SORT);
}

export function updateSearchParams(currentParams, next) {
  const current = new URLSearchParams(currentParams);
  Object.entries(next).forEach(([key, value]) => {
    if (value === "" || value == null) {
      current.delete(key);
    } else {
      current.set(key, String(value));
    }
  });
  return current;
}

export function resetSearchResultsState() {
  return {
    rows: [],
    hasMore: true,
    error: "",
    searchInput: "",
  };
}
