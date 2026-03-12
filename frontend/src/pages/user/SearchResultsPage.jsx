import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { searchProducts } from "../../api/client";
import { useAuth } from "../../auth/AuthContext";
import ProductCard from "../../components/user/ProductCard";
import UserShell from "../../components/user/UserShell";

const SORT_OPTIONS = [
  { value: "most_recent", label: "Most Recent" },
  { value: "most_helpful", label: "Most Helpful" },
  { value: "highest_rated", label: "Highest Rated" },
  { value: "lowest_rated", label: "Lowest Rated" },
];

export default function SearchResultsPage() {
  const { token } = useAuth();
  const [params, setParams] = useSearchParams();
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");
  const q = params.get("q") || "";
  const minRating = Number(params.get("min_rating") || "1");
  const sort = params.get("sort") || "most_recent";

  useEffect(() => {
    searchProducts(token, { q, min_rating: minRating, sort })
      .then(setRows)
      .catch((ex) => setError(ex.message || "Search failed"));
  }, [token, q, minRating, sort]);

  return (
    <UserShell title="Search Results">
      <div className="flex flex-wrap gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <input
          value={q}
          onChange={(e) => setParams({ q: e.target.value, min_rating: String(minRating), sort })}
          className="min-w-[220px] flex-1 rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100"
          placeholder="Search products"
        />
        <select value={minRating} onChange={(e) => setParams({ q, min_rating: e.target.value, sort })} className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100">
          <option value={4}>4 stars and above</option>
          <option value={3}>3 stars and above</option>
          <option value={2}>2 stars and above</option>
          <option value={1}>1 star and above</option>
        </select>
        <select value={sort} onChange={(e) => setParams({ q, min_rating: String(minRating), sort: e.target.value })} className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100">
          {SORT_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>
      {error ? <div className="rounded-lg bg-red-100 px-3 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-200">{error}</div> : null}
      <div className="grid gap-3 md:grid-cols-2">
        {rows.map((p) => (
          <ProductCard key={p.product_id} product={p} />
        ))}
      </div>
      {!rows.length && !error ? <p className="text-sm text-slate-600 dark:text-slate-300">No products found.</p> : null}
    </UserShell>
  );
}
