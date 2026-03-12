import { useEffect, useState } from "react";
import { deleteMyReview, getMyReviews } from "../../api/client";
import { useAuth } from "../../auth/AuthContext";
import UserShell from "../../components/user/UserShell";

export default function MyReviewsPage() {
  const { token } = useAuth();
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    getMyReviews(token).then(setRows).catch((ex) => setError(ex.message || "Failed to load reviews"));
  }, [token]);

  async function handleDelete(reviewId) {
    try {
      await deleteMyReview(token, reviewId);
      setRows((prev) => prev.filter((r) => r.review_id !== reviewId));
    } catch (ex) {
      setError(ex.message || "Delete failed");
    }
  }

  return (
    <UserShell title="My Reviews">
      {error ? <div className="rounded-lg bg-red-100 px-3 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-200">{error}</div> : null}
      <div className="space-y-3">
        {rows.map((r) => (
          <article key={r.review_id} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-semibold text-slate-900 dark:text-slate-100">Product: {r.product_id}</p>
                <p className="text-xs text-slate-500 dark:text-slate-300">{new Date(r.review_date).toLocaleString()}</p>
              </div>
              <div className="text-amber-600">{r.rating} ★</div>
            </div>
            {r.review_title ? <h3 className="mt-2 font-medium text-slate-900 dark:text-slate-100">{r.review_title}</h3> : null}
            <p className="mt-1 text-sm text-slate-700 dark:text-slate-200">{r.review_text}</p>
            <div className="mt-3 flex gap-2">
              <button type="button" disabled className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm text-slate-400 dark:border-slate-700 dark:text-slate-500">
                Edit (coming soon)
              </button>
              <button type="button" onClick={() => handleDelete(r.review_id)} className="rounded-lg border border-red-300 px-3 py-1.5 text-sm text-red-700 dark:border-red-700 dark:text-red-300">
                Delete
              </button>
            </div>
          </article>
        ))}
      </div>
    </UserShell>
  );
}
