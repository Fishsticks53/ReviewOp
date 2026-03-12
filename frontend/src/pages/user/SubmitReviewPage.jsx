import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { submitReview } from "../../api/client";
import { useAuth } from "../../auth/AuthContext";
import UserShell from "../../components/user/UserShell";

export default function SubmitReviewPage() {
  const { productId } = useParams();
  const { token } = useAuth();
  const nav = useNavigate();
  const [productRef, setProductRef] = useState(productId || "");
  const [productName, setProductName] = useState("");
  const [rating, setRating] = useState(5);
  const [reviewText, setReviewText] = useState("");
  const [reviewTitle, setReviewTitle] = useState("");
  const [pros, setPros] = useState("");
  const [cons, setCons] = useState("");
  const [recommendation, setRecommendation] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (productId) setProductRef(productId);
  }, [productId]);

  async function onSubmit(e) {
    e.preventDefault();
    setError("");
    const cleanProductId = (productRef || "").trim();
    if (!cleanProductId) {
      setError("Product reference is required.");
      return;
    }
    if (!reviewText.trim()) {
      setError("Review text is required.");
      return;
    }
    setLoading(true);
    try {
      await submitReview(token, {
        product_id: cleanProductId,
        product_name: productName.trim() || null,
        rating,
        review_text: reviewText,
        review_title: reviewTitle || null,
        pros: pros || null,
        cons: cons || null,
        recommendation,
      });
      nav(`/products/${encodeURIComponent(cleanProductId)}`);
    } catch (ex) {
      setError(ex.message || "Submit failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <UserShell title="Write Review">
      <form onSubmit={onSubmit} className="mx-auto w-full max-w-3xl space-y-3 rounded-xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        {error ? <div className="rounded-lg bg-red-100 px-3 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-200">{error}</div> : null}
        <div className="grid gap-3 md:grid-cols-2">
          <input value={productName} onChange={(e) => setProductName(e.target.value)} placeholder="Product name (e.g. iPhone 14)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
          <input value={productRef} onChange={(e) => setProductRef(e.target.value)} placeholder="Product ID (e.g. SKU-1001)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        </div>
        <select value={rating} onChange={(e) => setRating(Number(e.target.value))} className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100">
          <option value={5}>5 stars</option>
          <option value={4}>4 stars</option>
          <option value={3}>3 stars</option>
          <option value={2}>2 stars</option>
          <option value={1}>1 star</option>
        </select>
        <input value={reviewTitle} onChange={(e) => setReviewTitle(e.target.value)} placeholder="Review title (optional)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        <textarea value={reviewText} onChange={(e) => setReviewText(e.target.value)} rows={6} placeholder="Share your experience..." className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        <input value={pros} onChange={(e) => setPros(e.target.value)} placeholder="Pros (optional)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        <input value={cons} onChange={(e) => setCons(e.target.value)} placeholder="Cons (optional)" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
        <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-200">
          <input type="checkbox" checked={recommendation} onChange={(e) => setRecommendation(e.target.checked)} />
          I recommend this product
        </label>
        <button disabled={loading} className="rounded-lg bg-emerald-600 px-4 py-2 text-white disabled:opacity-60">
          {loading ? "Submitting..." : "Submit Review"}
        </button>
      </form>
    </UserShell>
  );
}
