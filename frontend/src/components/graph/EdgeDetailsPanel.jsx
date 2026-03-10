export default function EdgeDetailsPanel({ edge, scope = "batch", isDark = false }) {
  return (
    <div className={`rounded-2xl border p-4 ${isDark ? "border-slate-800 bg-[#0b1220]" : "border-slate-200 bg-white"}`}>
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-sky-500">Edge Details</p>
      {edge ? (
        <>
          <h4 className="mt-1 text-xl font-semibold">
            {edge.sourceLabel || edge.source} {"->"} {edge.targetLabel || edge.target}
          </h4>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <div>
              <p className="text-[11px] uppercase tracking-[0.18em] text-slate-400">Type</p>
              <p className="mt-1 text-sm font-medium">{edge.type || "-"}</p>
            </div>
            <div>
              <p className="text-[11px] uppercase tracking-[0.18em] text-slate-400">Weight</p>
              <p className="mt-1 text-sm font-medium">{Number(edge.weight || 0).toFixed(scope === "batch" ? 0 : 1)}</p>
            </div>
            <div>
              <p className="text-[11px] uppercase tracking-[0.18em] text-slate-400">Directional</p>
              <p className="mt-1 text-sm font-medium">{edge.directional ? "Yes" : "No"}</p>
            </div>
            <div>
              <p className="text-[11px] uppercase tracking-[0.18em] text-slate-400">Polarity Hint</p>
              <p className="mt-1 text-sm font-medium">{edge.polarity_hint || "-"}</p>
            </div>
            {scope === "batch" ? (
              <div>
                <p className="text-[11px] uppercase tracking-[0.18em] text-slate-400">Pair Count</p>
                <p className="mt-1 text-sm font-medium">{edge.pair_count ?? edge.weight ?? "-"}</p>
              </div>
            ) : null}
          </div>
        </>
      ) : (
        <p className={`mt-3 text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>
          Click a graph edge to inspect the relationship metadata.
        </p>
      )}
    </div>
  );
}
