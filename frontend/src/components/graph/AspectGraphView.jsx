import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

import EdgeDetailsPanel from "./EdgeDetailsPanel";
import GraphLegend from "./GraphLegend";
import NodeDetailsPanel from "./NodeDetailsPanel";
import ReviewEvidencePanel from "./ReviewEvidencePanel";

const sentimentColors = {
  positive: "#22c55e",
  neutral: "#94a3b8",
  negative: "#ef4444",
  mixed: "#f59e0b",
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(value, max));
}

function edgeColor(edge) {
  return sentimentColors[edge.polarity_hint] || "#38bdf8";
}

function normalizeNode(rawNode, index, scope) {
  if (!rawNode || typeof rawNode !== "object") return null;

  const fallbackId = `node-${index}`;
  const id = String(rawNode.id ?? rawNode.label ?? fallbackId).trim() || fallbackId;
  const label = String(rawNode.label ?? rawNode.id ?? fallbackId).trim() || fallbackId;
  const sentimentKey = scope === "single_review" ? rawNode.sentiment : rawNode.dominant_sentiment;
  const confidence = Number(rawNode.confidence ?? 0);
  const frequency = Number(rawNode.frequency ?? 1);
  const evidenceStart = Number(rawNode.evidence_start);
  const evidenceEnd = Number(rawNode.evidence_end);

  return {
    ...rawNode,
    id,
    label,
    sentiment: typeof rawNode.sentiment === "string" ? rawNode.sentiment.toLowerCase() : rawNode.sentiment,
    dominant_sentiment:
      typeof rawNode.dominant_sentiment === "string" ? rawNode.dominant_sentiment.toLowerCase() : rawNode.dominant_sentiment,
    polarity_hint: typeof rawNode.polarity_hint === "string" ? rawNode.polarity_hint.toLowerCase() : rawNode.polarity_hint,
    confidence: Number.isFinite(confidence) ? confidence : 0,
    frequency: Number.isFinite(frequency) ? frequency : 1,
    evidence: typeof rawNode.evidence === "string" ? rawNode.evidence : null,
    evidence_start: Number.isFinite(evidenceStart) ? evidenceStart : null,
    evidence_end: Number.isFinite(evidenceEnd) ? evidenceEnd : null,
    _sentiment_key:
      typeof sentimentKey === "string" && sentimentKey.trim() ? sentimentKey.toLowerCase() : "neutral",
  };
}

function normalizeGraph(graph, scope) {
  const rawNodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
  const normalizedNodes = rawNodes
    .map((node, index) => normalizeNode(node, index, scope))
    .filter(Boolean);

  const validNodeIds = new Set(normalizedNodes.map((node) => node.id));
  const rawEdges = Array.isArray(graph?.edges) ? graph.edges : [];
  const normalizedEdges = rawEdges
    .map((edge, index) => {
      if (!edge || typeof edge !== "object") return null;
      const source = String(edge.source ?? "").trim();
      const target = String(edge.target ?? "").trim();
      if (!source || !target || !validNodeIds.has(source) || !validNodeIds.has(target)) {
        return null;
      }

      const weight = Number(edge.weight ?? 1);
      return {
        ...edge,
        id: `${source}-${target}-${index}`,
        source,
        target,
        weight: Number.isFinite(weight) ? weight : 1,
        directional: Boolean(edge.directional),
        polarity_hint: typeof edge.polarity_hint === "string" ? edge.polarity_hint.toLowerCase() : null,
      };
    })
    .filter(Boolean);

  return {
    ...graph,
    nodes: normalizedNodes,
    edges: normalizedEdges,
  };
}

function buildGraphData(graph, scope) {
  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];
  const maxFrequency = Math.max(...nodes.map((node) => Number(node.frequency || 1)), 1);
  const maxWeight = Math.max(...edges.map((edge) => Number(edge.weight || 1)), 1);

  const nodeMap = new Map(
    nodes.map((node) => {
      const size =
        scope === "single_review"
          ? clamp(8 + Number(node.confidence || 0) * 8, 8, 16)
          : clamp(7 + (Number(node.frequency || 1) / maxFrequency) * 14, 7, 21);

      return [
        node.id,
        {
          ...node,
          id: node.id,
          color: sentimentColors[node._sentiment_key] || sentimentColors.neutral,
          radius: size,
        },
      ];
    }),
  );

  const rawLinks = edges
    .map((edge) => {
      const src = nodeMap.get(edge.source);
      const dst = nodeMap.get(edge.target);
      if (!src || !dst) return null;
      return {
        ...edge,
        source: src,
        target: dst,
        width: clamp(1 + (Number(edge.weight || 1) / maxWeight) * 4, 1, 5),
        color: edgeColor(edge),
        sourceLabel: src.label,
        targetLabel: dst.label,
      };
    })
    .filter(Boolean);

  // Protect UI responsiveness on dense corpus graphs.
  const linkList =
    scope === "batch" && rawLinks.length > 900
      ? rawLinks.sort((a, b) => Number(b.weight || 0) - Number(a.weight || 0)).slice(0, 900)
      : rawLinks;

  return {
    nodes: Array.from(nodeMap.values()),
    links: linkList,
  };
}

export default function AspectGraphView({
  graph,
  scope = "batch",
  isDark = false,
  reviewText = "",
  emptyMessage = "No graph data available yet.",
}) {
  const [selection, setSelection] = useState({ type: "node", data: null });
  const fgRef = useRef(null);
  const didFitRef = useRef(false);
  const normalizedGraph = useMemo(() => normalizeGraph(graph, scope), [graph, scope]);
  const forceData = useMemo(() => buildGraphData(normalizedGraph, scope), [normalizedGraph, scope]);
  const hasGraph = Boolean((forceData?.nodes || []).length);

  useEffect(() => {
    setSelection({ type: "node", data: null });
  }, [normalizedGraph, scope]);

  useEffect(() => {
    if (!fgRef.current || !hasGraph) return;
    const fg = fgRef.current;
    didFitRef.current = false;

    const linkDistance = scope === "single_review" ? 120 : 140;
    const chargeStrength = scope === "single_review" ? -260 : -380;

    const linkForce = fg.d3Force("link");
    if (linkForce && typeof linkForce.distance === "function") {
      linkForce.distance(linkDistance);
    }
    const chargeForce = fg.d3Force("charge");
    if (chargeForce && typeof chargeForce.strength === "function") {
      chargeForce.strength(chargeStrength);
    }
  }, [forceData, scope, hasGraph]);

  return (
    <div className="space-y-4">
      <GraphLegend scope={scope} isDark={isDark} />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.55fr)_360px]">
        <div className={`overflow-hidden rounded-[28px] border ${isDark ? "border-slate-800 bg-[#071120]" : "border-slate-200 bg-white"}`}>
          <div className={`flex items-center justify-between border-b px-5 py-4 ${isDark ? "border-slate-800" : "border-slate-200"}`}>
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-500">
                {scope === "single_review" ? "Single Review Explanation Graph" : "Corpus Aspect Graph"}
              </p>
              <p className={`mt-1 text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>
                {scope === "single_review"
                  ? "Directed edges follow evidence order inside one review."
                  : "Undirected edges connect aspects that co-occur across reviews."}
              </p>
            </div>
            <div className={`text-xs uppercase tracking-[0.18em] ${isDark ? "text-slate-500" : "text-slate-400"}`}>
              {(normalizedGraph?.nodes || []).length} nodes / {(normalizedGraph?.edges || []).length} edges
            </div>
          </div>

          {hasGraph ? (
            <div className="h-[540px] w-full">
              <ForceGraph2D
                ref={fgRef}
                graphData={forceData}
                nodeRelSize={6}
                linkDirectionalArrowLength={scope === "single_review" ? 9 : 0}
                linkDirectionalArrowRelPos={scope === "single_review" ? 0.96 : 0}
                linkDirectionalArrowColor={(link) => (scope === "single_review" ? link.color || "#38bdf8" : "transparent")}
                cooldownTicks={70}
                onNodeClick={(node) => {
                  setSelection({ type: "node", data: node || null });
                }}
                onLinkClick={(link) => {
                  setSelection({ type: "edge", data: link || null });
                }}
                onBackgroundClick={() => setSelection({ type: "node", data: null })}
                onNodeDragEnd={(node) => {
                  node.fx = node.x;
                  node.fy = node.y;
                }}
                onEngineStop={() => {
                  if (!didFitRef.current && fgRef.current) {
                    didFitRef.current = true;
                    fgRef.current.zoomToFit(500, 40);
                  }
                }}
                nodeCanvasObject={(node, ctx, globalScale) => {
                  const radius = Number(node.radius || 9);
                  ctx.beginPath();
                  ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
                  ctx.fillStyle = node.color || "#94a3b8";
                  ctx.fill();

                  const rawLabel = String(node.label || node.id || "");
                  if (!rawLabel) return;
                  const label = rawLabel.length > 20 ? `${rawLabel.slice(0, 20)}...` : rawLabel;
                  const fontSize = Math.max(9, 11 / Math.max(globalScale, 0.8));
                  ctx.font = `${fontSize}px sans-serif`;
                  ctx.textAlign = "center";
                  ctx.textBaseline = "top";
                  ctx.fillStyle = isDark ? "#e5eefc" : "#10223d";
                  ctx.fillText(label, node.x, node.y + radius + 2);
                }}
                linkWidth={(link) => Number(link.width || 1)}
                linkColor={(link) => link.color || "#38bdf8"}
                nodeLabel="label"
              />
            </div>
          ) : (
            <div className={`grid h-[540px] place-items-center px-6 text-center text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>
              {emptyMessage}
            </div>
          )}
        </div>

        <div className="space-y-4">
          <NodeDetailsPanel node={selection.type === "node" ? selection.data : null} scope={scope} isDark={isDark} />
          <EdgeDetailsPanel edge={selection.type === "edge" ? selection.data : null} scope={scope} isDark={isDark} />
        </div>
      </div>

      {scope === "single_review" ? <ReviewEvidencePanel text={reviewText} nodes={normalizedGraph?.nodes || []} isDark={isDark} /> : null}
    </div>
  );
}
