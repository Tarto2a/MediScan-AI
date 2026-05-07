export default function ResultsPanel({ result, loading }) {
  const confidence =
    typeof result?.confidence === "number"
      ? `${(result.confidence * 100).toFixed(1)}%`
      : "N/A";

  return (
    <section className="flex h-full min-h-0 flex-col rounded-lg border border-slate-200 bg-white shadow-sm">
      <div className="shrink-0 border-b border-slate-200 px-5 py-3">
        <h2 className="text-base font-semibold text-slate-950">Classification</h2>
        <p className="text-sm text-slate-500">Prediction, confidence, and attention overlay.</p>
      </div>

      <div className="min-h-0 flex-1 overflow-hidden p-4">
        {loading && (
          <div className="rounded-lg border border-teal-100 bg-teal-50 p-4">
            <div className="h-2 w-full overflow-hidden rounded-full bg-teal-100">
              <div className="h-full w-2/3 animate-pulse rounded-full bg-teal-600" />
            </div>
            <p className="mt-3 text-sm font-medium text-teal-900">Analyzing image...</p>
          </div>
        )}

        {!loading && !result && (
          <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 p-5 text-sm text-slate-500">
            Results will appear here after analysis.
          </div>
        )}

        {!loading && result?.error && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            {result.error}
          </div>
        )}

        {!loading && result && !result.error && (
          <div className="flex h-full min-h-0 flex-col gap-4 overflow-hidden">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Primary result</p>
              <div className="mt-2 rounded-lg bg-slate-950 p-3 text-white">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="text-xl font-semibold">{result.prediction || "N/A"}</p>
                    {result.confidence_band && (
                      <p className="mt-1 text-sm text-slate-300">{result.confidence_band} confidence</p>
                    )}
                  </div>
                  <p className="rounded-full bg-teal-400/15 px-3 py-1 text-sm font-semibold text-teal-200">
                    {confidence}
                  </p>
                </div>
              </div>
            </div>

            {result.gradcam && (
              <div className="min-h-0 flex-1 overflow-hidden">
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Attention overlay</p>
                <img
                  src={result.gradcam}
                  alt="Grad-CAM attention overlay"
                  className="max-h-[28vh] w-full rounded-lg border border-slate-200 object-contain"
                />
                {result.gradcam_method && (
                  <p className="mt-1 truncate text-xs text-slate-500">{result.gradcam_method}</p>
                )}
              </div>
            )}

            {Array.isArray(result.top_predictions) && result.top_predictions.length > 0 && (
              <div className="shrink-0">
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Ranked classes</p>
                <div className="space-y-1.5">
                  {result.top_predictions.slice(0, 3).map((item) => {
                    const percent = Math.max(0, Math.min(100, item.probability * 100));

                    return (
                      <div key={`${item.rank}-${item.class}`}>
                        <div className="mb-1 flex justify-between gap-3 text-xs">
                          <span className="font-medium text-slate-700">{item.rank}. {item.class}</span>
                          <span className="text-slate-500">{percent.toFixed(1)}%</span>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-full bg-slate-100">
                          <div className="h-full rounded-full bg-teal-600" style={{ width: `${percent}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
