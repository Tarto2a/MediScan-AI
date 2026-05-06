export default function ResultsPanel({ result, loading }) {
  return (
    <div className="bg-white p-5 rounded-xl shadow">
      <h3 className="font-semibold mb-3">Classification Results</h3>

      {loading && <p>Analyzing...</p>}

      {result && (
        <>
          <img
            src={result.gradcam}
            className="rounded-lg mb-3"
          />

          <p className="text-sm">
            <strong>CLASSIFICATION:</strong> {result.prediction}
          </p>

          <p className="text-sm text-gray-600">
            Confidence: {result.confidence}%
          </p>

          <button className="mt-3 bg-teal-600 text-white px-3 py-2 rounded">
            Generate Report
          </button>
        </>
      )}
    </div>
  );
}