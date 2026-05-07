import { useState } from "react";
import UploadPanel from "../components/UploadPanel";
import ResultsPanel from "../components/ResultsPanel";

export default function Dashboard() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  return (
    <main className="grid min-h-0 flex-1 grid-cols-1 gap-4 overflow-hidden p-4 xl:grid-cols-[minmax(0,1fr)_400px]">
      <div className="min-h-0 min-w-0">
        <UploadPanel
          image={image}
          setImage={setImage}
          setResult={setResult}
          setLoading={setLoading}
          loading={loading}
        />
      </div>

      <div className="min-h-0 min-w-0">
        <ResultsPanel result={result} loading={loading} />
      </div>
    </main>
  );
}
