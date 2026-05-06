import { useState } from "react";
import UploadPanel from "../components/UploadPanel";
import ResultsPanel from "../components/ResultsPanel";

export default function Dashboard() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="p-6 grid grid-cols-3 gap-6">
      <div className="col-span-2">
        <UploadPanel
          setImage={setImage}
          setResult={setResult}
          setLoading={setLoading}
          loading={loading}
        />
      </div>

      <div className="space-y-6">
        <ResultsPanel result={result} loading={loading} />
      </div>
    </div>
  );
}