import { useRef, useState } from "react";

export default function UploadPanel({
  image,
  setImage,
  setResult,
  setLoading,
  loading,
}) {
  const [file, setFile] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileSelect = (selectedFile) => {
    if (!selectedFile?.type?.startsWith("image/")) return;
    setFile(selectedFile);
    setResult(null);
    setImage(URL.createObjectURL(selectedFile));
  };

  const handleAnalyze = async () => {
    if (!file || loading) return;

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Prediction failed");
      }

      setResult(data);
    } catch (err) {
      console.error(err);
      setResult({
        error: err instanceof Error ? err.message : "Prediction failed",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    handleFileSelect(e.dataTransfer.files[0]);
  };

  return (
    <section className="flex h-full min-h-0 flex-col rounded-lg border border-slate-200 bg-white shadow-sm">
      <div className="shrink-0 border-b border-slate-200 px-5 py-3">
        <h2 className="text-base font-semibold text-slate-950">Input Image</h2>
        <p className="text-sm text-slate-500">Upload a chest X-ray or CT slice for classification.</p>
      </div>

      <div className="flex min-h-0 flex-1 flex-col p-4">
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragOver(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            setIsDragOver(false);
          }}
          onDrop={handleDrop}
          className={`flex min-h-0 flex-1 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-4 text-center transition ${
            isDragOver
              ? "border-teal-500 bg-teal-50"
              : "border-slate-300 bg-slate-50 hover:border-teal-500 hover:bg-white"
          }`}
        >
          {image ? (
            <img
              src={image}
              alt="Uploaded medical image"
              className="max-h-full max-w-full rounded-lg object-contain shadow-sm"
            />
          ) : (
            <div>
              <div className="mx-auto mb-4 flex size-14 items-center justify-center rounded-full bg-teal-50 text-2xl text-teal-700">
                +
              </div>
              <p className="font-medium text-slate-800">Drop an image here</p>
              <p className="mt-1 text-sm text-slate-500">or click to browse from your device</p>
            </div>
          )}
        </button>

        <input
          type="file"
          ref={fileInputRef}
          accept="image/*"
          onChange={(e) => handleFileSelect(e.target.files[0])}
          className="hidden"
        />

        <div className="mt-4 flex shrink-0 flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <p className="truncate text-sm font-medium text-slate-800">{file?.name || "No image selected"}</p>
            {file && <p className="text-xs text-slate-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>}
          </div>

          <button
            type="button"
            onClick={handleAnalyze}
            disabled={!file || loading}
            className="rounded-lg bg-teal-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-teal-700 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
          >
            {loading ? "Analyzing..." : "Analyze Image"}
          </button>
        </div>
      </div>
    </section>
  );
}
