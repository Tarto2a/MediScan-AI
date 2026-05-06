import { useState, useRef } from "react";

export default function UploadPanel({
  setImage,
  setResult,
  setLoading,
  loading,
}) {
  const [file, setFile] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileSelect = (selectedFile) => {
    if (!selectedFile.type.startsWith("image/")) return;
    setFile(selectedFile);
    setImage(URL.createObjectURL(selectedFile));
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      setResult(data);
    } catch (err) {
      console.error(err);
    }

    setLoading(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith("image/")) {
      handleFileSelect(droppedFile);
    }
  };

  return (
    <div className="bg-white p-6 rounded-xl shadow">
      <div
        className={`border-2 border-dashed rounded-lg p-10 text-center transition-colors ${
          isDragOver ? "border-blue-500 bg-blue-50" : "border-gray-300"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {file ? (
          <img
            src={URL.createObjectURL(file)}
            alt="Uploaded image"
            className="max-w-full max-h-64 mx-auto mb-4 rounded shadow"
          />
        ) : (
          <p className="text-gray-500 mb-4">
            Drop an image here or click Upload File
          </p>
        )}

        <input
          type="file"
          ref={fileInputRef}
          accept="image/*"
          onChange={(e) => {
            const selected = e.target.files[0];
            if (selected) {
              handleFileSelect(selected);
            }
          }}
          className="mb-4 hidden"
        />

        <div className="flex justify-center gap-4">
          <button
            onClick={() => fileInputRef.current.click()}
            className="cursor-pointer border px-4 py-2 rounded hover:bg-gray-100"
          >
            Upload File
          </button>

          <button
            onClick={handleAnalyze}
            className="cursor-pointer bg-teal-600 text-white px-6 py-2 rounded hover:bg-teal-700"
          >
            {loading ? "Analyzing..." : "Analyze"}
          </button>
        </div>
      </div>
    </div>
  );
}
