export default function Navbar() {
  return (
    <div className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-4">
      <div>
        <h1 className="text-lg font-semibold text-slate-950">Chest Image Analysis</h1>
        <p className="text-sm text-slate-500">ViT feature extraction with FT-Transformer classification</p>
      </div>

      <div className="flex items-center gap-3">
        <div className="text-right">
          <p className="text-sm font-medium text-slate-900">Clinical AI Console</p>
          <p className="text-xs text-slate-500">Local inference</p>
        </div>
        <div className="flex size-10 items-center justify-center rounded-full bg-teal-50 ring-1 ring-teal-100">
          <img src="/download-removebg-preview.png" alt="" className="size-8 object-contain" />
        </div>
      </div>
    </div>
  );
}
