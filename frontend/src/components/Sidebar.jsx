export default function Sidebar() {
  return (
    <aside className="flex w-64 shrink-0 flex-col bg-slate-950 px-5 py-6 text-white">
      <div className="mb-10">
        <img src="/src/assets/logo3.png" alt="Application logo" className="max-h-20 object-contain" />
      </div>

      <nav className="space-y-2">
        <div className="rounded-lg bg-white/10 px-3 py-2 text-sm font-medium">Dashboard</div>
        <div className="px-3 py-2 text-sm text-slate-400">Case Review</div>
        <div className="px-3 py-2 text-sm text-slate-400">Model Status</div>
      </nav>

      <div className="mt-auto rounded-lg border border-white/10 bg-white/5 p-3">
        <p className="text-xs font-medium text-slate-200">Pipeline</p>
        <p className="mt-1 text-xs leading-5 text-slate-400">
          ViT embeddings are classified by the FT-Transformer model.
        </p>
      </div>
    </aside>
  );
}
