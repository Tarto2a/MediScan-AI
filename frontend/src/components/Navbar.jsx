export default function Navbar() {
  return (
    <div className="flex justify-between items-center bg-white p-4 shadow">
      <h1 className="text-lg font-semibold">Dashboard</h1>

      <div className="flex items-center gap-4">
        <button className="bg-teal-600 text-white px-4 py-2 rounded">
          Upload Medical Image
        </button>
        <div className="size-8 rounded-full"><img src="/public/download-removebg-preview.png" alt="" /></div>
      </div>
    </div>
  );
}