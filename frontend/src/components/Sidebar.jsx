export default function Sidebar() {
  return (
    <div className="w-64 bg-linear-to-b from-blue-900 to-slate-900 text-white p-5 ">
      <img src="/src/assets/logo3.png" className="" />
      <br /><br /><br />
      <ul className="space-y-4">
        <li className="bg-white/10 p-2 rounded">Dashboard</li>
        <li className="opacity-70">My Cases</li>
        <li className="opacity-70">Patients</li>
        <li className="opacity-70">Reports</li>
        <li className="opacity-70">Settings</li>
      </ul>
    </div>
  );
}
