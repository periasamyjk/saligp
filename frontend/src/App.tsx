import { useState } from 'react';
import { Menu, X } from 'lucide-react';
import Dashboard from './pages/Dashboard';
import Predictions from './pages/Predictions';
import Analytics from './pages/Analytics';
import './App.css';

type Page = 'dashboard' | 'predictions' | 'analytics';

function App() {
  const [currentPage, setCurrentPage] = useState<Page>('dashboard');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: '📊' },
    { id: 'predictions', label: 'Predictions', icon: '🔮' },
    { id: 'analytics', label: 'Analytics', icon: '📈' },
  ];

  return (
    <div className="flex h-screen bg-slate-950">
      {/* Sidebar */}
      <aside className="hidden md:flex md:w-64 bg-slate-900 border-r border-slate-700 flex-col">
        <div className="p-6 border-b border-slate-700">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            SALIGP
          </h1>
          <p className="text-xs text-slate-400 mt-1">Duplicate Detection Intelligence</p>
        </div>

        <nav className="flex-1 p-4 space-y-2">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setCurrentPage(item.id as Page)}
              className={`w-full text-left px-4 py-3 rounded-lg transition-all ${
                currentPage === item.id
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20'
                  : 'text-slate-300 hover:bg-slate-800'
              }`}
            >
              <span className="mr-2">{item.icon}</span>
              {item.label}
            </button>
          ))}
        </nav>

        <div className="p-4 border-t border-slate-700 text-xs text-slate-500">
          <p>v1.0.0</p>
          <p className="mt-1">© 2024 SALIGP</p>
        </div>
      </aside>

      {/* Mobile Header */}
      <div className="md:hidden fixed top-0 left-0 right-0 h-16 bg-slate-900 border-b border-slate-700 flex items-center px-4 z-50">
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="p-2 hover:bg-slate-800 rounded-lg"
        >
          {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
        <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent ml-3">
          SALIGP
        </h1>
      </div>

      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div className="md:hidden fixed top-16 left-0 right-0 bg-slate-900 border-b border-slate-700 z-40">
          <nav className="p-4 space-y-2">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => {
                  setCurrentPage(item.id as Page);
                  setMobileMenuOpen(false);
                }}
                className={`w-full text-left px-4 py-3 rounded-lg transition-all ${
                  currentPage === item.id
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-300 hover:bg-slate-800'
                }`}
              >
                <span className="mr-2">{item.icon}</span>
                {item.label}
              </button>
            ))}
          </nav>
        </div>
      )}

      {/* Main Content */}
      <main className="flex-1 overflow-auto mt-16 md:mt-0">
        {currentPage === 'dashboard' && <Dashboard />}
        {currentPage === 'predictions' && <Predictions />}
        {currentPage === 'analytics' && <Analytics />}
      </main>
    </div>
  );
}

export default App;
