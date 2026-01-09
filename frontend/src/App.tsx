/**Main App component with routing.

Simple routing - no nested routes, no guards.
*/

import { BrowserRouter, Link, Route, Routes } from 'react-router-dom';
import { RunDetailPage } from './pages/RunDetailPage';
import { RunPage } from './pages/RunPage';
import { RunsListPage } from './pages/RunsListPage';

function App() {
  return (
    <BrowserRouter>
      <div>
        <nav
          style={{
            background: '#f8f9fa',
            padding: '1rem',
            borderBottom: '1px solid #dee2e6',
            marginBottom: '2rem',
          }}
        >
          <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', gap: '2rem' }}>
            <Link
              to="/"
              style={{ color: '#007bff', textDecoration: 'none', fontWeight: 'bold' }}
            >
              Create Run
            </Link>
            <Link
              to="/runs"
              style={{ color: '#007bff', textDecoration: 'none', fontWeight: 'bold' }}
            >
              Run History
            </Link>
          </div>
        </nav>

        <Routes>
          <Route path="/" element={<RunPage />} />
          <Route path="/runs" element={<RunsListPage />} />
          <Route path="/runs/:runId" element={<RunDetailPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
