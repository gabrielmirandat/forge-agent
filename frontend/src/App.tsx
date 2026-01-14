/**Main App component with routing.

Simple routing - no nested routes, no guards.
*/

import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { ChatPage } from './pages/ChatPage';
import { RunDetailPage } from './pages/RunDetailPage';
import { RunPage } from './pages/RunPage';
import { RunsListPage } from './pages/RunsListPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ChatPage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/chat/:sessionId" element={<ChatPage />} />
        {/* Legacy routes - kept for backward compatibility */}
        <Route path="/run" element={<RunPage />} />
        <Route path="/runs" element={<RunsListPage />} />
        <Route path="/runs/:runId" element={<RunDetailPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
