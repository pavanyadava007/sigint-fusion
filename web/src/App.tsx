import { Navigate, Route, Routes } from 'react-router-dom';
import { HealthProvider } from './components/HealthContext';
import { Layout } from './components/Layout';
import { Analyst } from './pages/Analyst';
import { Emitters } from './pages/Emitters';
import { Overview } from './pages/Overview';
import { SignalLab } from './pages/SignalLab';
import { System } from './pages/System';

export function App() {
  return (
    <HealthProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="emitters" element={<Emitters />} />
          <Route path="analyst" element={<Analyst />} />
          <Route path="lab" element={<SignalLab />} />
          <Route path="system" element={<System />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </HealthProvider>
  );
}
