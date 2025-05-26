import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Box, CircularProgress } from '@mui/material';
import { AnimatePresence } from 'framer-motion';

import { useAuth } from './contexts/AuthContext';
import Layout from './components/Layout/Layout';
import LoadingScreen from './components/Common/LoadingScreen';

// Lazy load pages for better performance
const LoginPage = lazy(() => import('./pages/Auth/LoginPage'));
const RegisterPage = lazy(() => import('./pages/Auth/RegisterPage'));
const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const TradingPage = lazy(() => import('./pages/Trading/TradingPage'));
const PortfolioPage = lazy(() => import('./pages/Portfolio/PortfolioPage'));
const AnalyticsPage = lazy(() => import('./pages/Analytics/AnalyticsPage'));
const RiskManagementPage = lazy(() => import('./pages/RiskManagement/RiskManagementPage'));
const SettingsPage = lazy(() => import('./pages/Settings/SettingsPage'));
const ProfilePage = lazy(() => import('./pages/Profile/ProfilePage'));

// Loading fallback component
const PageLoader: React.FC = () => (
  <Box
    display="flex"
    justifyContent="center"
    alignItems="center"
    minHeight="60vh"
  >
    <CircularProgress />
  </Box>
);

// Protected Route component
interface ProtectedRouteProps {
  children: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return <LoadingScreen />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
};

// Public Route component (redirects to dashboard if authenticated)
const PublicRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return <LoadingScreen />;
  }

  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
};

const App: React.FC = () => {
  return (
    <AnimatePresence mode="wait">
      <Suspense fallback={<LoadingScreen />}>
        <Routes>
          {/* Public routes */}
          <Route
            path="/login"
            element={
              <PublicRoute>
                <LoginPage />
              </PublicRoute>
            }
          />
          <Route
            path="/register"
            element={
              <PublicRoute>
                <RegisterPage />
              </PublicRoute>
            }
          />

          {/* Protected routes */}
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Layout />
              </ProtectedRoute>
            }
          >
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route
              path="dashboard"
              element={
                <Suspense fallback={<PageLoader />}>
                  <DashboardPage />
                </Suspense>
              }
            />
            <Route
              path="trading"
              element={
                <Suspense fallback={<PageLoader />}>
                  <TradingPage />
                </Suspense>
              }
            />
            <Route
              path="portfolio"
              element={
                <Suspense fallback={<PageLoader />}>
                  <PortfolioPage />
                </Suspense>
              }
            />
            <Route
              path="analytics"
              element={
                <Suspense fallback={<PageLoader />}>
                  <AnalyticsPage />
                </Suspense>
              }
            />
            <Route
              path="risk-management"
              element={
                <Suspense fallback={<PageLoader />}>
                  <RiskManagementPage />
                </Suspense>
              }
            />
            <Route
              path="settings"
              element={
                <Suspense fallback={<PageLoader />}>
                  <SettingsPage />
                </Suspense>
              }
            />
            <Route
              path="profile"
              element={
                <Suspense fallback={<PageLoader />}>
                  <ProfilePage />
                </Suspense>
              }
            />
          </Route>

          {/* Catch all route */}
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </Suspense>
    </AnimatePresence>
  );
};

export default App;
