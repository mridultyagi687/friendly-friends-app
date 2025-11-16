import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import NavBar from './components/NavBar';
import Login from './components/auth/Login';
import TodoList from './components/todo/TodoList';
import Paint from './components/paint/Paint';
import Members from './components/members/Members';
import Messages from './components/messages/Messages';
import VideoGallery from './components/video/VideoGallery';
import AiChat from './components/ai/AiChat';
import AiDocs from './components/ai/AiDocs';
import AdminDashboard from './components/admin/AdminDashboard';
import AiTraining from './components/admin/AiTraining';
import Blog from './components/blog/Blog';
import Roles from './components/roles/Roles';
import RoleAssignment from './components/roles/RoleAssignment';
import CloudPCs from './components/cloudpc/CloudPCs';
import CloudPCViewer from './components/cloudpc/CloudPCViewer';
import AppTour from './components/AppTour';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { CallProvider } from './contexts/CallContext';
import { hasRole, hasAnyRole } from './utils/roleUtils';
import { canAccessFeature } from './utils/roleEnforcement';

function FeatureGuard({ children, feature, fallback = '/videos' }) {
  const { user } = useAuth();
  if (!canAccessFeature(user, feature)) {
    return <Navigate to={fallback} replace />;
  }
  return children;
}

function ProtectedRoute({ children, allowedRoles, denyRoles = [], requireAdmin = false }) {
  const { user, loading } = useAuth();
  if (loading) return <div>Loading...</div>;
  if (!user) return <Navigate to="/" replace />;

  // Check for denied roles (new role system)
  if (denyRoles.length > 0 && hasAnyRole(user, denyRoles)) {
    // Check if user has any denied role
    const hasDeniedRole = denyRoles.some(role => hasRole(user, role));
    if (hasDeniedRole) {
      // Redirect based on user's first role or default
      const userRoles = user.roles || [];
      if (userRoles.length > 0) {
        return <Navigate to="/videos" replace />;
      }
      return <Navigate to="/videos" replace />;
    }
  }

  // Check admin requirement
  if (requireAdmin && !user.is_admin) {
    return <Navigate to="/videos" replace />;
  }
  
  // Check allowed roles (new role system)
  if (allowedRoles && allowedRoles.length > 0) {
    const hasAllowedRole = hasAnyRole(user, allowedRoles);
    if (!hasAllowedRole) {
      // User doesn't have any of the required roles
      return <Navigate to={user.is_admin ? '/admin' : '/videos'} replace />;
    }
  }
  
  return children;
}

function AppRoutes() {
  const { user, loading } = useAuth();
  if (loading) return <div>Loading...</div>;

  // Get base path for GitHub Pages (e.g., /repo-name/)
  // This is set by Vite during build via import.meta.env.BASE_URL
  const basename = import.meta.env.BASE_URL || '/';

  return (
    <Router basename={basename} future={{ v7_relativeSplatPath: true }}>
      <div className="app">
        {user && <NavBar />}
        <div style={{ marginLeft: user ? '250px' : '0', minHeight: '100vh' }}>
          <Routes>
            <Route
              path="/"
              element={
                user
                  ? (canAccessFeature(user, 'blog') && !canAccessFeature(user, 'members')
                      ? <Navigate to="/blog" replace />
                      : (user.is_admin ? <Navigate to="/admin" replace /> : <Navigate to="/members" replace />))
                  : <Login />
              }
            />
            <Route
              path="/blog"
              element={
                <ProtectedRoute>
                  <FeatureGuard feature="blog" fallback="/videos">
                    <Blog />
                  </FeatureGuard>
                </ProtectedRoute>
              }
            />
            <Route
              path="/blog/:blogId"
              element={
                <ProtectedRoute>
                  <FeatureGuard feature="blog" fallback="/videos">
                    <Blog />
                  </FeatureGuard>
                </ProtectedRoute>
              }
            />
            <Route
              path="/todos"
              element={
                <ProtectedRoute>
                  <FeatureGuard feature="todos" fallback="/videos">
                    <TodoList />
                  </FeatureGuard>
                </ProtectedRoute>
              }
            />
            <Route
              path="/paint"
              element={
                <ProtectedRoute>
                  <FeatureGuard feature="paint" fallback="/videos">
                    <Paint />
                  </FeatureGuard>
                </ProtectedRoute>
              }
            />
            <Route
              path="/members"
              element={
                <ProtectedRoute>
                  <FeatureGuard feature="members" fallback="/videos">
                    <Members />
                  </FeatureGuard>
                </ProtectedRoute>
              }
            />
            <Route
              path="/messages"
              element={
                <ProtectedRoute>
                  <FeatureGuard feature="messages" fallback="/videos">
                    <Messages />
                  </FeatureGuard>
                </ProtectedRoute>
              }
            />
            <Route
              path="/videos"
              element={
                <ProtectedRoute>
                  <FeatureGuard feature="videos" fallback="/blog">
                    <VideoGallery />
                  </FeatureGuard>
                </ProtectedRoute>
              }
            />
            <Route
              path="/ai-chat"
              element={
                <ProtectedRoute>
                  <FeatureGuard feature="ai-chat" fallback="/videos">
                    <AiChat />
                  </FeatureGuard>
                </ProtectedRoute>
              }
            />
            <Route
              path="/docs"
              element={
                <ProtectedRoute>
                  <FeatureGuard feature="docs" fallback="/videos">
                    <AiDocs />
                  </FeatureGuard>
                </ProtectedRoute>
              }
            />
            <Route
              path="/admin"
              element={<ProtectedRoute requireAdmin><AdminDashboard /></ProtectedRoute>}
            />
            <Route
              path="/admin/ai-training"
              element={<ProtectedRoute requireAdmin><AiTraining /></ProtectedRoute>}
            />
            <Route
              path="/roles"
              element={<ProtectedRoute><Roles /></ProtectedRoute>}
            />
            <Route
              path="/admin/role-assignment"
              element={<ProtectedRoute requireAdmin><RoleAssignment /></ProtectedRoute>}
            />
            <Route
              path="/cloud-pcs"
              element={
                <ProtectedRoute>
                  <CloudPCs />
                </ProtectedRoute>
              }
            />
            <Route
              path="/cloud-pcs/:pcId"
              element={
                <ProtectedRoute>
                  <CloudPCViewer />
                </ProtectedRoute>
              }
            />
          </Routes>
        </div>
        {user && <AppTour />}
      </div>
    </Router>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <CallProvider>
        <AppRoutes />
      </CallProvider>
    </AuthProvider>
  );
}