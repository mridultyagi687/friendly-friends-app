import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import api from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const checkAuth = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get('/api/me');
      if (res.data && res.data.ok) {
        setUser(res.data.user);
      } else {
        setUser(null);
      }
    } catch (err) {
      // Don't set error for 401 (unauthorized) - user is just not logged in
      // Also don't show errors for network issues on initial load
      if (err.response) {
        if (err.response.status === 401) {
          // Not logged in - this is normal
          setError(null);
          setUser(null);
        } else if (err.response.status === 500) {
          // Server error - log but don't block the app
          console.error('Server error during auth check:', err.response.data);
          setError(null); // Don't show error on initial load
          setUser(null);
        } else {
          setError(err);
        }
      } else {
        // Network error - don't show on initial load
        console.warn('Network error during auth check (backend may be starting):', err.message);
        setError(null);
        setUser(null);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  // Global presence heartbeat while logged in (every 30s)
  useEffect(() => {
    if (user) {
      const sendHeartbeat = async () => {
        try {
          await api.post('/api/presence/update');
        } catch (e) {
          // ignore
        }
      };
      // send immediately then every 30s
      sendHeartbeat();
      const id = setInterval(() => {
        sendHeartbeat();
      }, 30000);
      return () => {
        clearInterval(id);
      };
    }
    return undefined;
  }, [user]);

  const login = async (username, password) => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.post('/api/login', { username, password });
      if (res.data && res.data.ok) {
        setUser(res.data.user);
        setError(null); // Clear any previous errors
        return true;
      }
      setError(new Error(res.data?.error || 'Invalid credentials'));
      return false;
    } catch (err) {
      const errorMessage = err.response?.data?.error || err.message || 'Request Failed';
      setError(new Error(errorMessage));
      return false;
    } finally {
      setLoading(false);
    }
  };

  const register = async (username, email, password, role = '') => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.post('/api/register', { username, email, password, role });
      if (res.data && res.data.ok) {
        setUser(res.data.user);
        setError(null); // Clear any previous errors
        return true;
      }
      setError(new Error(res.data?.error || 'Registration failed'));
      return false;
    } catch (err) {
      const errorMessage = err.response?.data?.error || err.message || 'Registration failed';
      setError(new Error(errorMessage));
      return false;
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    try {
      await api.post('/api/logout');
    } catch (err) {
      // ignore
    }
    setUser(null);
    setError(null);
  };

  const clearError = () => {
    setError(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, error, checkAuth, login, register, logout, clearError }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
