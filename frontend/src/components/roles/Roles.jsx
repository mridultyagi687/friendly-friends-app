import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';
import AIRolePopup from './AIRolePopup';

const styles = {
  container: {
    padding: '2rem',
    maxWidth: '1400px',
    margin: '0 auto',
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)',
    backgroundSize: '400% 400%',
    animation: 'gradientShift 15s ease infinite',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '2rem',
    flexWrap: 'wrap',
    gap: '1rem',
  },
  title: {
    fontSize: '2.5rem',
    fontWeight: '700',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
    margin: 0,
  },
  createButton: {
    padding: '0.875rem 1.75rem',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '12px',
    cursor: 'pointer',
    fontSize: '1rem',
    fontWeight: '600',
    boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
    transition: 'all 0.3s ease',
  },
  rolesGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
    gap: '1.5rem',
    marginTop: '2rem',
  },
  roleCard: {
    background: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    padding: '1.5rem',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    border: '1px solid rgba(102, 126, 234, 0.3)',
    transition: 'transform 0.3s ease, box-shadow 0.3s ease',
  },
  roleCardHover: {
    transform: 'translateY(-5px)',
    boxShadow: '0 25px 70px rgba(0, 0, 0, 0.4)',
  },
  roleName: {
    fontSize: '1.5rem',
    fontWeight: '700',
    color: '#667eea',
    marginBottom: '0.5rem',
  },
  roleDescription: {
    color: '#666',
    marginBottom: '1rem',
    lineHeight: '1.6',
  },
  roleInstructions: {
    background: 'rgba(102, 126, 234, 0.1)',
    padding: '1rem',
    borderRadius: '10px',
    marginTop: '1rem',
    border: '1px solid rgba(102, 126, 234, 0.2)',
  },
  instructionsLabel: {
    fontSize: '0.9rem',
    fontWeight: '600',
    color: '#667eea',
    marginBottom: '0.5rem',
  },
  instructionsText: {
    color: '#555',
    fontSize: '0.9rem',
    lineHeight: '1.6',
    whiteSpace: 'pre-wrap',
  },
  roleMeta: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: '1rem',
    paddingTop: '1rem',
    borderTop: '1px solid rgba(102, 126, 234, 0.2)',
    fontSize: '0.85rem',
    color: '#999',
  },
  userCount: {
    fontWeight: '600',
    color: '#667eea',
  },
  loading: {
    textAlign: 'center',
    padding: '3rem',
    color: 'white',
    fontSize: '1.2rem',
  },
  emptyState: {
    textAlign: 'center',
    padding: '3rem',
    background: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    color: '#666',
    fontSize: '1.1rem',
  },
  error: {
    background: 'rgba(255, 235, 238, 0.95)',
    backdropFilter: 'blur(10px)',
    padding: '1rem 1.25rem',
    borderRadius: '12px',
    border: '1px solid rgba(211, 47, 47, 0.2)',
    color: '#d32f2f',
    marginBottom: '1.5rem',
  },
};

function Roles() {
  const { user } = useAuth();
  const [roles, setRoles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [hoveredCard, setHoveredCard] = useState(null);

  useEffect(() => {
    fetchRoles();
  }, []);

  const fetchRoles = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.get('/api/roles');
      setRoles(response.data?.roles || []);
    } catch (err) {
      console.error('Failed to fetch roles:', err);
      setError(err.response?.data?.error || 'Failed to load roles');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateRole = async (roleData) => {
    try {
      const response = await api.post('/api/roles', roleData);
      setRoles([response.data.role, ...roles]);
      setShowCreateModal(false);
      setError(null);
    } catch (err) {
      console.error('Failed to create role:', err);
      setError(err.response?.data?.error || 'Failed to create role');
    }
  };

  if (loading) {
    return (
      <div style={styles.container}>
        <div style={styles.loading}>Loading roles...</div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>👥 Roles</h1>
        <button
          onClick={() => setShowCreateModal(true)}
          style={styles.createButton}
        >
          + Create Role
        </button>
      </div>

      {error && (
        <div style={styles.error}>
          <span style={{ marginRight: '0.5rem' }}>⚠️</span>
          {error}
        </div>
      )}

      {roles.length === 0 ? (
        <div style={styles.emptyState}>
          <p>No roles created yet.</p>
          <p style={{ marginTop: '1rem' }}>Click "Create Role" to get started!</p>
        </div>
      ) : (
        <div style={styles.rolesGrid}>
          {roles.map((role) => (
            <div
              key={role.id}
              style={{
                ...styles.roleCard,
                ...(hoveredCard === role.id ? styles.roleCardHover : {}),
              }}
              onMouseEnter={() => setHoveredCard(role.id)}
              onMouseLeave={() => setHoveredCard(null)}
            >
              <h3 style={styles.roleName}>{role.name}</h3>
              {role.description && (
                <p style={styles.roleDescription}>{role.description}</p>
              )}
              {role.ai_instructions && (
                <div style={styles.roleInstructions}>
                  <div style={styles.instructionsLabel}>🤖 AI Instructions:</div>
                  <div style={styles.instructionsText}>{role.ai_instructions}</div>
                </div>
              )}
              <div style={styles.roleMeta}>
                <span>{new Date(role.created_at).toLocaleDateString()}</span>
                <span style={styles.userCount}>
                  {role.user_count || 0} {role.user_count === 1 ? 'user' : 'users'}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {showCreateModal && (
        <AIRolePopup
          onClose={() => setShowCreateModal(false)}
          onCreate={handleCreateRole}
        />
      )}
    </div>
  );
}

export default Roles;

