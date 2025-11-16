import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';

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
    marginBottom: '2rem',
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
  content: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '2rem',
    marginTop: '2rem',
  },
  panel: {
    background: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    padding: '2rem',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    border: '1px solid rgba(102, 126, 234, 0.3)',
  },
  panelTitle: {
    fontSize: '1.5rem',
    fontWeight: '600',
    color: '#667eea',
    marginBottom: '1.5rem',
  },
  userList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
  },
  userItem: {
    padding: '1rem',
    background: 'rgba(102, 126, 234, 0.05)',
    borderRadius: '10px',
    border: '1px solid rgba(102, 126, 234, 0.2)',
  },
  userName: {
    fontWeight: '600',
    color: '#333',
    marginBottom: '0.5rem',
  },
  userRoles: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '0.5rem',
    marginTop: '0.5rem',
  },
  roleBadge: {
    padding: '0.25rem 0.75rem',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    borderRadius: '20px',
    fontSize: '0.85rem',
    fontWeight: '500',
  },
  roleList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
  },
  roleItem: {
    padding: '1rem',
    background: 'rgba(102, 126, 234, 0.05)',
    borderRadius: '10px',
    border: '1px solid rgba(102, 126, 234, 0.2)',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  roleInfo: {
    flex: 1,
  },
  roleName: {
    fontWeight: '600',
    color: '#667eea',
    marginBottom: '0.25rem',
  },
  assignButton: {
    padding: '0.5rem 1rem',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '0.9rem',
    fontWeight: '500',
    transition: 'all 0.3s ease',
  },
  removeButton: {
    padding: '0.5rem 1rem',
    background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '0.9rem',
    fontWeight: '500',
    transition: 'all 0.3s ease',
  },
  loading: {
    textAlign: 'center',
    padding: '2rem',
    color: '#666',
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
  success: {
    background: 'rgba(232, 245, 233, 0.95)',
    backdropFilter: 'blur(10px)',
    padding: '1rem 1.25rem',
    borderRadius: '12px',
    border: '1px solid rgba(46, 125, 50, 0.2)',
    color: '#2e7d32',
    marginBottom: '1.5rem',
  },
};

function RoleAssignment() {
  const { user, checkAuth } = useAuth();
  const [users, setUsers] = useState([]);
  const [roles, setRoles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [selectedUser, setSelectedUser] = useState(null);

  useEffect(() => {
    if (!user?.is_admin) {
      return;
    }
    fetchData();
  }, [user]);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [usersRes, rolesRes] = await Promise.all([
        api.get('/api/members'),
        api.get('/api/roles'),
      ]);
      setUsers(usersRes.data?.members || []);
      setRoles(rolesRes.data?.roles || []);
    } catch (err) {
      console.error('Failed to fetch data:', err);
      setError(err.response?.data?.error || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  const handleAssignRole = async (userId, roleId) => {
    try {
      setError(null);
      setSuccess(null);
      await api.post(`/api/roles/${roleId}/assign`, { user_id: userId });
      setSuccess('Role assigned successfully');
      setTimeout(() => setSuccess(null), 3000);
      await fetchData();
      
      // If role was assigned to the current user, refresh their auth data
      if (user && user.id === userId) {
        await checkAuth();
      }
    } catch (err) {
      console.error('Failed to assign role:', err);
      setError(err.response?.data?.error || 'Failed to assign role');
    }
  };

  const handleRemoveRole = async (userId, roleId) => {
    try {
      setError(null);
      setSuccess(null);
      // DELETE endpoint expects JSON body
      await api.delete(`/api/roles/${roleId}/assign`, { data: { user_id: userId } });
      setSuccess('Role removed successfully');
      setTimeout(() => setSuccess(null), 3000);
      await fetchData();
      
      // If role was removed from the current user, refresh their auth data
      if (user && user.id === userId) {
        await checkAuth();
      }
    } catch (err) {
      console.error('Failed to remove role:', err);
      setError(err.response?.data?.error || 'Failed to remove role');
    }
  };

  if (!user?.is_admin) {
    return (
      <div style={styles.container}>
        <div style={styles.error}>Admin access required</div>
      </div>
    );
  }

  if (loading) {
    return (
      <div style={styles.container}>
        <div style={styles.loading}>Loading...</div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>⚙️ Role Assignment</h1>
      </div>

      {error && (
        <div style={styles.error}>
          <span style={{ marginRight: '0.5rem' }}>⚠️</span>
          {error}
        </div>
      )}

      {success && (
        <div style={styles.success}>
          <span style={{ marginRight: '0.5rem' }}>✓</span>
          {success}
        </div>
      )}

      <div style={styles.content}>
        <div style={styles.panel}>
          <h2 style={styles.panelTitle}>👥 Users</h2>
          <div style={styles.userList}>
            {users.map((u) => (
              <div
                key={u.id}
                style={{
                  ...styles.userItem,
                  ...(selectedUser === u.id ? { border: '2px solid #667eea' } : {}),
                }}
                onClick={() => setSelectedUser(u.id)}
              >
                <div style={styles.userName}>{u.username}</div>
                <div style={styles.userRoles}>
                  {u.roles && u.roles.length > 0 ? (
                    u.roles.map((role) => (
                      <span key={role.id} style={styles.roleBadge}>
                        {role.name}
                      </span>
                    ))
                  ) : (
                    <span style={{ color: '#999', fontSize: '0.9rem' }}>No roles assigned</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div style={styles.panel}>
          <h2 style={styles.panelTitle}>🎭 Available Roles</h2>
          {selectedUser ? (
            <div style={styles.roleList}>
              {roles.map((role) => {
                const user = users.find((u) => u.id === selectedUser);
                const hasRole = user?.roles?.some((r) => r.id === role.id);
                return (
                  <div key={role.id} style={styles.roleItem}>
                    <div style={styles.roleInfo}>
                      <div style={styles.roleName}>{role.name}</div>
                      {role.description && (
                        <div style={{ fontSize: '0.85rem', color: '#666' }}>
                          {role.description}
                        </div>
                      )}
                    </div>
                    {hasRole ? (
                      <button
                        onClick={() => handleRemoveRole(selectedUser, role.id)}
                        style={styles.removeButton}
                      >
                        Remove
                      </button>
                    ) : (
                      <button
                        onClick={() => handleAssignRole(selectedUser, role.id)}
                        style={styles.assignButton}
                      >
                        Assign
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '2rem', color: '#999' }}>
              Select a user to assign roles
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default RoleAssignment;

