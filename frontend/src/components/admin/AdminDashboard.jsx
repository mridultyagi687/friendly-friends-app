import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';

function AdminDashboard() {
  const [users, setUsers] = useState([]);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showNewMemberForm, setShowNewMemberForm] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [passwordButtonHovered, setPasswordButtonHovered] = useState(false);
  const [newMember, setNewMember] = useState({
    username: '',
    email: '',
    password: '',
    is_admin: false
  });
  const { user } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!user?.is_admin) {
      navigate('/');
      return;
    }
    fetchUsers();
  }, [user, navigate]);

  const fetchUsers = async () => {
    setLoading(true);
    try {
      const response = await api.get('/api/members');
      setUsers(response.data?.members || []);
      setError(null);
      setSuccess(null);
    } catch (err) {
      console.error('Failed to fetch members:', err);
      if (err.response?.status === 401) {
        setError('Please log in as an admin to manage members.');
      } else {
        setError('Failed to fetch members: ' + (err.response?.data?.error || err.message));
      }
      setSuccess(null);
      setUsers([]);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteUser = async (userId) => {
    if (!window.confirm('Are you sure you want to delete this user?')) {
      return;
    }

    try {
      await api.delete(`/api/members/${userId}`);
      setUsers(users.filter(u => u.id !== userId));
      setError(null);
      setSuccess('Member deleted successfully.');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      const message = err.response?.data?.error || err.message || 'Failed to delete member';
      if (err.response?.status === 401) {
        setError('Please log in as an admin to delete members.');
      } else {
        setError('Failed to delete member: ' + message);
      }
      setSuccess(null);
    }
  };

  const handleCreateMember = async (e) => {
    e.preventDefault();
    try {
      setSuccess(null);
      setError(null);
      const response = await api.post('/api/members', newMember);
      const created = response.data?.user;
      if (created) {
        setUsers([created, ...users]);
        setSuccess(`Member "${created.username}" created successfully.`);
      } else {
        setSuccess('Member created.');
        await fetchUsers();
      }
      setNewMember({ username: '', email: '', password: '', is_admin: false });
      setShowPassword(false);
      setPasswordButtonHovered(false);
      setShowNewMemberForm(false);
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      const message = err.response?.data?.error || err.message || 'Failed to create member';
      if (err.response?.status === 401) {
        setError('Please log in as an admin to create members.');
      } else {
        setError('Failed to create member: ' + message);
      }
      setSuccess(null);
    }
  };

  if (!user?.is_admin) {
    return null;
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>Admin Dashboard</h1>
        <button 
          onClick={() => {
            setShowNewMemberForm(!showNewMemberForm);
            setShowPassword(false);
            setPasswordButtonHovered(false);
            setSuccess(null);
            setError(null);
          }} 
          style={showNewMemberForm ? styles.cancelButton : styles.newMemberButton}
        >
          {showNewMemberForm ? '✕ Cancel' : '+ New Member'}
        </button>
      </div>
      
      {error && (
        <div style={styles.error}>
          <span style={styles.errorIcon}>⚠️</span>
          {error}
        </div>
      )}
      {success && (
        <div style={styles.success}>
          <span style={styles.successIcon}>✓</span>
          {success}
        </div>
      )}

      {showNewMemberForm && (
        <form onSubmit={handleCreateMember} style={styles.newMemberForm}>
          <h3 style={styles.formTitle}>Create New Member</h3>
          <div style={styles.formGrid}>
            <div style={styles.inputGroup}>
              <label style={styles.label}>Username</label>
              <input
                type="text"
                placeholder="Enter username"
                value={newMember.username}
                onChange={(e) => setNewMember({...newMember, username: e.target.value})}
                style={styles.formInput}
                required
              />
            </div>
            <div style={styles.inputGroup}>
              <label style={styles.label}>Email</label>
              <input
                type="email"
                placeholder="Enter email"
                value={newMember.email}
                onChange={(e) => setNewMember({...newMember, email: e.target.value})}
                style={styles.formInput}
                required
              />
            </div>
            <div style={styles.inputGroup}>
              <label style={styles.label}>Password</label>
              <div style={styles.passwordInputContainer}>
                <input
                  type={showPassword ? "text" : "password"}
                  placeholder="Enter password"
                  value={newMember.password}
                  onChange={(e) => setNewMember({...newMember, password: e.target.value})}
                  style={styles.passwordInput}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  onMouseEnter={() => setPasswordButtonHovered(true)}
                  onMouseLeave={() => setPasswordButtonHovered(false)}
                  style={{
                    ...styles.showPasswordButton,
                    opacity: passwordButtonHovered ? 1 : 0.7,
                    backgroundColor: passwordButtonHovered ? 'rgba(102, 126, 234, 0.1)' : 'transparent',
                  }}
                  title={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? "🙈" : "👁️"}
                </button>
              </div>
            </div>
            <div style={styles.checkboxGroup}>
              <label style={styles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={newMember.is_admin}
                  onChange={(e) => setNewMember({...newMember, is_admin: e.target.checked})}
                  style={styles.checkbox}
                />
                <span style={styles.checkboxText}>Grant admin privileges</span>
              </label>
            </div>
          </div>
          <div style={styles.formActions}>
            <button type="submit" style={styles.submitButton}>
              Create Member
            </button>
          </div>
        </form>
      )}
      
      <div style={styles.tableContainer}>
        <h2 style={styles.tableTitle}>Members ({users.length})</h2>
        {loading ? (
          <div style={styles.loading}>Loading members...</div>
        ) : users.length === 0 ? (
          <div style={styles.emptyState}>No members found.</div>
        ) : (
          <div style={styles.usersTable}>
            <div style={styles.tableHeader}>
              <div style={styles.headerCell}>Username</div>
              <div style={styles.headerCell}>Email</div>
              <div style={styles.headerCell}>Role</div>
              <div style={styles.headerCell}>Actions</div>
            </div>
            
            {users.map(u => (
              <div key={u.id} style={styles.tableRow}>
                <div style={styles.cell}>{u.username}</div>
                <div style={styles.cell}>{u.email}</div>
                <div style={styles.cell}>
                  <span style={u.is_admin ? styles.adminBadge : styles.memberBadge}>
                    {u.is_admin ? 'Admin' : 'Member'}
                  </span>
                </div>
                <div style={styles.cell}>
                  {user?.id !== u.id && (
                    <button
                      onClick={() => handleDeleteUser(u.id)}
                      style={styles.deleteButton}
                      title="Delete member"
                    >
                      🗑️ Delete
                    </button>
                  )}
                  {user?.id === u.id && (
                    <span style={styles.currentUserLabel}>You</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

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
  newMemberButton: {
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
  cancelButton: {
    padding: '0.875rem 1.75rem',
    background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '12px',
    cursor: 'pointer',
    fontSize: '1rem',
    fontWeight: '600',
    boxShadow: '0 4px 15px rgba(245, 87, 108, 0.4)',
    transition: 'all 0.3s ease',
  },
  error: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    color: '#d32f2f',
    padding: '1rem 1.25rem',
    marginBottom: '1.5rem',
    backgroundColor: 'rgba(255, 235, 238, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '12px',
    border: '1px solid rgba(211, 47, 47, 0.2)',
    boxShadow: '0 4px 15px rgba(211, 47, 47, 0.1)',
  },
  errorIcon: {
    fontSize: '1.25rem',
  },
  success: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    color: '#2e7d32',
    padding: '1rem 1.25rem',
    marginBottom: '1.5rem',
    backgroundColor: 'rgba(232, 245, 233, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '12px',
    border: '1px solid rgba(46, 125, 50, 0.2)',
    boxShadow: '0 4px 15px rgba(46, 125, 50, 0.1)',
  },
  successIcon: {
    fontSize: '1.25rem',
    fontWeight: 'bold',
  },
  newMemberForm: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    padding: '2rem',
    borderRadius: '20px',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    marginBottom: '2rem',
    border: '1px solid rgba(102, 126, 234, 0.3)',
  },
  formTitle: {
    marginBottom: '1.5rem',
    color: '#333',
    fontSize: '1.5rem',
    fontWeight: '600',
  },
  formGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
    gap: '1.5rem',
    marginBottom: '1.5rem',
  },
  inputGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
  },
  label: {
    fontSize: '0.9rem',
    fontWeight: '500',
    color: '#555',
  },
  formInput: {
    padding: '0.875rem 1rem',
    fontSize: '1rem',
    border: '2px solid rgba(102, 126, 234, 0.3)',
    borderRadius: '10px',
    transition: 'all 0.3s ease',
    backgroundColor: 'white',
  },
  passwordInputContainer: {
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
  },
  passwordInput: {
    padding: '0.875rem 3rem 0.875rem 1rem',
    fontSize: '1rem',
    border: '2px solid rgba(102, 126, 234, 0.3)',
    borderRadius: '10px',
    transition: 'all 0.3s ease',
    backgroundColor: 'white',
    width: '100%',
    boxSizing: 'border-box',
  },
  showPasswordButton: {
    position: 'absolute',
    right: '0.5rem',
    top: '50%',
    transform: 'translateY(-50%)',
    background: 'transparent',
    border: 'none',
    cursor: 'pointer',
    fontSize: '1.5rem',
    padding: '0.5rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: '6px',
    transition: 'all 0.2s ease',
    minWidth: '36px',
    minHeight: '36px',
    zIndex: 10,
    color: '#666',
    lineHeight: 1,
  },
  checkboxGroup: {
    gridColumn: '1 / -1',
  },
  checkboxLabel: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    cursor: 'pointer',
    padding: '0.75rem',
    borderRadius: '10px',
    transition: 'background-color 0.2s',
  },
  checkbox: {
    width: '20px',
    height: '20px',
    cursor: 'pointer',
    accentColor: '#667eea',
  },
  checkboxText: {
    fontSize: '1rem',
    color: '#333',
    fontWeight: '500',
  },
  formActions: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: '1rem',
  },
  submitButton: {
    padding: '0.875rem 2rem',
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
  tableContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    padding: '2rem',
    border: '1px solid rgba(102, 126, 234, 0.3)',
  },
  tableTitle: {
    marginBottom: '1.5rem',
    color: '#333',
    fontSize: '1.5rem',
    fontWeight: '600',
  },
  loading: {
    textAlign: 'center',
    padding: '2rem',
    color: '#666',
    fontSize: '1.1rem',
  },
  emptyState: {
    textAlign: 'center',
    padding: '3rem',
    color: '#999',
    fontSize: '1.1rem',
  },
  usersTable: {
    overflow: 'hidden',
    borderRadius: '12px',
    border: '1px solid rgba(102, 126, 234, 0.3)',
  },
  tableHeader: {
    display: 'grid',
    gridTemplateColumns: '2fr 2fr 1fr 1.5fr',
    padding: '1.25rem 1rem',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    fontWeight: '600',
    fontSize: '0.95rem',
  },
  tableRow: {
    display: 'grid',
    gridTemplateColumns: '2fr 2fr 1fr 1.5fr',
    padding: '1.25rem 1rem',
    borderBottom: '1px solid #f0f0f0',
    transition: 'background-color 0.2s',
    backgroundColor: 'white',
  },
  headerCell: {
    padding: '0 0.5rem',
  },
  cell: {
    padding: '0 0.5rem',
    display: 'flex',
    alignItems: 'center',
  },
  adminBadge: {
    display: 'inline-block',
    padding: '0.375rem 0.875rem',
    borderRadius: '20px',
    fontSize: '0.85rem',
    fontWeight: '600',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
  },
  memberBadge: {
    display: 'inline-block',
    padding: '0.375rem 0.875rem',
    borderRadius: '20px',
    fontSize: '0.85rem',
    fontWeight: '600',
    backgroundColor: '#e0e0e0',
    color: '#666',
  },
  deleteButton: {
    padding: '0.5rem 1rem',
    background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '0.9rem',
    fontWeight: '500',
    transition: 'all 0.3s ease',
    boxShadow: '0 2px 8px rgba(245, 87, 108, 0.3)',
  },
  currentUserLabel: {
    color: '#999',
    fontSize: '0.9rem',
    fontStyle: 'italic',
  },
};

export default AdminDashboard;
