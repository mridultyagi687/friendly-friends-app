import React, { useEffect, useState } from 'react';
import api from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';

function Members() {
  const { user } = useAuth();
  const [members, setMembers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!user) {
      setLoading(false);
      return;
    }
    const fetchMembers = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await api.get('/api/members');
        setMembers(res.data?.members || []);
      } catch (e) {
        console.error('Failed to load members:', e);
        if (e.response?.status === 401) {
          setError('Please log in to view members.');
        } else {
          setError('Failed to load members');
        }
        setMembers([]);
      } finally {
        setLoading(false);
      }
    };
    fetchMembers();
  }, [user]);

  if (!user) {
    return (
      <div style={styles.container}>
        <div style={styles.loginPrompt}>
          <h1 style={styles.title}>Members</h1>
          <p style={styles.promptText}>Please log in to view members.</p>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Community Members</h1>
      {loading && <div style={styles.status}>Loading members...</div>}
      {error && (
        <div style={styles.error}>
          <span style={styles.errorIcon}>⚠️</span>
          {error}
        </div>
      )}

      <div style={styles.tableContainer}>
        {members.length === 0 && !loading ? (
          <div style={styles.emptyState}>No members found.</div>
        ) : (
          <div style={styles.table}>
            <div style={styles.headerRow}>
              <div style={styles.headerCell}>Username</div>
              <div style={styles.headerCell}>Email</div>
              <div style={styles.headerCell}>Role</div>
            </div>
            {members.map((m) => (
              <div key={m.id} style={styles.row}>
                <div style={styles.cell}>
                  <span style={styles.username}>{m.username}</span>
                </div>
                <div style={styles.cell}>{m.email}</div>
                <div style={styles.cell}>
                  <span style={m.is_admin ? styles.adminBadge : styles.memberBadge}>
                    {m.is_admin ? '👑 Admin' : '👤 Member'}
                  </span>
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
    maxWidth: '1200px',
    margin: '0 auto',
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)',
    backgroundSize: '400% 400%',
    animation: 'gradientShift 15s ease infinite',
  },
  loginPrompt: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    padding: '3rem',
    textAlign: 'center',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    border: '1px solid rgba(102, 126, 234, 0.3)',
  },
  title: {
    fontSize: '2.5rem',
    fontWeight: '700',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
    marginBottom: '2rem',
    textAlign: 'center',
  },
  promptText: {
    fontSize: '1.1rem',
    color: '#666',
  },
  status: {
    textAlign: 'center',
    color: 'rgba(255, 255, 255, 0.9)',
    margin: '1rem 0',
    fontSize: '1.1rem',
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
  tableContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    padding: '2rem',
    border: '1px solid rgba(102, 126, 234, 0.3)',
  },
  emptyState: {
    textAlign: 'center',
    padding: '3rem',
    color: '#999',
    fontSize: '1.1rem',
  },
  table: {
    display: 'grid',
    gridTemplateColumns: '1fr',
    gap: '0.75rem',
    overflow: 'hidden',
  },
  headerRow: {
    display: 'grid',
    gridTemplateColumns: '2fr 2fr 1fr',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    padding: '1.25rem 1rem',
    borderRadius: '12px',
    fontWeight: '600',
    fontSize: '0.95rem',
  },
  headerCell: {
    padding: '0 0.5rem',
  },
  row: {
    display: 'grid',
    gridTemplateColumns: '2fr 2fr 1fr',
    backgroundColor: 'white',
    padding: '1.25rem 1rem',
    borderRadius: '12px',
    border: '1px solid #f0f0f0',
    transition: 'all 0.2s ease',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)',
  },
  cell: {
    padding: '0 0.5rem',
    display: 'flex',
    alignItems: 'center',
  },
  username: {
    fontWeight: '600',
    color: '#333',
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
};

export default Members;
