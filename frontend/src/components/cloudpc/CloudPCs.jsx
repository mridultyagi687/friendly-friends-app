import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import api from '../../services/api';
import './CloudPCs.css';

function CloudPCs() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [cloudPCs, setCloudPCs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [creating, setCreating] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(null);
  const [formData, setFormData] = useState({
    name: '',
    os_version: '1.0 beta'
  });

  useEffect(() => {
    fetchCloudPCs();
  }, []);

  const fetchCloudPCs = async () => {
    try {
      setLoading(true);
      const { data } = await api.get('/api/cloud-pcs');
      setCloudPCs(data.cloud_pcs || []);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to load cloud PCs');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!formData.name.trim()) {
      setError('Please provide a VM name');
      return;
    }

    setCreating(true);
    setError(null);
    setSuccess(null);

    try {
      const { data } = await api.post('/api/cloud-pcs', {
        name: formData.name.trim(),
        os_version: formData.os_version
      });

      setSuccess(data?.message || 'Cloud PC created successfully!');
      setFormData({ name: '', os_version: '1.0 beta' });
      setShowCreateForm(false);
      await fetchCloudPCs();
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      const message = err.response?.data?.error || 'Failed to create cloud PC';
      setError(message);
    } finally {
      setCreating(false);
    }
  };

  const handleAccess = (pcId) => {
    navigate(`/cloud-pcs/${pcId}`);
  };

  const handleDelete = async (pcId) => {
    try {
      await api.delete(`/api/cloud-pcs/${pcId}`);
      setSuccess('Cloud PC deleted successfully');
      setShowDeleteConfirm(null);
      await fetchCloudPCs();
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to delete cloud PC');
      setShowDeleteConfirm(null);
    }
  };

  const getStatusStyle = (status) => {
    switch (status) {
      case 'running':
        return { color: '#4ade80', fontWeight: '600' };
      case 'stopped':
        return { color: '#fbbf24', fontWeight: '600' };
      default:
        return { color: '#94a3b8', fontWeight: '600' };
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>💻 My Cloud PCs</h1>
        <button
          onClick={() => setShowCreateForm(!showCreateForm)}
          style={styles.createButton}
        >
          {showCreateForm ? 'Cancel' : '+ Create Cloud PC'}
        </button>
      </div>

      {error && (
        <div style={styles.alert} className="error">
          {error}
          <button onClick={() => setError(null)} style={styles.closeButton}>×</button>
        </div>
      )}

      {success && (
        <div style={styles.alert} className="success">
          {success}
          <button onClick={() => setSuccess(null)} style={styles.closeButton}>×</button>
        </div>
      )}

      {showCreateForm && (
        <div style={styles.panel}>
          <h2 style={{ marginTop: 0, marginBottom: '1.5rem', color: '#667eea' }}>Create New Cloud PC</h2>
          <form onSubmit={handleCreate} style={styles.form}>
            <div>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '600', color: '#333' }}>
                VM Name
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                placeholder="e.g., My Windows PC"
                style={styles.input}
                required
              />
            </div>
            <div style={{ marginTop: '1rem' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '600', color: '#333' }}>
                Operating System
              </label>
              <select
                value={formData.os_version}
                onChange={(e) => setFormData({ ...formData, os_version: e.target.value })}
                style={styles.input}
                required
              >
                <option value="1.0 beta">1.0 beta</option>
              </select>
              <p style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.5rem' }}>
                Only 1.0 beta is available at this time
              </p>
            </div>
            <button type="submit" disabled={creating} style={styles.submitButton}>
              {creating ? 'Creating...' : 'Create Cloud PC'}
            </button>
          </form>
        </div>
      )}

      {loading ? (
        <div style={styles.loading}>Loading Cloud PCs...</div>
      ) : cloudPCs.length === 0 ? (
        <div style={styles.empty}>
          <div style={{ fontSize: '48px', marginBottom: '1rem' }}>💻</div>
          <p>No Cloud PCs yet. Create one to get started!</p>
        </div>
      ) : (
        <div style={styles.grid}>
          {cloudPCs.map((pc) => (
            <div key={pc.id} style={styles.card}>
              <div style={styles.cardHeader}>
                <h3 style={styles.cardTitle}>{pc.name}</h3>
                <span style={getStatusStyle(pc.status)}>
                  {pc.status === 'running' ? '🟢 Running' : pc.status === 'stopped' ? '🟡 Stopped' : '⚪ Created'}
                </span>
              </div>
              <div style={styles.cardInfo}>
                <div><strong>OS:</strong> {pc.os_version}</div>
                <div><strong>Storage Used:</strong> {pc.storage_used_mb} MB</div>
                <div><strong>Created:</strong> {new Date(pc.created_at).toLocaleDateString()}</div>
              </div>
              <div style={styles.cardActions}>
                <button
                  onClick={() => handleAccess(pc.id)}
                  style={styles.accessButton}
                >
                  🚀 Access
                </button>
                <button
                  onClick={() => setShowDeleteConfirm(pc.id)}
                  style={styles.deleteButton}
                >
                  🗑️ Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {showDeleteConfirm && (
        <div style={styles.modalOverlay} onClick={() => setShowDeleteConfirm(null)}>
          <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
            <h3 style={{ marginTop: 0 }}>Delete Cloud PC?</h3>
            <p>Are you sure you want to delete this Cloud PC? This action cannot be undone.</p>
            <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end', marginTop: '1rem' }}>
              <button
                onClick={() => setShowDeleteConfirm(null)}
                style={{ ...styles.submitButton, background: '#f1f5f9', color: '#334155' }}
              >
                Cancel
              </button>
              <button
                onClick={() => handleDelete(showDeleteConfirm)}
                style={{ ...styles.submitButton, background: '#ef4444' }}
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    padding: '2rem',
    maxWidth: '1200px',
    margin: '0 auto'
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '2rem'
  },
  title: {
    fontSize: '2rem',
    fontWeight: '700',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    margin: 0
  },
  createButton: {
    padding: '0.75rem 1.5rem',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    fontSize: '1rem',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.2s',
    boxShadow: '0 4px 15px rgba(102, 126, 234, 0.3)'
  },
  alert: {
    padding: '1rem',
    borderRadius: '8px',
    marginBottom: '1rem',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  },
  closeButton: {
    background: 'none',
    border: 'none',
    fontSize: '1.5rem',
    cursor: 'pointer',
    color: 'inherit'
  },
  panel: {
    background: 'white',
    padding: '2rem',
    borderRadius: '12px',
    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
    marginBottom: '2rem'
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem'
  },
  input: {
    width: '100%',
    padding: '0.75rem',
    border: '2px solid #e2e8f0',
    borderRadius: '8px',
    fontSize: '1rem',
    transition: 'all 0.2s'
  },
  submitButton: {
    padding: '0.75rem 1.5rem',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    fontSize: '1rem',
    fontWeight: '600',
    cursor: 'pointer',
    marginTop: '0.5rem'
  },
  loading: {
    textAlign: 'center',
    padding: '3rem',
    color: '#666'
  },
  empty: {
    textAlign: 'center',
    padding: '4rem',
    color: '#666'
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
    gap: '1.5rem'
  },
  card: {
    background: 'white',
    padding: '1.5rem',
    borderRadius: '12px',
    boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
    transition: 'all 0.2s'
  },
  cardHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '1rem'
  },
  cardTitle: {
    margin: 0,
    fontSize: '1.25rem',
    fontWeight: '600',
    color: '#333'
  },
  cardInfo: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
    marginBottom: '1rem',
    fontSize: '0.9rem',
    color: '#666'
  },
  cardActions: {
    display: 'flex',
    gap: '0.5rem'
  },
  accessButton: {
    flex: 1,
    padding: '0.5rem',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600'
  },
  deleteButton: {
    padding: '0.5rem 1rem',
    background: '#ef4444',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontWeight: '600'
  },
  modalOverlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'rgba(0, 0, 0, 0.5)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 10000
  },
  modal: {
    background: 'white',
    padding: '2rem',
    borderRadius: '12px',
    minWidth: '400px',
    maxWidth: '500px',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)'
  }
};

export default CloudPCs;

