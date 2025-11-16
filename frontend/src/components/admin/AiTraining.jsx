import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';

function AiTraining() {
  const [trainingData, setTrainingData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [formData, setFormData] = useState({
    title: '',
    instructions: '',
    is_public: true
  });
  const { user } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!user?.is_admin) {
      navigate('/');
      return;
    }
    fetchTrainingData();
  }, [user, navigate]);

  const fetchTrainingData = async () => {
    try {
      setLoading(true);
      const response = await api.get('/api/ai/training');
      setTrainingData(response.data?.training || []);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch training data:', err);
      if (err.response?.status === 401) {
        setError('Please log in as an admin to manage AI training.');
      } else {
        setError('Failed to fetch training data: ' + (err.response?.data?.error || err.message));
      }
      setTrainingData([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.title.trim() || !formData.instructions.trim()) {
      setError('Title and instructions are required.');
      return;
    }

    try {
      setError(null);
      setSuccess(null);
      if (editingId) {
        await api.put(`/api/ai/training/${editingId}`, formData);
        setSuccess('Training data updated successfully.');
      } else {
        await api.post('/api/ai/training', formData);
        setSuccess('Training data created successfully.');
      }
      setFormData({ title: '', instructions: '', is_public: true });
      setShowForm(false);
      setEditingId(null);
      await fetchTrainingData();
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      const message = err.response?.data?.error || err.message || 'Failed to save training data';
      if (err.response?.status === 401) {
        setError('Please log in as an admin to save training data.');
      } else {
        setError('Failed to save training data: ' + message);
      }
    }
  };

  const handleEdit = (training) => {
    setFormData({
      title: training.title || '',
      instructions: training.instructions || '',
      is_public: training.is_public !== false
    });
    setEditingId(training.id);
    setShowForm(true);
    setError(null);
    setSuccess(null);
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this training data?')) {
      return;
    }

    try {
      await api.delete(`/api/ai/training/${id}`);
      setSuccess('Training data deleted successfully.');
      await fetchTrainingData();
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      const message = err.response?.data?.error || err.message || 'Failed to delete training data';
      if (err.response?.status === 401) {
        setError('Please log in as an admin to delete training data.');
      } else {
        setError('Failed to delete training data: ' + message);
      }
    }
  };

  const handleCancel = () => {
    setFormData({ title: '', instructions: '', is_public: true });
    setShowForm(false);
    setEditingId(null);
    setError(null);
    setSuccess(null);
  };

  if (!user?.is_admin) {
    return null;
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>🤖 AI Training</h1>
        <button
          onClick={() => {
            setShowForm(!showForm);
            if (showForm) {
              handleCancel();
            }
          }}
          style={showForm ? styles.cancelButton : styles.newButton}
        >
          {showForm ? '✕ Cancel' : '+ New Training Data'}
        </button>
      </div>
      
      <p style={styles.subtitle}>
        Define training data for the AI. When users ask about topics matching your keywords, the AI will use your custom instructions.
      </p>

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

      {showForm && (
        <form onSubmit={handleSubmit} style={styles.form}>
          <h3 style={styles.formTitle}>
            {editingId ? 'Edit Training Data' : 'Add New Training Data'}
          </h3>
          <div style={styles.formGrid}>
            <div style={styles.formGroup}>
              <label style={styles.label}>Title *</label>
              <input
                type="text"
                value={formData.title}
                onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                style={styles.input}
                placeholder="e.g., Health & Wellness Guidelines"
                required
              />
            </div>
            <div style={styles.formGroup}>
              <label style={styles.label}>Instructions *</label>
              <textarea
                value={formData.instructions}
                onChange={(e) => setFormData({ ...formData, instructions: e.target.value })}
                style={styles.textarea}
                placeholder="Enter instructions the AI should follow when users ask about topics related to this training data..."
                rows={6}
                required
              />
              <div style={styles.helpText}>
                These instructions will be used by the AI when processing relevant user queries.
              </div>
            </div>
            <div style={styles.checkboxGroup}>
              <label style={styles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={formData.is_public}
                  onChange={(e) => setFormData({ ...formData, is_public: e.target.checked })}
                  style={styles.checkbox}
                />
                <span style={styles.checkboxText}>Make this training data public (available to all users)</span>
              </label>
            </div>
          </div>
          <div style={styles.formActions}>
            <button type="submit" style={styles.submitButton}>
              {editingId ? 'Update' : 'Create'}
            </button>
            <button type="button" onClick={handleCancel} style={styles.cancelButton}>
              Cancel
            </button>
          </div>
        </form>
      )}

      <div style={styles.tableContainer}>
        {loading ? (
          <div style={styles.loading}>Loading training data...</div>
        ) : trainingData.length === 0 ? (
          <div style={styles.empty}>
            <span style={styles.emptyIcon}>📚</span>
            <p>No training data yet. Click "+ New Training Data" to get started.</p>
          </div>
        ) : (
          <div style={styles.table}>
            <div style={styles.tableHeader}>
              <div style={styles.headerCell}>Title</div>
              <div style={styles.headerCell}>Instructions</div>
              <div style={styles.headerCell}>Visibility</div>
              <div style={styles.headerCell}>Actions</div>
            </div>
            {trainingData.map((item) => (
              <div key={item.id} style={styles.tableRow}>
                <div style={styles.cell}>
                  <div style={styles.titleText}>{item.title}</div>
                </div>
                <div style={styles.cell}>
                  <div style={styles.instructionsPreview}>{item.instructions}</div>
                </div>
                <div style={styles.cell}>
                  <span style={item.is_public ? styles.publicBadge : styles.privateBadge}>
                    {item.is_public ? '🌐 Public' : '🔒 Private'}
                  </span>
                </div>
                <div style={styles.cell}>
                  <button
                    onClick={() => handleEdit(item)}
                    style={styles.editButton}
                    title="Edit"
                  >
                    ✏️
                  </button>
                  <button
                    onClick={() => handleDelete(item.id)}
                    style={styles.deleteButton}
                    title="Delete"
                  >
                    🗑️
                  </button>
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
    marginBottom: '1rem',
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
  subtitle: {
    fontSize: '1.1rem',
    color: 'rgba(255, 255, 255, 0.9)',
    marginBottom: '2rem',
    maxWidth: '800px',
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
  newButton: {
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
  form: {
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
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
    marginBottom: '1.5rem',
  },
  formGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
  },
  label: {
    fontSize: '0.95rem',
    fontWeight: '600',
    color: '#333',
  },
  helpText: {
    fontSize: '0.85rem',
    color: '#666',
    marginTop: '0.25rem',
    fontStyle: 'italic',
  },
  input: {
    padding: '0.875rem 1rem',
    fontSize: '1rem',
    border: '2px solid rgba(102, 126, 234, 0.3)',
    borderRadius: '12px',
    transition: 'all 0.3s ease',
    backgroundColor: 'white',
  },
  textarea: {
    padding: '0.875rem 1rem',
    fontSize: '1rem',
    border: '2px solid rgba(102, 126, 234, 0.3)',
    borderRadius: '12px',
    fontFamily: 'inherit',
    resize: 'vertical',
    minHeight: '120px',
    backgroundColor: 'white',
  },
  checkboxGroup: {
    marginTop: '0.5rem',
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
    gap: '1rem',
    marginTop: '1rem',
  },
  submitButton: {
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
  loading: {
    textAlign: 'center',
    padding: '3rem',
    color: 'rgba(255, 255, 255, 0.9)',
    fontSize: '1.1rem',
  },
  empty: {
    textAlign: 'center',
    padding: '3rem',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    border: '1px solid rgba(102, 126, 234, 0.3)',
  },
  emptyIcon: {
    fontSize: '3rem',
    display: 'block',
    marginBottom: '1rem',
  },
  tableContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    padding: '2rem',
    border: '1px solid rgba(102, 126, 234, 0.3)',
    overflow: 'hidden',
  },
  table: {
    display: 'grid',
    gridTemplateColumns: '1fr',
    gap: '0.75rem',
  },
  tableHeader: {
    display: 'grid',
    gridTemplateColumns: '2fr 3fr 1fr 1.5fr',
    padding: '1.25rem 1rem',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    fontWeight: '600',
    fontSize: '0.95rem',
    borderRadius: '12px',
  },
  tableRow: {
    display: 'grid',
    gridTemplateColumns: '2fr 3fr 1fr 1.5fr',
    padding: '1.25rem 1rem',
    borderBottom: '1px solid #f0f0f0',
    backgroundColor: 'white',
    borderRadius: '12px',
    transition: 'all 0.2s ease',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)',
  },
  headerCell: {
    padding: '0 0.5rem',
  },
  cell: {
    padding: '0 0.5rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  titleText: {
    fontWeight: '600',
    color: '#333',
    fontSize: '1rem',
  },
  instructionsPreview: {
    maxHeight: '60px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    fontSize: '0.9rem',
    color: '#666',
    lineHeight: '1.4',
  },
  publicBadge: {
    display: 'inline-block',
    padding: '0.375rem 0.875rem',
    borderRadius: '20px',
    fontSize: '0.85rem',
    fontWeight: '600',
    background: 'linear-gradient(135deg, #4caf50 0%, #45a049 100%)',
    color: 'white',
  },
  privateBadge: {
    display: 'inline-block',
    padding: '0.375rem 0.875rem',
    borderRadius: '20px',
    fontSize: '0.85rem',
    fontWeight: '600',
    backgroundColor: '#e0e0e0',
    color: '#666',
  },
  editButton: {
    padding: '0.5rem 1rem',
    background: 'linear-gradient(135deg, #ff9800 0%, #f57c00 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '1rem',
    boxShadow: '0 2px 8px rgba(255, 152, 0, 0.3)',
    transition: 'all 0.3s ease',
  },
  deleteButton: {
    padding: '0.5rem 1rem',
    background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '1rem',
    boxShadow: '0 2px 8px rgba(245, 87, 108, 0.3)',
    transition: 'all 0.3s ease',
  },
};

export default AiTraining;
