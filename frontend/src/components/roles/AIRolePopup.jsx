import React, { useState } from 'react';
import api from '../../services/api';

const styles = {
  overlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'rgba(0, 0, 0, 0.7)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 10000,
    backdropFilter: 'blur(5px)',
  },
  modal: {
    background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 250, 252, 0.98) 100%)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    padding: '2rem',
    maxWidth: '600px',
    width: '90%',
    maxHeight: '90vh',
    overflowY: 'auto',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.4)',
    border: '2px solid rgba(102, 126, 234, 0.4)',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '1.5rem',
  },
  title: {
    fontSize: '1.75rem',
    fontWeight: '700',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
    margin: 0,
  },
  closeButton: {
    background: 'transparent',
    border: 'none',
    fontSize: '1.5rem',
    color: '#999',
    cursor: 'pointer',
    padding: '0.5rem',
    borderRadius: '50%',
    width: '36px',
    height: '36px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.2s ease',
  },
  closeButtonHover: {
    background: 'rgba(102, 126, 234, 0.1)',
    color: '#667eea',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
  },
  inputGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
  },
  label: {
    fontSize: '0.9rem',
    fontWeight: '600',
    color: '#555',
  },
  input: {
    padding: '0.875rem 1rem',
    fontSize: '1rem',
    border: '2px solid rgba(102, 126, 234, 0.3)',
    borderRadius: '10px',
    transition: 'all 0.3s ease',
    backgroundColor: 'white',
  },
  textarea: {
    padding: '0.875rem 1rem',
    fontSize: '1rem',
    border: '2px solid rgba(102, 126, 234, 0.3)',
    borderRadius: '10px',
    minHeight: '100px',
    resize: 'vertical',
    fontFamily: 'inherit',
    transition: 'all 0.3s ease',
    backgroundColor: 'white',
  },
  aiButton: {
    padding: '0.75rem 1.5rem',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '10px',
    cursor: 'pointer',
    fontSize: '0.9rem',
    fontWeight: '600',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    transition: 'all 0.3s ease',
    alignSelf: 'flex-start',
  },
  aiInstructionsBox: {
    background: 'rgba(102, 126, 234, 0.1)',
    padding: '1rem',
    borderRadius: '10px',
    border: '1px solid rgba(102, 126, 234, 0.2)',
    marginTop: '0.5rem',
  },
  aiInstructionsLabel: {
    fontSize: '0.85rem',
    fontWeight: '600',
    color: '#667eea',
    marginBottom: '0.5rem',
  },
  aiInstructionsText: {
    color: '#555',
    fontSize: '0.9rem',
    lineHeight: '1.6',
    whiteSpace: 'pre-wrap',
  },
  actions: {
    display: 'flex',
    gap: '1rem',
    justifyContent: 'flex-end',
    marginTop: '1rem',
  },
  cancelButton: {
    padding: '0.75rem 1.5rem',
    background: 'rgba(102, 126, 234, 0.1)',
    color: '#667eea',
    border: '2px solid rgba(102, 126, 234, 0.3)',
    borderRadius: '10px',
    cursor: 'pointer',
    fontSize: '1rem',
    fontWeight: '600',
    transition: 'all 0.3s ease',
  },
  createButton: {
    padding: '0.75rem 1.5rem',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '10px',
    cursor: 'pointer',
    fontSize: '1rem',
    fontWeight: '600',
    boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
    transition: 'all 0.3s ease',
  },
  loading: {
    opacity: 0.6,
    pointerEvents: 'none',
  },
  error: {
    color: '#d32f2f',
    fontSize: '0.9rem',
    marginTop: '0.5rem',
  },
};

function AIRolePopup({ onClose, onCreate }) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [aiInstructions, setAiInstructions] = useState('');
  const [loading, setLoading] = useState(false);
  const [generatingAI, setGeneratingAI] = useState(false);
  const [error, setError] = useState(null);
  const [closeButtonHovered, setCloseButtonHovered] = useState(false);

  const handleGenerateAI = async () => {
    if (!name.trim()) {
      setError('Please enter a role name first');
      return;
    }

    setGeneratingAI(true);
    setError(null);
    try {
      const response = await api.post('/api/roles/ai-suggest', {
        role_name: name,
        role_description: description,
      });
      setAiInstructions(response.data.ai_instructions || '');
    } catch (err) {
      console.error('Failed to generate AI instructions:', err);
      setError(err.response?.data?.error || 'Failed to generate AI instructions');
    } finally {
      setGeneratingAI(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!name.trim()) {
      setError('Role name is required');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      await onCreate({
        name: name.trim(),
        description: description.trim(),
        ai_instructions: aiInstructions.trim(),
      });
    } catch (err) {
      console.error('Failed to create role:', err);
      setError(err.response?.data?.error || 'Failed to create role');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.overlay} onClick={onClose}>
      <div style={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div style={styles.header}>
          <h2 style={styles.title}>🤖 Create New Role</h2>
          <button
            onClick={onClose}
            onMouseEnter={() => setCloseButtonHovered(true)}
            onMouseLeave={() => setCloseButtonHovered(false)}
            style={{
              ...styles.closeButton,
              ...(closeButtonHovered ? styles.closeButtonHover : {}),
            }}
          >
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} style={{ ...styles.form, ...(loading ? styles.loading : {}) }}>
          <div style={styles.inputGroup}>
            <label style={styles.label}>Role Name *</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Moderator, Content Creator, Viewer"
              style={styles.input}
              required
              disabled={loading}
            />
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe what this role is for..."
              style={styles.textarea}
              disabled={loading}
            />
          </div>

          <div style={styles.inputGroup}>
            <label style={styles.label}>
              AI Instructions
              <button
                type="button"
                onClick={handleGenerateAI}
                disabled={generatingAI || loading || !name.trim()}
                style={styles.aiButton}
              >
                {generatingAI ? '🔄 Generating...' : '🤖 Generate with AI'}
              </button>
            </label>
            {aiInstructions && (
              <div style={styles.aiInstructionsBox}>
                <div style={styles.aiInstructionsLabel}>AI-Generated Instructions:</div>
                <div style={styles.aiInstructionsText}>{aiInstructions}</div>
              </div>
            )}
            <textarea
              value={aiInstructions}
              onChange={(e) => setAiInstructions(e.target.value)}
              placeholder="Instructions for what happens when this role is assigned (or use AI to generate)"
              style={styles.textarea}
              disabled={loading}
            />
          </div>

          {error && <div style={styles.error}>⚠️ {error}</div>}

          <div style={styles.actions}>
            <button
              type="button"
              onClick={onClose}
              style={styles.cancelButton}
              disabled={loading}
            >
              Cancel
            </button>
            <button
              type="submit"
              style={styles.createButton}
              disabled={loading || !name.trim()}
            >
              {loading ? 'Creating...' : 'Create Role'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default AIRolePopup;

