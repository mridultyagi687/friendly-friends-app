import React, { useState, useEffect, useCallback } from 'react';
import api from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';

function AiDocs() {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState('docs'); // 'docs' or 'images'
  
  // Document state
  const [docPrompt, setDocPrompt] = useState('');
  const [generatingDoc, setGeneratingDoc] = useState(false);
  const [docs, setDocs] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState(null);
  const [docTitle, setDocTitle] = useState('');
  const [docContent, setDocContent] = useState('');
  const [savingDoc, setSavingDoc] = useState(false);
  const [loadingDocs, setLoadingDocs] = useState(true);
  const [originalDocTitle, setOriginalDocTitle] = useState('');
  const [originalDocContent, setOriginalDocContent] = useState('');
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  
  // Image state
  const [imagePrompt, setImagePrompt] = useState('');
  const [generatingImage, setGeneratingImage] = useState(false);
  const [images, setImages] = useState([]);
  const [selectedImageId, setSelectedImageId] = useState(null);
  const [loadingImages, setLoadingImages] = useState(true);
  
  // Common state
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const selectedDoc = docs.find(d => d.id === selectedDocId);
  const selectedImage = images.find(img => img.id === selectedImageId);

  useEffect(() => {
    if (!user) {
      setLoadingDocs(false);
      setLoadingImages(false);
      return;
    }
    if (activeTab === 'docs') {
      fetchDocs();
    } else {
      fetchImages();
    }
  }, [user, activeTab]);

  const fetchDocs = async () => {
    setLoadingDocs(true);
    setError(null);
    try {
      const { data } = await api.get('/api/ai/docs');
      const docsList = Array.isArray(data?.docs) ? data.docs : [];
      setDocs(docsList);
      if (docsList.length > 0 && !selectedDocId) {
        setSelectedDocId(docsList[0].id);
      }
    } catch (err) {
      const message = err.response?.data?.error || 'Unable to load documents right now.';
      if (err.response?.status === 401) {
        setError('Please log in to view documents.');
      } else {
        setError(message);
      }
    } finally {
      setLoadingDocs(false);
    }
  };

  const fetchImages = async () => {
    setLoadingImages(true);
    setError(null);
    try {
      const { data } = await api.get('/api/ai/images');
      const imagesList = Array.isArray(data?.images) ? data.images : [];
      setImages(imagesList);
      if (imagesList.length > 0 && !selectedImageId) {
        setSelectedImageId(imagesList[0].id);
      }
    } catch (err) {
      const message = err.response?.data?.error || 'Unable to load images right now.';
      if (err.response?.status === 401) {
        setError('Please log in to view images.');
      } else {
        setError(message);
      }
    } finally {
      setLoadingImages(false);
    }
  };

  useEffect(() => {
    if (selectedDoc) {
      setDocTitle(selectedDoc.title);
      setDocContent(selectedDoc.content);
      setOriginalDocTitle(selectedDoc.title);
      setOriginalDocContent(selectedDoc.content);
      setHasUnsavedChanges(false);
      setSuccess(null); // Clear any success message when selecting a new doc
    } else {
      setDocTitle('');
      setDocContent('');
      setOriginalDocTitle('');
      setOriginalDocContent('');
      setHasUnsavedChanges(false);
    }
  }, [selectedDoc]);

  const handleGenerateDoc = async (e) => {
    e.preventDefault();
    const cleanedPrompt = docPrompt.trim();
    if (!cleanedPrompt || generatingDoc) return;

    setGeneratingDoc(true);
    setError(null);
    setSuccess(null);
    try {
      const { data } = await api.post('/api/ai/docs', {
        prompt: cleanedPrompt,
      });
      if (data?.doc) {
        setDocs(prev => [data.doc, ...prev]);
        setSelectedDocId(data.doc.id);
        setDocPrompt('');
        setSuccess('Document generated successfully!');
        setTimeout(() => setSuccess(null), 3000);
      } else {
        setError('No document content received from server.');
      }
    } catch (err) {
      const message = err.response?.data?.error || 'Failed to generate document. Please try again later.';
      if (err.response?.status === 401) {
        setError('Please log in to generate documents.');
      } else {
        setError(message);
      }
    } finally {
      setGeneratingDoc(false);
    }
  };

  const handleGenerateImage = async (e) => {
    e.preventDefault();
    const cleanedPrompt = imagePrompt.trim();
    if (!cleanedPrompt || generatingImage) return;

    setGeneratingImage(true);
    setError(null);
    setSuccess(null);
    try {
      const { data } = await api.post('/api/ai/images', {
        prompt: cleanedPrompt,
      });
      if (data?.image) {
        setImages(prev => [data.image, ...prev]);
        setSelectedImageId(data.image.id);
        setImagePrompt('');
        setSuccess('Image generated successfully!');
        setTimeout(() => setSuccess(null), 3000);
      } else {
        setError('No image received from server.');
      }
    } catch (err) {
      console.error('Image generation error:', err);
      const message = err.response?.data?.error || err.message || 'Failed to generate image. Please try again later.';
      if (err.response?.status === 401) {
        setError('Please log in to generate images.');
      } else {
        setError(message);
      }
    } finally {
      setGeneratingImage(false);
    }
  };

  const handleSaveDoc = useCallback(async () => {
    if (!selectedDocId || savingDoc) return;
    
    // Check if there are actual changes
    const titleChanged = docTitle.trim() !== originalDocTitle.trim();
    const contentChanged = docContent !== originalDocContent;
    
    if (!titleChanged && !contentChanged) {
      // No changes, don't save
      return;
    }
    
    setSavingDoc(true);
    setError(null);
    try {
      const { data } = await api.put(`/api/ai/docs/${selectedDocId}`, {
        title: docTitle.trim(),
        content: docContent,
      });
      if (data?.doc) {
        setDocs(prev => prev.map(doc => doc.id === data.doc.id ? data.doc : doc));
        // Update original values to match saved values
        setOriginalDocTitle(data.doc.title);
        setOriginalDocContent(data.doc.content);
        setHasUnsavedChanges(false);
        // Only show success message if there were actual changes
        if (titleChanged || contentChanged) {
          setSuccess('Document saved!');
          setTimeout(() => setSuccess(null), 2000);
        }
      }
    } catch (err) {
      const message = err.response?.data?.error || 'Unable to save changes right now.';
      setError(message);
    } finally {
      setSavingDoc(false);
    }
  }, [selectedDocId, docTitle, docContent, originalDocTitle, originalDocContent, savingDoc]);

  // Track changes to detect unsaved edits
  useEffect(() => {
    if (selectedDocId) {
      const titleChanged = docTitle.trim() !== originalDocTitle.trim();
      const contentChanged = docContent !== originalDocContent;
      setHasUnsavedChanges(titleChanged || contentChanged);
    }
  }, [docTitle, docContent, originalDocTitle, originalDocContent, selectedDocId]);

  // Auto-save only when there are actual changes
  useEffect(() => {
    if (activeTab === 'docs' && selectedDocId && hasUnsavedChanges && docTitle.trim() && docContent.trim()) {
      const timer = setTimeout(() => {
        handleSaveDoc();
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [docTitle, docContent, selectedDocId, activeTab, hasUnsavedChanges, handleSaveDoc]);

  const handleDeleteDoc = async () => {
    if (!selectedDocId) return;
    if (!window.confirm('Delete this document?')) return;

    try {
      await api.delete(`/api/ai/docs/${selectedDocId}`);
      setDocs(prev => prev.filter(doc => doc.id !== selectedDocId));
      const remaining = docs.filter(doc => doc.id !== selectedDocId);
      setSelectedDocId(remaining.length > 0 ? remaining[0].id : null);
      setSuccess('Document deleted!');
      setTimeout(() => setSuccess(null), 2000);
    } catch (err) {
      const message = err.response?.data?.error || 'Unable to delete document right now.';
      setError(message);
    }
  };

  const handleDeleteImage = async () => {
    if (!selectedImageId) return;
    if (!window.confirm('Delete this image?')) return;

    try {
      await api.delete(`/api/ai/images/${selectedImageId}`);
      setImages(prev => prev.filter(img => img.id !== selectedImageId));
      const remaining = images.filter(img => img.id !== selectedImageId);
      setSelectedImageId(remaining.length > 0 ? remaining[0].id : null);
      setSuccess('Image deleted!');
      setTimeout(() => setSuccess(null), 2000);
    } catch (err) {
      const message = err.response?.data?.error || 'Unable to delete image right now.';
      setError(message);
    }
  };

  const handleCopyDoc = () => {
    if (docContent) {
      navigator.clipboard.writeText(docContent).then(() => {
        setSuccess('Document copied to clipboard!');
        setTimeout(() => setSuccess(null), 2000);
      }).catch(() => {
        setError('Failed to copy document.');
      });
    }
  };

  const handleDownloadDoc = () => {
    if (!docTitle || !docContent) return;
    
    const sanitizedTitle = docTitle.trim().replace(/[^a-z0-9]/gi, '_').toLowerCase() || 'untitled_document';
    const filename = `${sanitizedTitle}.txt`;
    
    const blob = new Blob([docContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    setSuccess('Document downloaded!');
    setTimeout(() => setSuccess(null), 2000);
  };

  const handleDownloadImage = () => {
    if (!selectedImage) return;
    
    const imageUrl = `/uploads/ai_images/${selectedImage.filename}`;
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = selectedImage.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    setSuccess('Image downloaded!');
    setTimeout(() => setSuccess(null), 2000);
  };

  const getImageUrl = (filename) => {
    return `/uploads/ai_images/${filename}`;
  };

  if (!user) {
    return (
      <div style={styles.container}>
        <div style={styles.loginPrompt}>
          <h1 style={styles.title}>📄🖼️ My Docs and Images</h1>
          <p style={styles.promptText}>Please log in to generate AI documents and images.</p>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div>
          <h1 style={styles.title}>📄🖼️ My Docs and Images</h1>
          <p style={styles.subtitle}>
            Generate structured documents and JPEG images with Friendly Friends AI. All content is saved automatically.
          </p>
        </div>
        {savingDoc && (
          <span style={styles.statusText}>Saving...</span>
        )}
      </header>

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

      <div style={styles.tabs}>
        <button
          style={{
            ...styles.tab,
            ...(activeTab === 'docs' ? styles.tabActive : {}),
          }}
          onClick={() => setActiveTab('docs')}
        >
          📄 Documents
        </button>
        <button
          style={{
            ...styles.tab,
            ...(activeTab === 'images' ? styles.tabActive : {}),
          }}
          onClick={() => setActiveTab('images')}
        >
          🖼️ Images
        </button>
      </div>

      {activeTab === 'docs' ? (
        <div style={styles.layout}>
          <div style={styles.panel}>
            <div style={styles.panelHeader}>
              <h2 style={styles.panelTitle}>Create new document</h2>
            </div>
            <form style={styles.formBody} onSubmit={handleGenerateDoc}>
              <textarea
                style={styles.textarea}
                placeholder="Example: Draft a one-page press release announcing our new wellness studio with three key sections."
                value={docPrompt}
                onChange={(e) => setDocPrompt(e.target.value)}
                disabled={generatingDoc}
                rows={8}
              />
              <div style={styles.formActions}>
                <button
                  type="submit"
                  style={{
                    ...styles.submitButton,
                    ...((generatingDoc || !docPrompt.trim()) ? styles.submitDisabled : {}),
                  }}
                  disabled={generatingDoc || !docPrompt.trim()}
                >
                  {generatingDoc ? 'Generating…' : 'Generate Document'}
                </button>
              </div>
            </form>
          </div>

          <div style={styles.panel}>
            <div style={styles.panelHeader}>
              <h2 style={styles.panelTitle}>Your documents</h2>
            </div>
            <div style={styles.list}>
              {loadingDocs ? (
                <div style={styles.emptyState}>Loading your docs…</div>
              ) : docs.length === 0 ? (
                <div style={styles.emptyState}>
                  <span style={styles.emptyIcon}>📝</span>
                  <p>No docs yet. Generate your first document using the prompt box on the left.</p>
                </div>
              ) : (
                docs.map((doc) => (
                  <div
                    key={doc.id}
                    style={{
                      ...styles.listItem,
                      ...(doc.id === selectedDocId ? styles.listItemActive : {}),
                    }}
                    onClick={() => setSelectedDocId(doc.id)}
                  >
                    <h3 style={styles.listTitle}>{doc.title || 'Untitled doc'}</h3>
                    <div style={styles.listMeta}>
                      {doc.updated_at ? `Updated ${new Date(doc.updated_at).toLocaleString()}` : `Created ${new Date(doc.created_at).toLocaleString()}`}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          <div style={{ ...styles.panel, gridColumn: '1 / span 2', minHeight: '500px' }}>
            {selectedDoc ? (
              <>
                <div style={styles.editorHeader}>
                  <input
                    style={styles.input}
                    value={docTitle}
                    onChange={(e) => setDocTitle(e.target.value)}
                    placeholder="Document title"
                  />
                  <div style={styles.editorActions}>
                    <button
                      onClick={handleCopyDoc}
                      style={styles.actionButton}
                      title="Copy to clipboard"
                    >
                      📋 Copy
                    </button>
                    <button
                      onClick={handleDownloadDoc}
                      style={styles.downloadButton}
                      title="Download document"
                    >
                      ⬇️ Download
                    </button>
                    <button
                      onClick={handleDeleteDoc}
                      style={styles.deleteButton}
                      title="Delete document"
                    >
                      🗑️ Delete
                    </button>
                  </div>
                </div>
                <div style={styles.editorBody}>
                  <textarea
                    style={styles.documentTextarea}
                    value={docContent}
                    onChange={(e) => setDocContent(e.target.value)}
                    placeholder="Document content..."
                    rows={20}
                  />
                  <div style={styles.statusText}>
                    {savingDoc ? 'Saving changes…' : hasUnsavedChanges ? 'Unsaved changes - saving automatically...' : 'All changes saved'}
                  </div>
                </div>
              </>
            ) : (
              <div style={styles.emptyState}>
                <span style={styles.emptyIcon}>📄</span>
                <p>Select a document on the left to start editing, or generate a new one.</p>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div style={styles.layout}>
          <div style={styles.panel}>
            <div style={styles.panelHeader}>
              <h2 style={styles.panelTitle}>Generate new image</h2>
            </div>
            <form style={styles.formBody} onSubmit={handleGenerateImage}>
              <textarea
                style={styles.textarea}
                placeholder="Example: A serene mountain landscape at sunset with a lake reflecting the colors"
                value={imagePrompt}
                onChange={(e) => setImagePrompt(e.target.value)}
                disabled={generatingImage}
                rows={8}
              />
              <div style={styles.formActions}>
                <button
                  type="submit"
                  style={{
                    ...styles.submitButton,
                    ...((generatingImage || !imagePrompt.trim()) ? styles.submitDisabled : {}),
                  }}
                  disabled={generatingImage || !imagePrompt.trim()}
                >
                  {generatingImage ? 'Generating…' : 'Generate Image'}
                </button>
              </div>
            </form>
          </div>

          <div style={styles.panel}>
            <div style={styles.panelHeader}>
              <h2 style={styles.panelTitle}>Your images</h2>
            </div>
            <div style={styles.imageGrid}>
              {loadingImages ? (
                <div style={styles.emptyState}>Loading your images…</div>
              ) : images.length === 0 ? (
                <div style={styles.emptyState}>
                  <span style={styles.emptyIcon}>🖼️</span>
                  <p>No images yet. Generate your first image using the prompt box on the left.</p>
                </div>
              ) : (
                images.map((img) => (
                  <div
                    key={img.id}
                    style={{
                      ...styles.imageCard,
                      ...(img.id === selectedImageId ? styles.imageCardActive : {}),
                    }}
                    onClick={() => setSelectedImageId(img.id)}
                  >
                    <img
                      src={getImageUrl(img.filename)}
                      alt={img.title}
                      style={styles.imageThumbnail}
                      onError={(e) => {
                        e.target.style.display = 'none';
                      }}
                    />
                    <div style={styles.imageCardInfo}>
                      <h3 style={styles.imageCardTitle}>{img.title || 'Untitled image'}</h3>
                      <div style={styles.imageCardMeta}>
                        {new Date(img.created_at).toLocaleDateString()}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          <div style={{ ...styles.panel, gridColumn: '1 / span 2', minHeight: '500px' }}>
            {selectedImage ? (
              <>
                <div style={styles.editorHeader}>
                  <div style={styles.imageTitle}>{selectedImage.title || 'Untitled image'}</div>
                  <div style={styles.editorActions}>
                    <button
                      onClick={handleDownloadImage}
                      style={styles.downloadButton}
                      title="Download image"
                    >
                      ⬇️ Download JPEG
                    </button>
                    <button
                      onClick={handleDeleteImage}
                      style={styles.deleteButton}
                      title="Delete image"
                    >
                      🗑️ Delete
                    </button>
                  </div>
                </div>
                <div style={styles.imageViewer}>
                  <img
                    src={getImageUrl(selectedImage.filename)}
                    alt={selectedImage.title}
                    style={styles.imageFull}
                    onError={(e) => {
                      e.target.style.display = 'none';
                    }}
                  />
                  {selectedImage.prompt && (
                    <div style={styles.imagePrompt}>
                      <strong>Prompt:</strong> {selectedImage.prompt}
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div style={styles.emptyState}>
                <span style={styles.emptyIcon}>🖼️</span>
                <p>Select an image on the left to view it, or generate a new one.</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    maxWidth: '1400px',
    margin: '0 auto',
    padding: '2rem 1.5rem 4rem',
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
  header: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '1.5rem',
    alignItems: 'center',
    justifyContent: 'space-between',
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
  subtitle: {
    fontSize: '1.1rem',
    color: 'rgba(255, 255, 255, 0.9)',
    maxWidth: '800px',
    marginTop: '0.5rem',
  },
  promptText: {
    fontSize: '1.1rem',
    color: '#666',
  },
  tabs: {
    display: 'flex',
    gap: '1rem',
    marginBottom: '1.5rem',
  },
  tab: {
    padding: '0.75rem 1.5rem',
    background: 'rgba(102, 126, 234, 0.2)',
    border: '2px solid rgba(102, 126, 234, 0.3)',
    borderRadius: '12px',
    color: 'rgba(255, 255, 255, 0.9)',
    fontSize: '1rem',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
  },
  tabActive: {
    background: 'rgba(255, 255, 255, 0.95)',
    color: '#667eea',
    boxShadow: '0 4px 15px rgba(0, 0, 0, 0.2)',
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
  statusText: {
    fontSize: '0.9rem',
    color: '#64748b',
  },
  layout: {
    display: 'grid',
    gap: '1.5rem',
    gridTemplateColumns: 'minmax(300px, 1fr) minmax(300px, 1fr)',
  },
  panel: {
    borderRadius: '20px',
    border: '1px solid rgba(102, 126, 234, 0.3)',
    background: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
  },
  panelHeader: {
    padding: '1.25rem 1.5rem',
    borderBottom: '1px solid rgba(102, 126, 234, 0.2)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '1rem',
    background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
  },
  panelTitle: {
    fontSize: '1.25rem',
    fontWeight: '600',
    color: '#1f2937',
    margin: 0,
  },
  formBody: {
    padding: '1.5rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
    flex: 1,
  },
  textarea: {
    minHeight: '120px',
    width: '100%',
    borderRadius: '12px',
    border: '2px solid rgba(102, 126, 234, 0.3)',
    padding: '1rem',
    backgroundColor: 'white',
    color: '#0f172a',
    resize: 'vertical',
    fontSize: '1rem',
    lineHeight: 1.5,
    transition: 'all 0.3s ease',
  },
  formActions: {
    display: 'flex',
    gap: '1rem',
    flexWrap: 'wrap',
  },
  submitButton: {
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: '#ffffff',
    border: 'none',
    padding: '0.875rem 1.75rem',
    borderRadius: '12px',
    fontSize: '1rem',
    fontWeight: '600',
    cursor: 'pointer',
    boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
    transition: 'all 0.3s ease',
  },
  submitDisabled: {
    backgroundColor: '#94a3b8',
    cursor: 'not-allowed',
    boxShadow: 'none',
    background: 'linear-gradient(135deg, #94a3b8 0%, #757575 100%)',
  },
  list: {
    flex: 1,
    overflowY: 'auto',
  },
  listItem: {
    padding: '1rem 1.25rem',
    borderBottom: '1px solid #f1f5f9',
    cursor: 'pointer',
    backgroundColor: '#ffffff',
    transition: 'background-color 0.15s ease',
  },
  listItemActive: {
    backgroundColor: 'rgba(102, 126, 234, 0.1)',
  },
  listTitle: {
    margin: 0,
    fontWeight: 600,
    color: '#0f172a',
    fontSize: '1rem',
  },
  listMeta: {
    fontSize: '0.75rem',
    color: '#94a3b8',
    marginTop: '0.25rem',
  },
  imageGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))',
    gap: '1rem',
    padding: '1.5rem',
    overflowY: 'auto',
    flex: 1,
  },
  imageCard: {
    borderRadius: '12px',
    overflow: 'hidden',
    cursor: 'pointer',
    border: '2px solid transparent',
    transition: 'all 0.3s ease',
    backgroundColor: '#ffffff',
  },
  imageCardActive: {
    border: '2px solid #667eea',
    boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
  },
  imageThumbnail: {
    width: '100%',
    height: '150px',
    objectFit: 'cover',
  },
  imageCardInfo: {
    padding: '0.75rem',
  },
  imageCardTitle: {
    margin: 0,
    fontSize: '0.9rem',
    fontWeight: 600,
    color: '#0f172a',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  imageCardMeta: {
    fontSize: '0.7rem',
    color: '#94a3b8',
    marginTop: '0.25rem',
  },
  editorHeader: {
    padding: '1.25rem 1.5rem',
    borderBottom: '1px solid #e2e8f0',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '1rem',
  },
  input: {
    flex: 1,
    borderRadius: '10px',
    border: '2px solid rgba(102, 126, 234, 0.3)',
    padding: '0.75rem 1rem',
    fontSize: '1rem',
    backgroundColor: 'white',
    color: '#0f172a',
    transition: 'all 0.3s ease',
  },
  imageTitle: {
    flex: 1,
    fontSize: '1.1rem',
    fontWeight: '600',
    color: '#0f172a',
  },
  editorActions: {
    display: 'flex',
    gap: '0.75rem',
  },
  actionButton: {
    background: 'linear-gradient(135deg, #4caf50 0%, #45a049 100%)',
    color: '#ffffff',
    border: 'none',
    padding: '0.5rem 1rem',
    borderRadius: '8px',
    fontSize: '0.9rem',
    fontWeight: '600',
    cursor: 'pointer',
    boxShadow: '0 2px 8px rgba(76, 175, 80, 0.3)',
    transition: 'all 0.3s ease',
  },
  downloadButton: {
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: '#ffffff',
    border: 'none',
    padding: '0.5rem 1rem',
    borderRadius: '8px',
    fontSize: '0.9rem',
    fontWeight: '600',
    cursor: 'pointer',
    boxShadow: '0 2px 8px rgba(102, 126, 234, 0.3)',
    transition: 'all 0.3s ease',
  },
  deleteButton: {
    background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    color: '#ffffff',
    border: 'none',
    padding: '0.5rem 1rem',
    borderRadius: '8px',
    fontSize: '0.9rem',
    fontWeight: '600',
    cursor: 'pointer',
    boxShadow: '0 2px 8px rgba(245, 87, 108, 0.3)',
    transition: 'all 0.3s ease',
  },
  editorBody: {
    display: 'flex',
    flexDirection: 'column',
    padding: '1.5rem',
    gap: '1rem',
    flex: 1,
  },
  documentTextarea: {
    minHeight: '400px',
    borderRadius: '12px',
    border: '2px solid rgba(102, 126, 234, 0.3)',
    padding: '1rem',
    backgroundColor: 'white',
    color: '#0f172a',
    fontSize: '1rem',
    lineHeight: 1.6,
    resize: 'vertical',
    fontFamily: 'inherit',
    transition: 'all 0.3s ease',
  },
  imageViewer: {
    padding: '1.5rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  imageFull: {
    maxWidth: '100%',
    maxHeight: '60vh',
    borderRadius: '12px',
    boxShadow: '0 4px 15px rgba(0, 0, 0, 0.2)',
  },
  imagePrompt: {
    padding: '1rem',
    backgroundColor: 'rgba(102, 126, 234, 0.1)',
    borderRadius: '12px',
    fontSize: '0.9rem',
    color: '#1f2937',
    maxWidth: '800px',
  },
  emptyState: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '3rem 1rem',
    color: '#94a3b8',
    fontWeight: 500,
    textAlign: 'center',
  },
  emptyIcon: {
    fontSize: '3rem',
    display: 'block',
    marginBottom: '1rem',
  },
};

export default AiDocs;
