import React, { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import api from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';

function Blog() {
  const { user } = useAuth();
  const { blogId } = useParams();
  const [blogs, setBlogs] = useState([]);
  const [selectedBlog, setSelectedBlog] = useState(null);
  const [editingBlog, setEditingBlog] = useState(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [saving, setSaving] = useState(false);
  
  const [title, setTitle] = useState('');
  const [body, setBody] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (!user) {
      setLoading(false);
      return;
    }
    fetchBlogs();
  }, [user]);

  useEffect(() => {
    if (blogId && blogs.length > 0) {
      const blog = blogs.find(b => b.id === parseInt(blogId));
      if (blog) {
        setSelectedBlog(blog);
      } else {
        fetchBlogById(blogId);
      }
    }
  }, [blogId, blogs]);

  const fetchBlogById = async (id) => {
    try {
      const res = await api.get(`/api/blogs/${id}`);
      setSelectedBlog(res.data);
    } catch (e) {
      console.error('Failed to load blog:', e);
      setError('Blog not found');
    }
  };

  const fetchBlogs = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.get('/api/blogs');
      setBlogs(res.data?.blogs || []);
    } catch (e) {
      console.error('Failed to load blogs:', e);
      if (e.response?.status === 401) {
        setError('Please log in to view blogs.');
      } else {
        setError('Failed to load blogs');
      }
      setBlogs([]);
    } finally {
      setLoading(false);
    }
  };

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const removeImage = () => {
    setImageFile(null);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!title.trim() || !body.trim()) {
      setError('Title and body are required');
      return;
    }

    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      const formData = new FormData();
      formData.append('title', title.trim());
      formData.append('body', body.trim());
      if (imageFile) {
        formData.append('image', imageFile);
      }

      let blog;
      if (editingBlog) {
        const updateData = { title: title.trim(), body: body.trim() };
        const res = await api.put(`/api/blogs/${editingBlog.id}`, updateData);
        blog = res.data?.blog || res.data;
        
        if (imageFile) {
          const imgFormData = new FormData();
          imgFormData.append('image', imageFile);
          await api.post(`/api/blogs/${editingBlog.id}`, imgFormData);
        }
      } else {
        const res = await api.post('/api/blogs', formData);
        blog = res.data?.blog || res.data;
      }

      setSuccess(editingBlog ? 'Blog updated successfully.' : 'Blog created successfully.');
      await fetchBlogs();
      resetForm();
      if (blog) setSelectedBlog(blog);
      setTimeout(() => setSuccess(null), 3000);
    } catch (e) {
      console.error('Failed to save blog:', e);
      const message = e.response?.data?.error || 'Failed to save blog';
      if (e.response?.status === 401) {
        setError('Please log in to create/edit blogs.');
      } else {
        setError(message);
      }
    } finally {
      setSaving(false);
    }
  };

  const resetForm = () => {
    setTitle('');
    setBody('');
    setImageFile(null);
    setImagePreview(null);
    setShowCreateForm(false);
    setEditingBlog(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleEdit = (blog) => {
    setEditingBlog(blog);
    setTitle(blog.title);
    setBody(blog.body);
    setImageFile(null);
    setImagePreview(blog.image_filename ? `/uploads/blogs/${blog.image_filename}` : null);
    setShowCreateForm(true);
    setError(null);
    setSuccess(null);
  };

  const handleDelete = async (blogId) => {
    if (!window.confirm('Delete this blog post?')) return;
    try {
      await api.delete(`/api/blogs/${blogId}`);
      setSuccess('Blog deleted successfully.');
      await fetchBlogs();
      if (selectedBlog?.id === blogId) setSelectedBlog(null);
      setTimeout(() => setSuccess(null), 3000);
    } catch (e) {
      console.error('Failed to delete blog:', e);
      const message = e.response?.data?.error || 'Failed to delete blog';
      if (e.response?.status === 401) {
        setError('Please log in to delete blogs.');
      } else {
        setError(message);
      }
    }
  };

  const getImageUrl = (filename) => {
    if (!filename) return null;
    return `/uploads/blogs/${filename}`;
  };

  if (!user) {
    return (
      <div style={styles.container}>
        <div style={styles.loginPrompt}>
          <h1 style={styles.title}>Research Blog</h1>
          <p style={styles.promptText}>Please log in to view and create blogs.</p>
        </div>
      </div>
    );
  }

  if (selectedBlog) {
    return (
      <div style={styles.container}>
        <button onClick={() => setSelectedBlog(null)} style={styles.backButton}>← Back to Blogs</button>
        <article style={styles.blogPost}>
          <h1 style={styles.blogTitle}>{selectedBlog.title}</h1>
          <div style={styles.blogMeta}>
            <div style={styles.metaItem}>
              <strong>Published:</strong> {new Date(selectedBlog.created_at).toLocaleDateString('en-US', { 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric' 
              })}
            </div>
          </div>
          {selectedBlog.image_filename && (
            <div style={styles.blogImages}>
              <img 
                src={getImageUrl(selectedBlog.image_filename)} 
                alt={selectedBlog.title} 
                style={styles.blogImage}
                onError={(e) => {
                  console.error('Failed to load blog image:', selectedBlog.image_filename);
                  e.target.style.display = 'none';
                }}
              />
            </div>
          )}
          <div style={styles.blogContent}>{selectedBlog.body}</div>
          {(user && (user.id === selectedBlog.owner_id || user.is_admin)) && (
            <div style={styles.blogActions}>
              <button onClick={() => handleEdit(selectedBlog)} style={styles.editButton}>✏️ Edit</button>
              <button onClick={() => handleDelete(selectedBlog.id)} style={styles.deleteButton}>🗑️ Delete</button>
            </div>
          )}
        </article>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>Research Blog</h1>
        {user && (
          <button onClick={() => { resetForm(); setShowCreateForm(true); }} style={styles.createButton}>
            + Create New Blog
          </button>
        )}
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

      {showCreateForm && (
        <form onSubmit={handleSubmit} style={styles.createForm}>
          <h2 style={styles.formTitle}>{editingBlog ? 'Edit Blog' : 'Create New Blog'}</h2>
          <div style={styles.formGroup}>
            <label style={styles.label}>Title *</label>
            <input
              type="text"
              placeholder="Enter blog title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              style={styles.input}
              required
            />
          </div>
          <div style={styles.formGroup}>
            <label style={styles.label}>Content *</label>
            <textarea
              placeholder="Write your blog content here..."
              value={body}
              onChange={(e) => setBody(e.target.value)}
              style={styles.textarea}
              rows={15}
              required
            />
          </div>
          <div style={styles.imageSection}>
            <label style={styles.fileLabel}>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                style={styles.fileInput}
              />
              📷 {imageFile ? 'Change Image' : 'Add Image'}
            </label>
            {imagePreview && (
              <div style={styles.imagePreview}>
                <img src={imagePreview} alt="Preview" style={styles.previewImg} />
                <button type="button" onClick={removeImage} style={styles.removeImg}>×</button>
              </div>
            )}
          </div>
          <div style={styles.formActions}>
            <button type="submit" disabled={saving} style={styles.submitButton}>
              {saving ? 'Saving...' : (editingBlog ? 'Update Blog' : 'Publish Blog')}
            </button>
            <button type="button" onClick={resetForm} style={styles.cancelButton}>Cancel</button>
          </div>
        </form>
      )}

      {loading && <div style={styles.status}>Loading blogs...</div>}
      
      {!loading && blogs.length === 0 && (
        <div style={styles.emptyState}>
          <span style={styles.emptyIcon}>📝</span>
          <p>No blogs yet. Create your first blog post!</p>
        </div>
      )}
      
          {blogs.length > 0 && (
        <div style={styles.blogGrid}>
          {blogs.map(blog => (
            <div key={blog.id} style={styles.blogCard} onClick={() => setSelectedBlog(blog)}>
              {blog.image_filename && (
                <img 
                  src={getImageUrl(blog.image_filename)} 
                  alt={blog.title} 
                  style={styles.cardImage}
                  onError={(e) => {
                    console.error('Failed to load blog thumbnail:', blog.image_filename);
                    e.target.style.display = 'none';
                  }}
                />
              )}
              <div style={styles.cardContent}>
                <h3 style={styles.cardTitle}>{blog.title}</h3>
                <p style={styles.cardExcerpt}>{blog.body.substring(0, 150)}...</p>
                <div style={styles.cardMeta}>
                  <span>{new Date(blog.created_at).toLocaleDateString()}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const styles = {
  container: { 
    maxWidth: '1400px', 
    margin: '2rem auto', 
    padding: '0 1rem',
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
  promptText: {
    fontSize: '1.1rem',
    color: '#666',
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
  error: { 
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    color: '#d32f2f', 
    margin: '0.5rem 0', 
    padding: '1rem 1.25rem', 
    background: 'rgba(255, 235, 238, 0.95)',
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
    margin: '0.5rem 0',
    padding: '1rem 1.25rem',
    background: 'rgba(232, 245, 233, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '12px',
    border: '1px solid rgba(46, 125, 50, 0.2)',
    boxShadow: '0 4px 15px rgba(46, 125, 50, 0.1)',
  },
  successIcon: {
    fontSize: '1.25rem',
    fontWeight: 'bold',
  },
  createForm: { 
    background: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    padding: '2rem', 
    borderRadius: '20px', 
    marginBottom: '2rem', 
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    border: '1px solid rgba(102, 126, 234, 0.3)',
  },
  formTitle: { 
    marginTop: 0, 
    marginBottom: '1.5rem',
    color: '#333',
    fontSize: '1.5rem',
    fontWeight: '600',
  },
  formGroup: {
    marginBottom: '1.5rem',
  },
  label: {
    display: 'block',
    marginBottom: '0.5rem',
    fontWeight: '600',
    color: '#333',
    fontSize: '0.95rem',
  },
  input: { 
    width: '100%', 
    padding: '0.875rem 1rem', 
    border: '2px solid rgba(102, 126, 234, 0.3)', 
    borderRadius: '12px', 
    fontSize: '1rem', 
    backgroundColor: 'white',
    transition: 'all 0.3s ease',
  },
  textarea: { 
    width: '100%', 
    padding: '0.875rem 1rem', 
    border: '2px solid rgba(102, 126, 234, 0.3)', 
    borderRadius: '12px', 
    fontSize: '1rem', 
    fontFamily: 'inherit',
    backgroundColor: 'white',
    resize: 'vertical',
    transition: 'all 0.3s ease',
  },
  imageSection: { marginBottom: '1.5rem' },
  fileLabel: { 
    display: 'inline-block', 
    padding: '0.75rem 1.5rem', 
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
    color: 'white',
    border: 'none',
    borderRadius: '12px', 
    cursor: 'pointer', 
    marginBottom: '1rem',
    fontSize: '0.95rem',
    fontWeight: '600',
    boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
    transition: 'all 0.3s ease',
  },
  fileInput: { display: 'none' },
  imagePreview: { 
    position: 'relative', 
    display: 'inline-block',
    marginTop: '1rem',
  },
  previewImg: { 
    width: '200px', 
    height: '200px', 
    objectFit: 'cover', 
    borderRadius: '12px',
    border: '2px solid rgba(102, 126, 234, 0.3)',
  },
  removeImg: { 
    position: 'absolute', 
    top: -10, 
    right: -10, 
    background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', 
    color: 'white', 
    border: 'none', 
    borderRadius: '50%', 
    width: '32px', 
    height: '32px', 
    cursor: 'pointer',
    fontSize: '1.2rem',
    fontWeight: 'bold',
    boxShadow: '0 2px 8px rgba(245, 87, 108, 0.4)',
  },
  formActions: { display: 'flex', gap: '1rem' },
  submitButton: { 
    padding: '0.875rem 1.75rem', 
    background: 'linear-gradient(135deg, #4caf50 0%, #45a049 100%)', 
    color: 'white', 
    border: 'none', 
    borderRadius: '12px', 
    cursor: 'pointer', 
    fontSize: '1rem',
    fontWeight: '600',
    boxShadow: '0 4px 15px rgba(76, 175, 80, 0.4)',
    transition: 'all 0.3s ease',
  },
  cancelButton: { 
    padding: '0.875rem 1.75rem', 
    background: 'linear-gradient(135deg, #9e9e9e 0%, #757575 100%)', 
    color: 'white', 
    border: 'none', 
    borderRadius: '12px', 
    cursor: 'pointer', 
    fontSize: '1rem',
    fontWeight: '600',
    transition: 'all 0.3s ease',
  },
  status: { 
    textAlign: 'center', 
    color: 'rgba(255, 255, 255, 0.9)', 
    margin: '2rem 0',
    fontSize: '1.1rem',
  },
  emptyState: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    padding: '3rem',
    textAlign: 'center',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    border: '1px solid rgba(102, 126, 234, 0.3)',
  },
  emptyIcon: {
    fontSize: '3rem',
    display: 'block',
    marginBottom: '1rem',
  },
  blogGrid: { 
    display: 'grid', 
    gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', 
    gap: '1.5rem' 
  },
  blogCard: { 
    background: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '15px', 
    overflow: 'hidden', 
    boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)', 
    cursor: 'pointer', 
    transition: 'all 0.3s ease',
    border: '1px solid rgba(102, 126, 234, 0.3)',
  },
  cardImage: { 
    width: '100%', 
    height: '200px', 
    objectFit: 'cover' 
  },
  cardContent: { padding: '1.25rem' },
  cardTitle: { 
    margin: '0 0 0.5rem 0', 
    color: '#333', 
    fontSize: '1.2rem',
    fontWeight: '600',
  },
  cardExcerpt: { 
    color: '#666', 
    margin: '0 0 0.5rem 0', 
    fontSize: '0.9rem',
    lineHeight: '1.5',
  },
  cardMeta: { 
    fontSize: '0.8rem', 
    color: '#999',
  },
  backButton: { 
    padding: '0.75rem 1.5rem', 
    background: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    border: '1px solid rgba(102, 126, 234, 0.3)', 
    borderRadius: '12px', 
    cursor: 'pointer', 
    marginBottom: '1.5rem',
    fontSize: '1rem',
    fontWeight: '500',
    boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)',
    transition: 'all 0.3s ease',
  },
  blogPost: { 
    background: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    padding: '2rem', 
    borderRadius: '20px', 
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    border: '1px solid rgba(102, 126, 234, 0.3)',
  },
  blogTitle: { 
    marginTop: 0, 
    color: '#333', 
    fontSize: '2.5rem',
    fontWeight: '700',
    marginBottom: '1rem',
  },
  blogMeta: { 
    color: '#666', 
    marginBottom: '1.5rem', 
    fontSize: '0.95rem', 
    display: 'flex', 
    flexDirection: 'column',
    gap: '0.5rem',
    padding: '1rem',
    backgroundColor: 'rgba(102, 126, 234, 0.1)',
    borderRadius: '12px',
    borderLeft: '4px solid #667eea',
  },
  metaItem: {
    display: 'flex',
    gap: '0.5rem',
  },
  blogImages: { 
    marginBottom: '1.5rem', 
    display: 'flex', 
    flexDirection: 'column', 
    gap: '1rem' 
  },
  blogImage: { 
    width: '100%', 
    maxHeight: '500px', 
    objectFit: 'contain', 
    borderRadius: '12px',
    border: '2px solid rgba(102, 126, 234, 0.3)',
  },
  blogContent: { 
    lineHeight: '1.8', 
    color: '#333', 
    fontSize: '1.1rem', 
    whiteSpace: 'pre-wrap',
    marginBottom: '1rem',
  },
  blogActions: { 
    marginTop: '2rem', 
    paddingTop: '1rem', 
    borderTop: '1px solid #eee', 
    display: 'flex', 
    gap: '1rem' 
  },
  editButton: { 
    padding: '0.75rem 1.5rem', 
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
    color: 'white', 
    border: 'none', 
    borderRadius: '12px', 
    cursor: 'pointer',
    fontSize: '0.9rem',
    fontWeight: '600',
    boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
    transition: 'all 0.3s ease',
  },
  deleteButton: { 
    padding: '0.75rem 1.5rem', 
    background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', 
    color: 'white', 
    border: 'none', 
    borderRadius: '12px', 
    cursor: 'pointer',
    fontSize: '0.9rem',
    fontWeight: '600',
    boxShadow: '0 4px 15px rgba(245, 87, 108, 0.4)',
    transition: 'all 0.3s ease',
  }
};

export default Blog;
