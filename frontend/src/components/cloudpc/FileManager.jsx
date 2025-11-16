import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import './FileManager.css';

function FileManager({ pcId }) {
  const [files, setFiles] = useState([]);
  const [currentPath, setCurrentPath] = useState('/');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showCreateFile, setShowCreateFile] = useState(false);
  const [showCreateFolder, setShowCreateFolder] = useState(false);
  const [newFileName, setNewFileName] = useState('');
  const [newFolderName, setNewFolderName] = useState('');
  const [editingFile, setEditingFile] = useState(null);
  const [fileContent, setFileContent] = useState('');
  const [uploading, setUploading] = useState(false);
  const [popup, setPopup] = useState(null); // { type: 'info'|'error'|'success', message: string }
  const [showBrowseDialog, setShowBrowseDialog] = useState(false);
  const [selectedFilePath, setSelectedFilePath] = useState('');
  const [imagePreview, setImagePreview] = useState(null);
  const [renamingFile, setRenamingFile] = useState(null);
  const [renameFileName, setRenameFileName] = useState('');

  const showPopup = (type, message) => {
    setPopup({ type, message });
    setTimeout(() => setPopup(null), 3000);
  };

  useEffect(() => {
    loadFiles();
  }, [currentPath]);

  const loadFiles = async () => {
    try {
      setLoading(true);
      const { data } = await api.get(`/api/cloud-pcs/${pcId}/files`, {
        params: { path: currentPath }
      });
      setFiles(data.files || []);
    } catch (err) {
      const errorMsg = err.response?.data?.error || 'Failed to load files';
      setError(errorMsg);
      showPopup('error', errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleFileClick = async (file) => {
    if (file.type === 'directory') {
      setCurrentPath(currentPath === '/' ? `/${file.name}` : `${currentPath}/${file.name}`);
    } else {
      // Try to open any file
      const filePath = currentPath === '/' ? `/${file.name}` : `${currentPath}/${file.name}`;
      
      // Check if it's a text file that can be edited
      const textExtensions = ['.txt', '.md', '.json', '.js', '.jsx', '.css', '.html', '.xml', '.yaml', '.yml', '.log'];
      const isTextFile = textExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
      
      if (isTextFile) {
        try {
          const { data } = await api.get(`/api/cloud-pcs/${pcId}/files/read`, {
            params: { path: filePath }
          });
          setEditingFile(filePath);
          setFileContent(data.content || '');
        } catch (err) {
          showPopup('error', err.response?.data?.error || 'Failed to read file');
        }
      } else {
        // For non-text files (images, binaries), try to download or display
        try {
          const { data } = await api.get(`/api/cloud-pcs/${pcId}/files/read`, {
            params: { path: filePath }
          });
          
          // Check if it's an image file (by extension or if it's binary with image mime type)
          const isImage = file.name.toLowerCase().match(/\.(png|jpg|jpeg|gif|webp|svg|bmp)$/) || 
                         (data.is_binary && data.mime_type && data.mime_type.startsWith('image/'));
          
          if (data.is_binary && isImage) {
            // Display image in a modal within the VM
            const imageUrl = `data:${data.mime_type || 'image/png'};base64,${data.content}`;
            setImagePreview({ url: imageUrl, name: file.name });
          } else if (data.is_binary) {
            // For other binary files, download them
            const blob = new Blob([Uint8Array.from(atob(data.content), c => c.charCodeAt(0))], { type: data.mime_type || 'application/octet-stream' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = file.name;
            a.click();
            URL.revokeObjectURL(url);
            showPopup('success', `Downloaded ${file.name}`);
          } else {
            showPopup('info', `File: ${file.name}\nSize: ${(file.size / 1024).toFixed(2)} KB`);
          }
        } catch (err) {
          showPopup('error', err.response?.data?.error || `Cannot open ${file.name}. This file type is not supported for viewing.`);
        }
      }
    }
  };

  const handleCreateFile = async (e) => {
    e.preventDefault();
    if (!newFileName.trim()) return;

    try {
      const filename = newFileName.endsWith('.txt') ? newFileName : `${newFileName}.txt`;
      await api.post(`/api/cloud-pcs/${pcId}/files/create`, {
        path: currentPath,
        name: filename,
        type: 'file',
        content: ''
      });
      setShowCreateFile(false);
      setNewFileName('');
      loadFiles();
      showPopup('success', `File "${filename}" created successfully`);
    } catch (err) {
      const errorMsg = err.response?.data?.error || 'Failed to create file';
      showPopup('error', errorMsg);
    }
  };

  const handleCreateFolder = async (e) => {
    e.preventDefault();
    if (!newFolderName.trim()) return;

    try {
      await api.post(`/api/cloud-pcs/${pcId}/files/create`, {
        path: currentPath,
        name: newFolderName.trim(),
        type: 'directory'
      });
      setShowCreateFolder(false);
      setNewFolderName('');
      loadFiles();
      showPopup('success', `Folder "${newFolderName.trim()}" created successfully`);
    } catch (err) {
      const errorMsg = err.response?.data?.error || 'Failed to create folder';
      showPopup('error', errorMsg);
    }
  };

  const handleSaveFile = async () => {
    if (!editingFile) return;

    try {
      await api.put(`/api/cloud-pcs/${pcId}/files`, {
        path: editingFile,
        content: fileContent
      });
      setEditingFile(null);
      setFileContent('');
      loadFiles();
      showPopup('success', 'File saved successfully');
    } catch (err) {
      const errorMsg = err.response?.data?.error || 'Failed to save file';
      showPopup('error', errorMsg);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('path', currentPath);

    try {
      await api.post(`/api/cloud-pcs/${pcId}/files/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      loadFiles();
      showPopup('success', `File "${file.name}" uploaded successfully`);
    } catch (err) {
      const errorMsg = err.response?.data?.error || 'Failed to upload file';
      showPopup('error', errorMsg);
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  const [showDeleteConfirm, setShowDeleteConfirm] = useState(null);

  const handleRenameFile = async (file, newName) => {
    if (!newName.trim() || newName.trim() === file.name) {
      setRenamingFile(null);
      setRenameFileName('');
      return;
    }

    const oldPath = currentPath === '/' ? `/${file.name}` : `${currentPath}/${file.name}`;
    const newPath = currentPath === '/' ? `/${newName.trim()}` : `${currentPath}/${newName.trim()}`;
    
    try {
      await api.put(`/api/cloud-pcs/${pcId}/files/rename`, {
        old_path: oldPath,
        new_path: newPath
      });
      loadFiles();
      showPopup('success', `${file.type === 'directory' ? 'Folder' : 'File'} renamed successfully`);
    } catch (err) {
      showPopup('error', err.response?.data?.error || 'Failed to rename file');
    }
    setRenamingFile(null);
    setRenameFileName('');
  };

  const handleDeleteFile = async (file) => {
    const filePath = currentPath === '/' ? `/${file.name}` : `${currentPath}/${file.name}`;
    try {
      await api.delete(`/api/cloud-pcs/${pcId}/files`, {
        data: { path: filePath }
      });
      loadFiles();
      showPopup('success', `${file.type === 'directory' ? 'Folder' : 'File'} deleted successfully`);
    } catch (err) {
      showPopup('error', err.response?.data?.error || 'Failed to delete file');
    }
    setShowDeleteConfirm(null);
  };

  const handleRightClick = (e, file) => {
    e.preventDefault();
    // Right-click menu could be added here
  };

  const navigateUp = () => {
    if (currentPath === '/') return;
    const parts = currentPath.split('/').filter(p => p);
    parts.pop();
    setCurrentPath(parts.length === 0 ? '/' : '/' + parts.join('/'));
  };

  if (editingFile) {
    return (
      <div className="file-manager">
        <div className="file-manager-header">
          <button onClick={() => { setEditingFile(null); setFileContent(''); }} className="back-button">
            ← Back
          </button>
          <span className="editing-file-name">{editingFile.split('/').pop()}</span>
          <button onClick={handleSaveFile} className="save-button">Save</button>
        </div>
        <textarea
          value={fileContent}
          onChange={(e) => setFileContent(e.target.value)}
          className="file-editor"
          placeholder="File content..."
        />
      </div>
    );
  }

  return (
    <div className="file-manager">
      <div className="file-manager-header">
        <div className="path-navigation">
          <button onClick={navigateUp} disabled={currentPath === '/'} className="nav-button">
            ↑
          </button>
          <span className="current-path">{currentPath || '/'}</span>
        </div>
        <div className="file-manager-actions">
          <label className="upload-button">
            {uploading ? 'Uploading...' : '📤 Upload File'}
            <input type="file" onChange={handleFileUpload} style={{ display: 'none' }} />
          </label>
          <button onClick={() => setShowBrowseDialog(true)} className="browse-button">
            📂 Browse
          </button>
          <button onClick={() => setShowCreateFile(true)} className="create-button">
            📄 New File
          </button>
          <button onClick={() => setShowCreateFolder(true)} className="create-button">
            📁 New Folder
          </button>
        </div>
      </div>

      {error && (
        <div className="file-manager-error">{error}</div>
      )}

      {showCreateFile && (
        <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) setShowCreateFile(false); }}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Create New File</h3>
            <form onSubmit={handleCreateFile}>
              <input
                type="text"
                value={newFileName}
                onChange={(e) => setNewFileName(e.target.value)}
                placeholder="filename.txt"
                className="modal-input"
                autoFocus
              />
              <div className="modal-actions">
                <button type="submit" className="modal-submit">Create</button>
                <button type="button" onClick={() => setShowCreateFile(false)} className="modal-cancel">Cancel</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {showCreateFolder && (
        <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) setShowCreateFolder(false); }}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Create New Folder</h3>
            <form onSubmit={handleCreateFolder}>
              <input
                type="text"
                value={newFolderName}
                onChange={(e) => setNewFolderName(e.target.value)}
                placeholder="Folder name"
                className="modal-input"
                autoFocus
              />
              <div className="modal-actions">
                <button type="submit" className="modal-submit">Create</button>
                <button type="button" onClick={() => setShowCreateFolder(false)} className="modal-cancel">Cancel</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {showBrowseDialog && (
        <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) setShowBrowseDialog(false); }}>
          <div className="modal browse-modal" onClick={(e) => e.stopPropagation()}>
            <h3>Browse Files</h3>
            <div className="browse-content">
              <div className="files-list">
                {files.map((file, index) => (
                  <div
                    key={index}
                    className={`browse-file-item ${file.type}`}
                    onClick={() => {
                      if (file.type === 'directory') {
                        setCurrentPath(currentPath === '/' ? `/${file.name}` : `${currentPath}/${file.name}`);
                        setShowBrowseDialog(false);
                      } else {
                        setSelectedFilePath(currentPath === '/' ? `/${file.name}` : `${currentPath}/${file.name}`);
                      }
                    }}
                  >
                    <span className="browse-file-icon">
                      {file.type === 'directory' ? '📁' : getFileIcon(file.name)}
                    </span>
                    <span className="browse-file-name">{file.name}</span>
                    {file.type === 'file' && (
                      <span className="browse-file-size">{(file.size / 1024).toFixed(2)} KB</span>
                    )}
                  </div>
                ))}
              </div>
              <div className="current-path-display">
                <strong>Current Path:</strong> {currentPath || '/'}
              </div>
            </div>
            <div className="modal-actions">
              <button type="button" onClick={() => setShowBrowseDialog(false)} className="modal-cancel">Close</button>
            </div>
          </div>
        </div>
      )}

      {renamingFile && (
        <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) { setRenamingFile(null); setRenameFileName(''); } }}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Rename {renamingFile.type === 'directory' ? 'Folder' : 'File'}</h3>
            <input
              type="text"
              value={renameFileName}
              onChange={(e) => setRenameFileName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleRenameFile(renamingFile, renameFileName);
                } else if (e.key === 'Escape') {
                  setRenamingFile(null);
                  setRenameFileName('');
                }
              }}
              className="modal-input"
              autoFocus
              placeholder="Enter new name"
            />
            <div className="modal-actions">
              <button
                className="modal-submit"
                onClick={() => handleRenameFile(renamingFile, renameFileName)}
              >
                Rename
              </button>
              <button
                className="modal-cancel"
                onClick={() => { setRenamingFile(null); setRenameFileName(''); }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {showDeleteConfirm && (
        <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) setShowDeleteConfirm(null); }}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Delete {showDeleteConfirm.type === 'directory' ? 'Folder' : 'File'}?</h3>
            <p>Are you sure you want to delete "{showDeleteConfirm.name}"?</p>
            <div className="modal-actions">
              <button
                className="modal-submit delete-confirm-button"
                onClick={() => handleDeleteFile(showDeleteConfirm)}
              >
                Delete
              </button>
              <button
                className="modal-cancel"
                onClick={() => setShowDeleteConfirm(null)}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {imagePreview && (
        <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) setImagePreview(null); }}>
          <div className="image-preview-modal" onClick={(e) => e.stopPropagation()}>
            <div className="image-preview-header">
              <h3>{imagePreview.name}</h3>
              <button className="image-preview-close" onClick={() => setImagePreview(null)}>×</button>
            </div>
            <div className="image-preview-content">
              <img src={imagePreview.url} alt={imagePreview.name} />
            </div>
          </div>
        </div>
      )}

      {popup && (
        <div className={`popup popup-${popup.type}`}>
          <div className="popup-content">
            <span className="popup-icon">
              {popup.type === 'success' ? '✅' : popup.type === 'error' ? '❌' : 'ℹ️'}
            </span>
            <span className="popup-message">{popup.message}</span>
            <button className="popup-close" onClick={() => setPopup(null)}>×</button>
          </div>
        </div>
      )}

      <div className="file-manager-content">
        {loading ? (
          <div className="loading">Loading files...</div>
        ) : files.length === 0 ? (
          <div className="empty-folder">
            <div className="empty-icon">📁</div>
            <p>This folder is empty</p>
            <p className="empty-hint">Upload files or create new ones</p>
          </div>
        ) : (
          <div className="files-grid">
            {files.map((file, index) => (
              <div
                key={index}
                className={`file-item ${file.type}`}
                onClick={() => handleFileClick(file)}
                onContextMenu={(e) => handleRightClick(e, file)}
              >
                <div className="file-icon">
                  {file.type === 'directory' ? '📁' : getFileIcon(file.name)}
                </div>
                <div className="file-name" title={file.name}>{file.name}</div>
                {file.type === 'file' && (
                  <div className="file-size">{(file.size / 1024).toFixed(2)} KB</div>
                )}
                <div className="file-actions">
                  <button
                    className="file-rename-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setRenamingFile(file);
                      setRenameFileName(file.name);
                    }}
                    title="Rename"
                  >
                    ✏️
                  </button>
                  <button
                    className="file-delete-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setShowDeleteConfirm(file);
                    }}
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

function getFileIcon(filename) {
  if (filename.endsWith('.txt') || filename.endsWith('.md')) return '📄';
  if (filename.endsWith('.png') || filename.endsWith('.jpg') || filename.endsWith('.jpeg')) return '🖼️';
  if (filename.endsWith('.pdf')) return '📕';
  return '📄';
}

export default FileManager;

