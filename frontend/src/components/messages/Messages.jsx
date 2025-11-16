import React, { useEffect, useState, useRef } from 'react';
import api from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';

function Messages() {
  const { user } = useAuth();
  const [allUsers, setAllUsers] = useState([]);
  const [selectedUsername, setSelectedUsername] = useState(null);
  const [thread, setThread] = useState([]);
  const [text, setText] = useState('');
  const [file, setFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [viewingImage, setViewingImage] = useState(null);
  const [hoveredImage, setHoveredImage] = useState(null);
  const bottomRef = useRef(null);
  const fileInputRef = useRef(null);
  const [notificationPermission, setNotificationPermission] = useState('default');

  useEffect(() => {
    if ('Notification' in window) {
      setNotificationPermission(Notification.permission);
    }
  }, []);

  useEffect(() => {
    if (!user) {
      setAllUsers([]);
      return;
    }
    fetchAllUsers();
    const interval = setInterval(fetchAllUsers, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [user]);

  useEffect(() => {
    if (!selectedUsername) {
      setThread([]);
      return;
    }
    fetchThread();
    const interval = setInterval(fetchThread, 3000);
    return () => clearInterval(interval);
  }, [selectedUsername]);

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [thread]);

  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && viewingImage) {
        closeImageViewer();
      }
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [viewingImage]);

  const fetchAllUsers = async () => {
    try {
      const res = await api.get('/api/members');
      const members = res.data?.members || [];
      // Filter out current user
      const otherUsers = members.filter(m => m.id !== user?.id);
      setAllUsers(otherUsers);
      setError(null);
    } catch (e) {
      console.error('Failed to load users:', e);
      if (e.response?.status === 401) {
        setError('Please log in to view users.');
      } else {
        setError('Failed to load users');
      }
    }
  };

  const handleEnableNotifications = async () => {
    if (!('Notification' in window)) {
      setError('Your browser does not support notifications');
      return;
    }

    if (Notification.permission === 'granted') {
      setNotificationPermission('granted');
      setSuccess('Notifications are already enabled!');
      setTimeout(() => setSuccess(null), 2000);
      return;
    }

    try {
      const permission = await Notification.requestPermission();
      setNotificationPermission(permission);
      if (permission === 'granted') {
        setSuccess('Notifications enabled! You will receive alerts for new messages.');
        setTimeout(() => setSuccess(null), 3000);
      } else {
        setError('Notification permission denied. Please enable it in your browser settings.');
      }
    } catch (e) {
      console.error('Failed to request notification permission:', e);
      setError('Failed to enable notifications');
    }
  };

  const fetchThread = async () => {
    if (!selectedUsername) return;
    try {
      const res = await api.get(`/api/messages/${selectedUsername}`);
      const messages = res.data?.messages || [];
      setThread(messages);
      setError(null);
      
      if (messages.length > 0 && Notification.permission === 'granted') {
        const lastMessage = messages[messages.length - 1];
        if (lastMessage.sender_id !== user?.id && !document.hasFocus()) {
          showNotification(`New message from ${selectedUsername}`, {
            body: lastMessage.body || (lastMessage.attachment || lastMessage.attachment_filename ? 'Sent an attachment' : 'Sent a message'),
            icon: '/favicon.ico',
            tag: `message-${lastMessage.id}`,
          });
        }
      }
    } catch (e) {
      console.error('Failed to load conversation:', e);
      if (e.response?.status === 404) {
        setError('User not found');
      } else {
        setError('Failed to load conversation');
      }
    }
  };

  const showNotification = (title, options = {}) => {
    if ('Notification' in window && Notification.permission === 'granted') {
      try {
        const notification = new Notification(title, {
          icon: '/favicon.ico',
          badge: '/favicon.ico',
          ...options
        });
        notification.onclick = () => {
          window.focus();
          notification.close();
        };
        setTimeout(() => notification.close(), 5000);
      } catch (e) {
        console.error('Failed to show notification:', e);
      }
    }
  };

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      if (selectedFile.type.startsWith('image/')) {
        setFilePreview(URL.createObjectURL(selectedFile));
      } else {
        setFilePreview(null);
      }
    }
  };

  const removeFile = () => {
    setFile(null);
    setFilePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const send = async (e) => {
    e.preventDefault();
    if (!selectedUsername) return;
    if (!text.trim() && !file) {
      setError('Please enter a message or attach a file');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);
    try {
      const formData = new FormData();
      formData.append('recipient', selectedUsername);
      if (text.trim()) formData.append('body', text.trim());
      if (file) formData.append('attachment', file);

      await api.post('/api/messages', formData);
      setText('');
      setFile(null);
      setFilePreview(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      setSuccess('Message sent!');
      setTimeout(() => setSuccess(null), 2000);
      await fetchThread();
    } catch (e) {
      console.error('Failed to send message:', e);
      const message = e.response?.data?.error || 'Failed to send message';
      if (e.response?.status === 401) {
        setError('Please log in to send messages.');
      } else if (e.response?.status === 404) {
        setError('Recipient not found');
      } else {
        setError(message);
      }
    } finally {
      setLoading(false);
    }
  };

  const getAttachmentUrl = (filename) => {
    return `/uploads/messages/${filename}`;
  };

  const handleImageClick = (filename) => {
    setViewingImage(filename);
  };

  const closeImageViewer = () => {
    setViewingImage(null);
  };

  const downloadAttachment = (filename) => {
    const url = getAttachmentUrl(filename);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setSuccess(`Downloading ${filename}...`);
    setTimeout(() => setSuccess(null), 2000);
  };

  const getFileIcon = (filename) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    const iconMap = {
      pdf: '📄',
      doc: '📝',
      docx: '📝',
      xls: '📊',
      xlsx: '📊',
      ppt: '📽️',
      pptx: '📽️',
      txt: '📃',
      zip: '📦',
      rar: '📦',
      '7z': '📦',
      mp3: '🎵',
      wav: '🎵',
      mp4: '🎬',
      avi: '🎬',
      mov: '🎬',
      exe: '⚙️',
      dmg: '💿',
      pkg: '📦',
    };
    return iconMap[ext] || '📎';
  };

  if (!user) {
    return (
      <div style={styles.container}>
        <div style={styles.loginPrompt}>
          <h1 style={styles.title}>💬 Messages</h1>
          <p style={styles.promptText}>Please log in to send and receive messages.</p>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>💬 Messages</h1>
        <button
          onClick={handleEnableNotifications}
          style={{
            ...styles.notificationButton,
            ...(notificationPermission === 'granted' ? styles.notificationButtonEnabled : {}),
          }}
          title={
            notificationPermission === 'granted'
              ? 'Notifications enabled'
              : notificationPermission === 'denied'
              ? 'Notifications blocked. Enable in browser settings'
              : 'Enable desktop notifications'
          }
        >
          {notificationPermission === 'granted' ? '🔔 Notifications On' : '🔕 Enable Notifications'}
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

      <div style={styles.layout}>
        <div style={styles.sidebar}>
          <h2 style={styles.sidebarTitle}>All Users</h2>
          {allUsers.length === 0 ? (
            <div style={styles.emptyState}>
              <span style={styles.emptyIcon}>👥</span>
              <p>No other users found.</p>
            </div>
          ) : (
            <div style={styles.partnerList}>
              {allUsers.map((member) => (
                <div
                  key={member.id}
                  style={{
                    ...styles.partnerItem,
                    ...(selectedUsername === member.username ? styles.partnerItemActive : {}),
                  }}
                  onClick={() => setSelectedUsername(member.username)}
                >
                  <div style={styles.partnerAvatar}>
                    {member.username.charAt(0).toUpperCase()}
                  </div>
                  <div style={styles.partnerInfo}>
                    <div style={styles.partnerName}>
                      {member.username}
                      {member.is_admin && <span style={styles.adminBadge}> 👑</span>}
                    </div>
                    <div style={styles.partnerEmail}>{member.email}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div style={styles.chatArea}>
          {selectedUsername ? (
            <>
              <div style={styles.chatHeader}>
                <h3 style={styles.chatTitle}>Chat with {selectedUsername}</h3>
              </div>
              <div style={styles.messagesContainer}>
                {thread.map((msg) => {
                  const isOwn = msg.sender_id === user.id;
                  return (
                    <div
                      key={msg.id}
                      style={{
                        ...styles.message,
                        ...(isOwn ? styles.messageOwn : styles.messageOther),
                      }}
                    >
                      <div style={styles.messageContent}>
                        {msg.body && <div style={styles.messageText}>{msg.body}</div>}
                        {(msg.attachment || msg.attachment_filename) && (
                          <div style={styles.attachmentContainer}>
                            {(() => {
                              const filename = msg.attachment || msg.attachment_filename;
                              const isImage = filename && filename.match(/\.(jpg|jpeg|png|gif|webp)$/i);
                              const fileExtension = filename ? filename.split('.').pop()?.toUpperCase() : 'FILE';
                              
                              if (isImage) {
                                return (
                                  <div style={styles.imageAttachmentWrapper}>
                                    <div 
                                      style={styles.imageWrapper}
                                      onMouseEnter={() => setHoveredImage(filename)}
                                      onMouseLeave={() => setHoveredImage(null)}
                                    >
                                      <img
                                        src={getAttachmentUrl(filename)}
                                        alt="Attachment"
                                        style={styles.attachmentImage}
                                        onClick={() => handleImageClick(filename)}
                                        onError={(e) => {
                                          console.error('Failed to load attachment:', filename);
                                          e.target.style.display = 'none';
                                        }}
                                      />
                                      <div 
                                        style={{
                                          ...styles.imageOverlay,
                                          opacity: hoveredImage === filename ? 1 : 0,
                                        }} 
                                        onClick={() => handleImageClick(filename)}
                                      >
                                        <span style={styles.viewImageText}>👁️ Click to view fullscreen</span>
                                      </div>
                                    </div>
                                    <div style={styles.imageAttachmentInfo}>
                                      <div style={styles.imageFileName}>{filename}</div>
                                      <div style={styles.imageFileType}>{fileExtension} Image</div>
                                      <button
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          downloadAttachment(filename);
                                        }}
                                        style={styles.imageDownloadButton}
                                        title={`Download ${filename}`}
                                      >
                                        ⬇️ Download
                                      </button>
                                    </div>
                                  </div>
                                );
                              } else {
                                // Document (non-image file)
                                return (
                                  <div style={styles.documentContainer}>
                                    <div style={styles.documentInfo}>
                                      <span style={styles.documentIcon}>{getFileIcon(filename)}</span>
                                      <div style={styles.documentDetails}>
                                        <div style={styles.documentName}>{filename}</div>
                                        <div style={styles.documentType}>
                                          {fileExtension} Document
                                        </div>
                                      </div>
                                    </div>
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        downloadAttachment(filename);
                                      }}
                                      style={styles.documentDownloadButton}
                                      title={`Download ${filename}`}
                                    >
                                      ⬇️ Download
                                    </button>
                                  </div>
                                );
                              }
                            })()}
                          </div>
                        )}
                        <div style={styles.messageTime}>
                          {new Date(msg.created_at).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  );
                })}
                <div ref={bottomRef} />
              </div>
              <form onSubmit={send} style={styles.composer}>
                {filePreview && (
                  <div style={styles.filePreview}>
                    <img src={filePreview} alt="Preview" style={styles.previewImage} />
                    <button type="button" onClick={removeFile} style={styles.removeFileButton}>
                      ×
                    </button>
                  </div>
                )}
                {file && !filePreview && (
                  <div style={styles.filePreview}>
                    <span>📎 {file.name}</span>
                    <button type="button" onClick={removeFile} style={styles.removeFileButton}>
                      ×
                    </button>
                  </div>
                )}
                <div style={styles.composerInputs}>
                  <input
                    type="text"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Type a message..."
                    style={styles.textInput}
                    disabled={loading}
                  />
                  <label style={styles.fileLabel}>
                    <input
                      ref={fileInputRef}
                      type="file"
                      onChange={handleFileSelect}
                      style={styles.fileInput}
                      disabled={loading}
                    />
                    📎
                  </label>
                  <button
                    type="submit"
                    disabled={loading || (!text.trim() && !file)}
                    style={{
                      ...styles.sendButton,
                      ...((loading || (!text.trim() && !file)) ? styles.sendButtonDisabled : {}),
                    }}
                  >
                    {loading ? 'Sending...' : 'Send'}
                  </button>
                </div>
              </form>
            </>
          ) : (
            <div style={styles.emptyChat}>
              <span style={styles.emptyChatIcon}>💬</span>
              <p>Select a conversation to start messaging</p>
            </div>
          )}
        </div>
      </div>

      {viewingImage && (
        <div 
          style={styles.imageModal}
          onClick={closeImageViewer}
        >
          <div 
            style={styles.imageModalContent}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={styles.imageModalHeader}>
              <button
                onClick={closeImageViewer}
                style={styles.imageModalClose}
                title="Close (ESC)"
              >
                ×
              </button>
              <button
                onClick={() => downloadAttachment(viewingImage)}
                style={styles.imageModalDownload}
                title="Download image"
              >
                ⬇️ Download
              </button>
            </div>
            <img
              src={getAttachmentUrl(viewingImage)}
              alt="Fullscreen view"
              style={styles.imageModalImage}
              onError={(e) => {
                console.error('Failed to load fullscreen image:', viewingImage);
                setViewingImage(null);
              }}
            />
          </div>
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
  notificationButton: {
    padding: '0.75rem 1.5rem',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '12px',
    cursor: 'pointer',
    fontSize: '1rem',
    fontWeight: '600',
    boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
    transition: 'all 0.3s ease',
    whiteSpace: 'nowrap',
  },
  notificationButtonEnabled: {
    background: 'linear-gradient(135deg, #4caf50 0%, #45a049 100%)',
    boxShadow: '0 4px 15px rgba(76, 175, 80, 0.4)',
  },
  promptText: {
    fontSize: '1.1rem',
    color: '#666',
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
  layout: {
    display: 'grid',
    gridTemplateColumns: '300px 1fr',
    gap: '1.5rem',
    height: 'calc(100vh - 200px)',
  },
  sidebar: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    padding: '1.5rem',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    border: '1px solid rgba(102, 126, 234, 0.3)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  sidebarTitle: {
    fontSize: '1.25rem',
    fontWeight: '600',
    marginBottom: '1rem',
    color: '#333',
  },
  partnerList: {
    flex: 1,
    overflowY: 'auto',
  },
  partnerItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    padding: '1rem',
    borderRadius: '12px',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    marginBottom: '0.5rem',
  },
  partnerItemActive: {
    background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%)',
  },
  partnerAvatar: {
    width: '40px',
    height: '40px',
    borderRadius: '50%',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: '600',
    fontSize: '1.1rem',
  },
  partnerInfo: {
    flex: 1,
  },
  partnerName: {
    fontWeight: '600',
    color: '#333',
    fontSize: '1rem',
  },
  partnerEmail: {
    fontSize: '0.85rem',
    color: '#666',
  },
  adminBadge: {
    fontSize: '0.9rem',
  },
  chatArea: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    border: '1px solid rgba(102, 126, 234, 0.3)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  chatHeader: {
    padding: '1.5rem',
    borderBottom: '1px solid rgba(102, 126, 234, 0.2)',
    background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
  },
  chatTitle: {
    fontSize: '1.25rem',
    fontWeight: '600',
    margin: 0,
    color: '#333',
  },
  messagesContainer: {
    flex: 1,
    overflowY: 'auto',
    padding: '1.5rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
  },
  message: {
    display: 'flex',
    maxWidth: '70%',
  },
  messageOwn: {
    alignSelf: 'flex-end',
  },
  messageOther: {
    alignSelf: 'flex-start',
  },
  messageContent: {
    padding: '0.875rem 1rem',
    borderRadius: '12px',
    background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
    border: '1px solid rgba(102, 126, 234, 0.2)',
  },
  messageText: {
    color: '#333',
    fontSize: '1rem',
    lineHeight: 1.5,
    marginBottom: '0.25rem',
  },
  attachmentContainer: {
    marginTop: '0.5rem',
  },
  imageWrapper: {
    position: 'relative',
    display: 'inline-block',
    cursor: 'pointer',
  },
  imageAttachmentWrapper: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
  },
  imageAttachmentInfo: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '1rem',
    padding: '0.75rem',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: '10px',
    border: '1px solid rgba(102, 126, 234, 0.2)',
    flexWrap: 'wrap',
  },
  imageFileName: {
    fontWeight: '600',
    color: '#333',
    fontSize: '0.9rem',
    flex: 1,
    minWidth: 0,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  imageFileType: {
    fontSize: '0.8rem',
    color: '#666',
    flexShrink: 0,
  },
  imageDownloadButton: {
    padding: '0.5rem 1rem',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '0.85rem',
    fontWeight: '600',
    boxShadow: '0 2px 8px rgba(102, 126, 234, 0.3)',
    transition: 'all 0.3s ease',
    flexShrink: 0,
    whiteSpace: 'nowrap',
  },
  attachmentImage: {
    maxWidth: '300px',
    maxHeight: '300px',
    borderRadius: '8px',
    border: '2px solid rgba(102, 126, 234, 0.2)',
    transition: 'transform 0.2s ease',
    display: 'block',
  },
  imageOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    background: 'linear-gradient(to top, rgba(0, 0, 0, 0.7), transparent)',
    color: 'white',
    padding: '0.5rem',
    borderRadius: '0 0 8px 8px',
    fontSize: '0.85rem',
    textAlign: 'center',
    transition: 'opacity 0.2s ease',
    pointerEvents: 'auto',
    cursor: 'pointer',
  },
  viewImageText: {
    display: 'block',
  },
  documentContainer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '1rem',
    padding: '1rem',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: '12px',
    border: '2px solid rgba(102, 126, 234, 0.3)',
    minWidth: '280px',
    maxWidth: '100%',
    width: '100%',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
    overflow: 'visible',
  },
  documentInfo: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    flex: 1,
    minWidth: 0,
    overflow: 'hidden',
  },
  documentIcon: {
    fontSize: '2rem',
    flexShrink: 0,
  },
  documentDetails: {
    flex: 1,
    minWidth: 0,
  },
  documentName: {
    fontWeight: '600',
    color: '#333',
    fontSize: '0.95rem',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    marginBottom: '0.25rem',
  },
  documentType: {
    fontSize: '0.8rem',
    color: '#666',
  },
  documentDownloadButton: {
    padding: '0.625rem 1.25rem',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '10px',
    cursor: 'pointer',
    fontSize: '0.9rem',
    fontWeight: '600',
    boxShadow: '0 2px 8px rgba(102, 126, 234, 0.3)',
    transition: 'all 0.3s ease',
    flexShrink: 0,
    whiteSpace: 'nowrap',
    minWidth: 'fit-content',
    display: 'inline-block',
    visibility: 'visible',
    opacity: 1,
  },
  attachmentLink: {
    color: '#667eea',
    textDecoration: 'none',
    fontSize: '0.9rem',
    display: 'inline-block',
    padding: '0.5rem',
    borderRadius: '8px',
    background: 'rgba(102, 126, 234, 0.1)',
  },
  messageTime: {
    fontSize: '0.75rem',
    color: '#999',
    marginTop: '0.25rem',
  },
  composer: {
    padding: '1.5rem',
    borderTop: '1px solid rgba(102, 126, 234, 0.2)',
    background: 'rgba(102, 126, 234, 0.4)',
  },
  filePreview: {
    position: 'relative',
    display: 'inline-block',
    marginBottom: '0.75rem',
  },
  previewImage: {
    maxWidth: '200px',
    maxHeight: '200px',
    borderRadius: '8px',
    border: '2px solid rgba(102, 126, 234, 0.2)',
  },
  removeFileButton: {
    position: 'absolute',
    top: '-10px',
    right: '-10px',
    background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '50%',
    width: '24px',
    height: '24px',
    cursor: 'pointer',
    fontSize: '1rem',
    fontWeight: 'bold',
  },
  composerInputs: {
    display: 'flex',
    gap: '0.75rem',
    alignItems: 'center',
  },
  textInput: {
    flex: 1,
    padding: '0.875rem 1rem',
    border: '2px solid rgba(102, 126, 234, 0.3)',
    borderRadius: '12px',
    fontSize: '1rem',
    backgroundColor: 'white',
  },
  fileLabel: {
    padding: '0.875rem 1rem',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    borderRadius: '12px',
    cursor: 'pointer',
    fontSize: '1.25rem',
    boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
  },
  fileInput: {
    display: 'none',
  },
  sendButton: {
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
  sendButtonDisabled: {
    background: 'linear-gradient(135deg, #94a3b8 0%, #757575 100%)',
    cursor: 'not-allowed',
    boxShadow: 'none',
  },
  emptyState: {
    textAlign: 'center',
    padding: '2rem',
    color: '#999',
  },
  emptyIcon: {
    fontSize: '2rem',
    display: 'block',
    marginBottom: '0.5rem',
  },
  emptyChat: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#999',
  },
  emptyChatIcon: {
    fontSize: '4rem',
    marginBottom: '1rem',
  },
  imageModal: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.95)',
    backdropFilter: 'blur(5px)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 10000,
    padding: '2rem',
  },
  imageModalContent: {
    position: 'relative',
    maxWidth: '90vw',
    maxHeight: '90vh',
    display: 'flex',
    flexDirection: 'column',
    backgroundColor: 'rgba(102, 126, 234, 0.15)',
    borderRadius: '20px',
    overflow: 'hidden',
  },
  imageModalHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '1rem 1.5rem',
    background: 'rgba(0, 0, 0, 0.7)',
    backdropFilter: 'blur(10px)',
  },
  imageModalClose: {
    background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '50%',
    width: '40px',
    height: '40px',
    cursor: 'pointer',
    fontSize: '1.5rem',
    fontWeight: 'bold',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: '0 4px 15px rgba(245, 87, 108, 0.4)',
    transition: 'all 0.3s ease',
  },
  imageModalDownload: {
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '12px',
    padding: '0.75rem 1.5rem',
    cursor: 'pointer',
    fontSize: '1rem',
    fontWeight: '600',
    boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
    transition: 'all 0.3s ease',
  },
  imageModalImage: {
    maxWidth: '100%',
    maxHeight: 'calc(90vh - 80px)',
    objectFit: 'contain',
    display: 'block',
  },
};

export default Messages;
