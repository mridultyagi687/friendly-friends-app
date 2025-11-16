import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import api from '../../services/api';
import './AIAppBuilder.css';

function AIAppBuilder({ pcId }) {
  const { user } = useAuth();
  const [myApps, setMyApps] = useState([]);
  const [liveApps, setLiveApps] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedApp, setSelectedApp] = useState(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newAppName, setNewAppName] = useState('');
  const [newAppDescription, setNewAppDescription] = useState('');
  const [generating, setGenerating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [popup, setPopup] = useState(null);
  const [activeTab, setActiveTab] = useState('my-apps'); // 'my-apps' or 'community'
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [sendingMessage, setSendingMessage] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(null);
  const chatMessagesEndRef = useRef(null);

  useEffect(() => {
    loadApps();
  }, []);

  // Auto-scroll chat to bottom when new messages arrive
  useEffect(() => {
    if (showChat && chatMessagesEndRef.current) {
      chatMessagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatMessages, showChat, sendingMessage]);

  const showPopupMessage = (type, message) => {
    setPopup({ type, message });
    setTimeout(() => setPopup(null), 3000);
  };

  const loadApps = async () => {
    try {
      setLoading(true);
      const { data } = await api.get('/api/ai-apps');
      setMyApps(data.my_apps || []);
      setLiveApps(data.live_apps || []);
    } catch (err) {
      showPopupMessage('error', err.response?.data?.error || 'Failed to load apps');
    } finally {
      setLoading(false);
    }
  };

  const handleCreateApp = async (e) => {
    e.preventDefault();
    if (!newAppName.trim()) {
      showPopupMessage('error', 'App name is required');
      return;
    }

    try {
      const { data } = await api.post('/api/ai-apps', {
        name: newAppName,
        description: newAppDescription
      });
      setMyApps([data.app, ...myApps]);
      setSelectedApp(data.app);
      setShowCreateModal(false);
      setNewAppName('');
      setNewAppDescription('');
      showPopupMessage('success', 'App created successfully!');
    } catch (err) {
      showPopupMessage('error', err.response?.data?.error || 'Failed to create app');
    }
  };

  const handleGenerateCode = async () => {
    if (!selectedApp) return;

    setGenerating(true);
    try {
      const prompt = selectedApp.description || `Create a ${selectedApp.name} app`;
      const { data } = await api.post(`/api/ai-apps/${selectedApp.id}/generate`, {
        prompt: prompt,
        pc_id: pcId // Pass Cloud PC ID so AI knows the context
      });
      
      setSelectedApp({ ...selectedApp, code: data.code });
      showPopupMessage('success', 'Code generated successfully!');
    } catch (err) {
      showPopupMessage('error', err.response?.data?.error || 'Failed to generate code');
    } finally {
      setGenerating(false);
    }
  };

  const handleSaveCode = async () => {
    if (!selectedApp) return;

    setSaving(true);
    try {
      await api.put(`/api/ai-apps/${selectedApp.id}`, {
        code: selectedApp.code
      });
      showPopupMessage('success', 'Code saved successfully!');
    } catch (err) {
      showPopupMessage('error', err.response?.data?.error || 'Failed to save code');
    } finally {
      setSaving(false);
    }
  };

  const handleGoLive = async () => {
    if (!selectedApp) return;

    if (!selectedApp.code || !selectedApp.code.trim()) {
      showPopupMessage('error', 'Cannot go live: App code is empty. Generate code first.');
      return;
    }

    try {
      const { data } = await api.post(`/api/ai-apps/${selectedApp.id}/go-live`);
      setSelectedApp({ ...selectedApp, is_live: true, live_at: data.app.live_at });
      setMyApps(myApps.map(app => app.id === selectedApp.id ? { ...app, is_live: true, live_at: data.app.live_at } : app));
      loadApps(); // Reload to update live apps list
      showPopupMessage('success', '🎉 Your app is now live and visible to everyone!');
    } catch (err) {
      showPopupMessage('error', err.response?.data?.error || 'Failed to make app live');
    }
  };

  const handleDeleteApp = async (appId) => {
    setShowDeleteConfirm(appId);
  };

  const confirmDeleteApp = async () => {
    const appId = showDeleteConfirm;
    setShowDeleteConfirm(null);
    
    try {
      await api.delete(`/api/ai-apps/${appId}`);
      setMyApps(myApps.filter(app => app.id !== appId));
      if (selectedApp && selectedApp.id === appId) {
        setSelectedApp(null);
        setChatMessages([]);
        setShowChat(false);
      }
      showPopupMessage('success', 'App deleted successfully');
    } catch (err) {
      showPopupMessage('error', err.response?.data?.error || 'Failed to delete app');
    }
  };

  const handleSelectApp = async (appId) => {
    try {
      const { data } = await api.get(`/api/ai-apps/${appId}`);
      setSelectedApp(data.app);
      loadChatMessages(appId);
      setShowChat(true); // Show chat by default when app is selected
    } catch (err) {
      showPopupMessage('error', err.response?.data?.error || 'Failed to load app');
    }
  };

  const loadChatMessages = async (appId) => {
    try {
      const { data } = await api.get(`/api/ai-apps/${appId}/chat`);
      setChatMessages(data.messages || []);
    } catch (err) {
      // Chat might not exist yet, that's okay
      setChatMessages([]);
    }
  };

  const handleSendChatMessage = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() || !selectedApp || sendingMessage) return;

    const message = chatInput.trim();
    setChatInput('');
    setSendingMessage(true);

    // Add user message optimistically
    const userMsg = {
      id: Date.now(),
      message: message,
      is_user_message: true,
      created_at: new Date().toISOString()
    };
    setChatMessages([...chatMessages, userMsg]);

    try {
      const { data } = await api.post(`/api/ai-apps/${selectedApp.id}/chat`, {
        message: message,
        pc_id: pcId // Pass Cloud PC ID so AI knows the context
      });

      // Replace optimistic message with real one and add AI response
      setChatMessages(prev => {
        const filtered = prev.filter(m => m.id !== userMsg.id);
        return [...filtered, data.user_message, data.ai_response];
      });

      // Extract and apply code automatically if AI provided it
      const aiResponseText = data.ai_response?.response || data.ai_response?.message || '';
      let extractedCode = null;

      // Try multiple patterns to extract HTML code
      // Pattern 1: ```html ... ```
      const htmlCodeBlockMatch = aiResponseText.match(/```html\s*\n?([\s\S]*?)\n?```/);
      if (htmlCodeBlockMatch && htmlCodeBlockMatch[1]) {
        extractedCode = htmlCodeBlockMatch[1].trim();
      } else {
        // Pattern 2: ``` ... ``` (generic code block)
        const genericCodeBlockMatch = aiResponseText.match(/```\s*\n?([\s\S]*?)\n?```/);
        if (genericCodeBlockMatch && genericCodeBlockMatch[1]) {
          const code = genericCodeBlockMatch[1].trim();
          // Check if it looks like HTML (contains HTML tags)
          if (code.includes('<html') || code.includes('<!DOCTYPE') || code.includes('<div') || code.includes('<body')) {
            extractedCode = code;
          }
        } else {
          // Pattern 3: Look for HTML content directly (between <html> tags or <!DOCTYPE)
          const htmlDirectMatch = aiResponseText.match(/(<!DOCTYPE[\s\S]*?<\/html>|<html[\s\S]*?<\/html>)/i);
          if (htmlDirectMatch && htmlDirectMatch[1]) {
            extractedCode = htmlDirectMatch[1].trim();
          }
        }
      }

      // If code was extracted, update and save automatically
      if (extractedCode) {
        const updatedApp = { ...selectedApp, code: extractedCode };
        setSelectedApp(updatedApp);
        
        // Auto-save the code
        try {
          await api.put(`/api/ai-apps/${selectedApp.id}`, {
            code: extractedCode
          });
          showPopupMessage('success', '✅ Code generated and saved automatically!');
        } catch (saveErr) {
          showPopupMessage('warning', 'Code generated but failed to save. Please save manually.');
        }
      }
    } catch (err) {
      showPopupMessage('error', err.response?.data?.error || 'Failed to send message');
      // Remove optimistic message on error
      setChatMessages(prev => prev.filter(m => m.id !== userMsg.id));
    } finally {
      setSendingMessage(false);
    }
  };

  const handleUpdateDescription = async () => {
    if (!selectedApp) return;

    try {
      await api.put(`/api/ai-apps/${selectedApp.id}`, {
        description: selectedApp.description
      });
      showPopupMessage('success', 'Description updated');
    } catch (err) {
      showPopupMessage('error', err.response?.data?.error || 'Failed to update description');
    }
  };

  if (loading) {
    return <div className="ai-app-builder loading">Loading AI App Builder...</div>;
  }

  return (
    <div className="ai-app-builder">
      <div className="ai-app-builder-header">
        <h2>🤖 AI App Builder</h2>
        <button className="create-app-button" onClick={() => setShowCreateModal(true)}>
          ➕ Create New App
        </button>
      </div>

      <div className="ai-app-builder-content">
        <div className="apps-sidebar">
          <div className="tabs">
            <button 
              className={`tab ${activeTab === 'my-apps' ? 'active' : ''}`}
              onClick={() => setActiveTab('my-apps')}
            >
              My Apps ({myApps.length})
            </button>
            <button 
              className={`tab ${activeTab === 'community' ? 'active' : ''}`}
              onClick={() => setActiveTab('community')}
            >
              Community ({liveApps.length})
            </button>
          </div>

          <div className="apps-list">
            {activeTab === 'my-apps' ? (
              myApps.length === 0 ? (
                <div className="empty-state">
                  <p>No apps yet. Create your first app!</p>
                </div>
              ) : (
                myApps.map((app) => (
                  <div
                    key={app.id}
                    className={`app-item ${selectedApp?.id === app.id ? 'selected' : ''}`}
                    onClick={() => handleSelectApp(app.id)}
                  >
                    <div className="app-item-header">
                      <span className="app-item-name">{app.name}</span>
                      {app.is_live && <span className="live-badge">LIVE</span>}
                    </div>
                    {app.description && (
                      <div className="app-item-description">{app.description}</div>
                    )}
                    {selectedApp?.id === app.id && (
                      <button
                        className="delete-app-button"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteApp(app.id);
                        }}
                      >
                        🗑️ Delete
                      </button>
                    )}
                  </div>
                ))
              )
            ) : (
              liveApps.length === 0 ? (
                <div className="empty-state">
                  <p>No live apps yet. Be the first to publish!</p>
                </div>
              ) : (
                liveApps.map((app) => (
                  <div
                    key={app.id}
                    className={`app-item ${selectedApp?.id === app.id ? 'selected' : ''}`}
                    onClick={() => handleSelectApp(app.id)}
                  >
                    <div className="app-item-header">
                      <span className="app-item-name">{app.name}</span>
                      <span className="live-badge">LIVE</span>
                    </div>
                    {app.description && (
                      <div className="app-item-description">{app.description}</div>
                    )}
                  </div>
                ))
              )
            )}
          </div>
        </div>

        <div className="app-editor">
          {selectedApp ? (
            <>
              <div className="editor-header">
                <div className="app-info">
                  <input
                    type="text"
                    value={selectedApp.name}
                    onChange={(e) => setSelectedApp({ ...selectedApp, name: e.target.value })}
                    onBlur={() => api.put(`/api/ai-apps/${selectedApp.id}`, { name: selectedApp.name })}
                    className="app-name-input"
                  />
                  <textarea
                    value={selectedApp.description || ''}
                    onChange={(e) => setSelectedApp({ ...selectedApp, description: e.target.value })}
                    onBlur={handleUpdateDescription}
                    placeholder="Describe what your app should do..."
                    className="app-description-input"
                    rows="2"
                  />
                </div>
                <div className="editor-actions">
                  {selectedApp.developer_id === user?.id && (
                    <>
                      <button
                        className="generate-button"
                        onClick={handleGenerateCode}
                        disabled={generating}
                      >
                        {generating ? '⏳ Generating...' : '✨ Generate Code'}
                      </button>
                      <button
                        className="save-button"
                        onClick={handleSaveCode}
                        disabled={saving || !selectedApp.code}
                      >
                        {saving ? '💾 Saving...' : '💾 Save'}
                      </button>
                      {!selectedApp.is_live && (
                        <button
                          className="go-live-button"
                          onClick={handleGoLive}
                          disabled={!selectedApp.code || !selectedApp.code.trim()}
                        >
                          🚀 Go Live
                        </button>
                      )}
                      {selectedApp.is_live && (
                        <span className="live-status">✅ Live</span>
                      )}
                    </>
                  )}
                </div>
              </div>

              <div className="editor-tabs">
                <button
                  className={`editor-tab ${showChat ? 'active' : ''}`}
                  onClick={() => setShowChat(true)}
                >
                  💬 Chat
                </button>
                <button
                  className={`editor-tab ${!showChat ? 'active' : ''}`}
                  onClick={() => setShowChat(false)}
                >
                  👁️ Preview
                </button>
              </div>

              <div className="editor-content">
                {showChat ? (
                  <div className="chat-container">
                    <div className="chat-messages">
                      {chatMessages.length === 0 ? (
                        <div className="chat-empty">
                          <div className="chat-empty-icon">💬</div>
                          <h3>Start developing your app</h3>
                          <p>Chat with AI to build, fix bugs, and improve your app. Describe what you want to create or ask for help!</p>
                          {selectedApp.developer_id === user?.id && !selectedApp.code && (
                            <p style={{ marginTop: '1rem', fontStyle: 'italic' }}>
                              💡 Tip: Start by describing what your app should do, and AI will generate the code for you.
                            </p>
                          )}
                        </div>
                      ) : (
                        chatMessages.map((msg, idx) => (
                          <div key={idx} className={`chat-message ${msg.is_user_message ? 'user-message' : 'ai-message'}`}>
                            <div className="chat-message-header">
                              <span className="chat-message-author">
                                {msg.is_user_message ? 'You' : 'AI Assistant'}
                              </span>
                              {msg.created_at && (
                                <span className="chat-message-time">
                                  {new Date(msg.created_at).toLocaleTimeString()}
                                </span>
                              )}
                            </div>
                            <div className="chat-message-content">
                              {msg.is_user_message ? msg.message : (msg.response || msg.message)}
                            </div>
                          </div>
                        ))
                      )}
                      {sendingMessage && (
                        <div className="chat-message ai-message">
                          <div className="chat-message-header">
                            <span className="chat-message-author">AI Assistant</span>
                          </div>
                          <div className="chat-message-content">
                            <span className="typing-indicator">Typing...</span>
                          </div>
                        </div>
                      )}
                      <div ref={chatMessagesEndRef} />
                    </div>
                    <form className="chat-input-form" onSubmit={handleSendChatMessage}>
                      <input
                        type="text"
                        value={chatInput}
                        onChange={(e) => setChatInput(e.target.value)}
                        placeholder={selectedApp.developer_id === user?.id 
                          ? "Describe what you want to build, ask for fixes, or request features..." 
                          : "Chat about this app..."}
                        className="chat-input"
                        disabled={sendingMessage || !selectedApp}
                      />
                      <button
                        type="submit"
                        className="chat-send-button"
                        disabled={!chatInput.trim() || sendingMessage || !selectedApp}
                      >
                        {sendingMessage ? '⏳' : '📤'}
                      </button>
                    </form>
                  </div>
                ) : (
                  selectedApp.code ? (
                    <div className="preview-container">
                      <iframe
                        srcDoc={selectedApp.code}
                        title="App Preview"
                        className="app-preview"
                        sandbox="allow-scripts allow-same-origin"
                      />
                    </div>
                  ) : (
                    <div className="no-code-message">
                      <div className="no-code-icon">💻</div>
                      <h3>No code yet</h3>
                      {selectedApp.developer_id === user?.id ? (
                        <>
                          <p>Switch to the Chat tab to start building your app with AI</p>
                          <button className="generate-button-large" onClick={() => setShowChat(true)}>
                            💬 Open Chat
                          </button>
                        </>
                      ) : (
                        <p>This app hasn't been built yet.</p>
                      )}
                    </div>
                  )
                )}
              </div>
            </>
          ) : (
            <div className="no-app-selected">
              <div className="no-app-icon">🤖</div>
              <h3>Select an app to view or create a new one</h3>
              <button className="create-app-button-large" onClick={() => setShowCreateModal(true)}>
                ➕ Create New App
              </button>
            </div>
          )}
        </div>
      </div>

      {showCreateModal && (
        <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) setShowCreateModal(false); }}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Create New AI App</h3>
            <form onSubmit={handleCreateApp}>
              <input
                type="text"
                value={newAppName}
                onChange={(e) => setNewAppName(e.target.value)}
                placeholder="App Name"
                className="modal-input"
                autoFocus
                required
              />
              <textarea
                value={newAppDescription}
                onChange={(e) => setNewAppDescription(e.target.value)}
                placeholder="Describe what your app should do (e.g., 'A todo list app with dark mode')"
                className="modal-textarea"
                rows="4"
              />
              <div className="modal-actions">
                <button type="submit" className="modal-submit">Create</button>
                <button type="button" onClick={() => setShowCreateModal(false)} className="modal-cancel">
                  Cancel
                </button>
              </div>
            </form>
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

      {showDeleteConfirm && (
        <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) setShowDeleteConfirm(null); }}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Delete App?</h3>
            <p>Are you sure you want to delete this app? This action cannot be undone.</p>
            <div className="modal-actions">
              <button
                className="modal-submit delete-confirm-button"
                onClick={confirmDeleteApp}
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
    </div>
  );
}

export default AIAppBuilder;

