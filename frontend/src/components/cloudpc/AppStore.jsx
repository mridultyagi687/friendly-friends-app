import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import api from '../../services/api';
import './AppStore.css';

function AppStore({ pcId, onOpenApp }) {
  const { user } = useAuth();
  const [apps, setApps] = useState([]);
  const [myDownloads, setMyDownloads] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('store'); // 'store' or 'my-downloads'
  const [popup, setPopup] = useState(null);
  const [selectedApp, setSelectedApp] = useState(null);
  const [showPreview, setShowPreview] = useState(false);
  const [pinnedApps, setPinnedApps] = useState(() => {
    const saved = localStorage.getItem('cloudpc_pinnedApps');
    return saved ? JSON.parse(saved) : [];
  });

  useEffect(() => {
    loadApps();
    loadMyDownloads();
  }, []);

  const showPopupMessage = (type, message) => {
    setPopup({ type, message });
    setTimeout(() => setPopup(null), 3000);
  };

  const loadApps = async () => {
    try {
      setLoading(true);
      const { data } = await api.get('/api/app-store');
      setApps(data.apps || []);
    } catch (err) {
      showPopupMessage('error', err.response?.data?.error || 'Failed to load app store');
    } finally {
      setLoading(false);
    }
  };

  const loadMyDownloads = async () => {
    try {
      const { data } = await api.get('/api/app-store/my-downloads');
      setMyDownloads(data.apps || []);
    } catch (err) {
      // Silently fail - downloads might not exist yet
    }
  };

  const handleDownload = async (appId) => {
    try {
      await api.post(`/api/app-store/${appId}/download`);
      showPopupMessage('success', 'App downloaded successfully!');
      loadApps(); // Refresh to update download status
      loadMyDownloads(); // Refresh downloads list
    } catch (err) {
      showPopupMessage('error', err.response?.data?.error || 'Failed to download app');
    }
  };

  const handleRemoveDownload = async (appId) => {
    try {
      await api.delete(`/api/app-store/${appId}/download`);
      showPopupMessage('success', 'App removed from downloads');
      loadApps(); // Refresh to update download status
      loadMyDownloads(); // Refresh downloads list
    } catch (err) {
      showPopupMessage('error', err.response?.data?.error || 'Failed to remove download');
    }
  };

  const handlePreview = (app) => {
    setSelectedApp(app);
    setShowPreview(true);
  };

  const handleOpenApp = (app) => {
    // Open the app as a Cloud PC window
    if (onOpenApp) {
      onOpenApp({
        name: app.name,
        code: app.code,
        appId: app.id,
        isDownloadedApp: true
      });
      setShowPreview(false);
    }
  };

  const handlePinToStart = (app) => {
    const pinnedApp = {
      id: app.id,
      name: app.name,
      code: app.code,
      developer: app.developer
    };
    
    // Check if already pinned
    if (pinnedApps.find(p => p.id === app.id)) {
      showPopupMessage('info', 'App is already pinned to start menu');
      return;
    }
    
    const updatedPinned = [...pinnedApps, pinnedApp];
    setPinnedApps(updatedPinned);
    localStorage.setItem('cloudpc_pinnedApps', JSON.stringify(updatedPinned));
    showPopupMessage('success', 'App pinned to start menu!');
  };

  const handleUnpinFromStart = (appId) => {
    const updatedPinned = pinnedApps.filter(p => p.id !== appId);
    setPinnedApps(updatedPinned);
    localStorage.setItem('cloudpc_pinnedApps', JSON.stringify(updatedPinned));
    showPopupMessage('success', 'App unpinned from start menu');
  };

  const isPinned = (appId) => {
    return pinnedApps.some(p => p.id === appId);
  };

  if (loading) {
    return <div className="app-store loading">Loading App Store...</div>;
  }

  return (
    <div className="app-store">
      <div className="app-store-header">
        <h2>🛒 App Store</h2>
        <div className="app-store-tabs">
          <button
            className={`tab ${activeTab === 'store' ? 'active' : ''}`}
            onClick={() => setActiveTab('store')}
          >
            Store ({apps.length})
          </button>
          <button
            className={`tab ${activeTab === 'my-downloads' ? 'active' : ''}`}
            onClick={() => setActiveTab('my-downloads')}
          >
            My Downloads ({myDownloads.length})
          </button>
        </div>
      </div>

      <div className="app-store-content">
        {activeTab === 'store' ? (
          apps.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">📱</div>
              <h3>No apps available</h3>
              <p>Be the first to publish an app!</p>
            </div>
          ) : (
            <div className="apps-grid">
              {apps.map((app) => (
                <div key={app.id} className="app-card">
                  <div className="app-card-header">
                    <h3>{app.name}</h3>
                    {app.is_downloaded && <span className="downloaded-badge">✓ Downloaded</span>}
                  </div>
                  {app.description && (
                    <p className="app-description">{app.description}</p>
                  )}
                  <div className="app-meta">
                    <span className="app-developer">By {app.developer}</span>
                    {app.live_at && (
                      <span className="app-date">
                        {new Date(app.live_at).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                  <div className="app-actions">
                    <button
                      className="preview-button"
                      onClick={() => handlePreview(app)}
                    >
                      👁️ Preview
                    </button>
                    {app.is_downloaded ? (
                      <>
                        <button
                          className="open-button"
                          onClick={() => handleOpenApp(app)}
                        >
                          ▶️ Open
                        </button>
                        <button
                          className={isPinned(app.id) ? "unpin-button" : "pin-button"}
                          onClick={() => isPinned(app.id) ? handleUnpinFromStart(app.id) : handlePinToStart(app)}
                          title={isPinned(app.id) ? "Unpin from Start" : "Pin to Start"}
                        >
                          {isPinned(app.id) ? '📌 Pinned' : '📌 Pin'}
                        </button>
                        <button
                          className="remove-button"
                          onClick={() => handleRemoveDownload(app.id)}
                        >
                          🗑️ Remove
                        </button>
                      </>
                    ) : (
                      <button
                        className="download-button"
                        onClick={() => handleDownload(app.id)}
                      >
                        ⬇️ Download
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )
        ) : (
          myDownloads.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">📥</div>
              <h3>No downloads yet</h3>
              <p>Browse the store to download apps!</p>
            </div>
          ) : (
            <div className="apps-grid">
              {myDownloads.map((app) => (
                <div key={app.id} className="app-card">
                  <div className="app-card-header">
                    <h3>{app.name}</h3>
                    <span className="downloaded-badge">✓ Downloaded</span>
                  </div>
                  {app.description && (
                    <p className="app-description">{app.description}</p>
                  )}
                  <div className="app-meta">
                    <span className="app-developer">By {app.developer}</span>
                    {app.downloaded_at && (
                      <span className="app-date">
                        Downloaded {new Date(app.downloaded_at).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                  <div className="app-actions">
                    <button
                      className="preview-button"
                      onClick={() => handlePreview(app)}
                    >
                      👁️ Preview
                    </button>
                    <button
                      className="open-button"
                      onClick={() => handleOpenApp(app)}
                    >
                      ▶️ Open
                    </button>
                    <button
                      className={isPinned(app.id) ? "unpin-button" : "pin-button"}
                      onClick={() => isPinned(app.id) ? handleUnpinFromStart(app.id) : handlePinToStart(app)}
                      title={isPinned(app.id) ? "Unpin from Start" : "Pin to Start"}
                    >
                      {isPinned(app.id) ? '📌 Pinned' : '📌 Pin'}
                    </button>
                    <button
                      className="remove-button"
                      onClick={() => handleRemoveDownload(app.id)}
                    >
                      🗑️ Remove
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )
        )}
      </div>

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

      {showPreview && selectedApp && (
        <div className="modal-overlay" onClick={() => setShowPreview(false)}>
          <div className="modal preview-modal" onClick={(e) => e.stopPropagation()}>
            <div className="preview-header">
              <h3>{selectedApp.name}</h3>
              <button className="close-button" onClick={() => setShowPreview(false)}>×</button>
            </div>
            <div className="preview-content">
              <iframe
                srcDoc={selectedApp.code}
                title={selectedApp.name}
                className="app-preview-iframe"
                sandbox="allow-scripts allow-same-origin"
              />
            </div>
            <div className="preview-actions">
              {!selectedApp.is_downloaded && (
                <button
                  className="download-button"
                  onClick={() => {
                    handleDownload(selectedApp.id);
                    setShowPreview(false);
                  }}
                >
                  ⬇️ Download
                </button>
              )}
              {selectedApp.is_downloaded && (
                <>
                  <button
                    className="open-button"
                    onClick={() => {
                      handleOpenApp(selectedApp);
                    }}
                  >
                    ▶️ Open
                  </button>
                  <button
                    className={isPinned(selectedApp.id) ? "unpin-button" : "pin-button"}
                    onClick={() => {
                      if (isPinned(selectedApp.id)) {
                        handleUnpinFromStart(selectedApp.id);
                      } else {
                        handlePinToStart(selectedApp);
                      }
                    }}
                  >
                    {isPinned(selectedApp.id) ? '📌 Pinned' : '📌 Pin to Start'}
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default AppStore;

