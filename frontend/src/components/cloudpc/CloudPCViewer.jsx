import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import api from '../../services/api';
import FileManager from './FileManager';
import DrawingApp from './DrawingApp';
import AIAppBuilder from './AIAppBuilder';
import Settings from './Settings';
import AppStore from './AppStore';
import './CloudPCViewer.css';

function CloudPCViewer() {
  const { pcId } = useParams();
  const { user } = useAuth();
  const navigate = useNavigate();
  const [cloudPC, setCloudPC] = useState(null);
  const [loading, setLoading] = useState(true);
  const [passwordVerified, setPasswordVerified] = useState(false);
  const [password, setPassword] = useState('');
  const [passwordError, setPasswordError] = useState('');
  const [showStartMenu, setShowStartMenu] = useState(false);
  const [openApps, setOpenApps] = useState(() => {
    // Load saved apps from localStorage
    if (pcId) {
      const saved = localStorage.getItem(`cloudpc_openApps_${pcId}`);
      if (saved) {
        try {
          return JSON.parse(saved);
        } catch (e) {
          return [];
        }
      }
    }
    return [];
  });
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [taskbarVisible, setTaskbarVisible] = useState(true);
  const [taskbarTimeout, setTaskbarTimeout] = useState(null);
  const viewerRef = useRef(null);
  const draggingWindowRef = useRef(null);
  const dragOffsetRef = useRef({ x: 0, y: 0 });
  const resizingWindowRef = useRef(null);
  const resizeStartRef = useRef({ x: 0, y: 0, width: 0, height: 0, edge: '' });

  useEffect(() => {
    fetchCloudPC();
    loadSettings();
    
    // Listen for settings changes from Settings component
    const handleSettingsChange = () => {
      loadSettings();
    };
    
    window.addEventListener('cloudpc-settings-changed', handleSettingsChange);
    
    return () => {
      window.removeEventListener('cloudpc-settings-changed', handleSettingsChange);
    };
  }, [pcId]);

  const loadSettings = () => {
    // Load theme
    const theme = localStorage.getItem('cloudpc_theme') || 'auto';
    applyTheme(theme);
    
    // Load font
    const font = localStorage.getItem('cloudpc_font') || '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    document.documentElement.style.setProperty('--cloudpc-font', font);
    
    // Load font size
    const fontSize = localStorage.getItem('cloudpc_fontSize') || '16';
    document.documentElement.style.setProperty('--cloudpc-font-size', `${fontSize}px`);
    
    // Load animations
    const animations = localStorage.getItem('cloudpc_animations');
    const animationsEnabled = animations !== null ? animations === 'true' : true;
    document.documentElement.style.setProperty('--cloudpc-animations', animationsEnabled ? '1' : '0');
    
    // Load taskbar position and wallpaper
    const taskbarPosition = localStorage.getItem('cloudpc_taskbarPosition') || 'bottom';
    const wallpaper = localStorage.getItem('cloudpc_wallpaper');
    
    // Use viewerRef if available, otherwise query selector
    const viewer = viewerRef.current || document.querySelector('.cloud-pc-viewer');
    if (viewer) {
      viewer.setAttribute('data-taskbar-position', taskbarPosition);
      
      // Apply wallpaper
      if (wallpaper) {
        viewer.style.setProperty('--cloudpc-wallpaper', `url(${wallpaper})`);
      } else {
        // Default modern gradient wallpaper
        viewer.style.setProperty('--cloudpc-wallpaper', 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)');
      }
    }
  };

  const applyTheme = (themeName) => {
    const root = document.documentElement;
    
    if (themeName === 'auto') {
      // Determine theme based on time of day
      const hour = new Date().getHours();
      const isDark = hour < 6 || hour >= 20; // Dark from 8 PM to 6 AM
      
      if (isDark) {
        root.style.setProperty('--cloudpc-bg', '#1a1a1a');
        root.style.setProperty('--cloudpc-text', '#ffffff');
        root.style.setProperty('--cloudpc-surface', '#2d2d2d');
        root.style.setProperty('--cloudpc-border', '#404040');
        root.style.setProperty('--cloudpc-accent', '#8b5cf6');
        // Taskbar: dark background, light text
        root.style.setProperty('--cloudpc-taskbar-bg', '#1a1a1a');
        root.style.setProperty('--cloudpc-taskbar-text', '#ffffff');
      } else {
        root.style.setProperty('--cloudpc-bg', '#ffffff');
        root.style.setProperty('--cloudpc-text', '#1a1a1a');
        root.style.setProperty('--cloudpc-surface', '#f5f5f5');
        root.style.setProperty('--cloudpc-border', '#e0e0e0');
        root.style.setProperty('--cloudpc-accent', '#667eea');
        // Taskbar: light background, dark text
        root.style.setProperty('--cloudpc-taskbar-bg', '#ffffff');
        root.style.setProperty('--cloudpc-taskbar-text', '#1a1a1a');
      }
    } else if (themeName === 'dark') {
      root.style.setProperty('--cloudpc-bg', '#1a1a1a');
      root.style.setProperty('--cloudpc-text', '#ffffff');
      root.style.setProperty('--cloudpc-surface', '#2d2d2d');
      root.style.setProperty('--cloudpc-border', '#404040');
      root.style.setProperty('--cloudpc-accent', '#8b5cf6');
      // Taskbar: dark background, light text
      root.style.setProperty('--cloudpc-taskbar-bg', '#1a1a1a');
      root.style.setProperty('--cloudpc-taskbar-text', '#ffffff');
    } else {
      root.style.setProperty('--cloudpc-bg', '#ffffff');
      root.style.setProperty('--cloudpc-text', '#1a1a1a');
      root.style.setProperty('--cloudpc-surface', '#f5f5f5');
      root.style.setProperty('--cloudpc-border', '#e0e0e0');
      root.style.setProperty('--cloudpc-accent', '#667eea');
      // Taskbar: light background, dark text
      root.style.setProperty('--cloudpc-taskbar-bg', '#ffffff');
      root.style.setProperty('--cloudpc-taskbar-text', '#1a1a1a');
    }
  };

  const fetchCloudPC = async () => {
    try {
      const { data } = await api.get(`/api/cloud-pcs/${pcId}`);
      setCloudPC(data.cloud_pc);
    } catch (err) {
      console.error('Failed to load cloud PC:', err);
    } finally {
      setLoading(false);
    }
  };

  const handlePasswordSubmit = async (e) => {
    e.preventDefault();
    setPasswordError('');

    try {
      await api.post(`/api/cloud-pcs/${pcId}/verify-password`, { password });
      setPasswordVerified(true);
      
      // Restore saved apps from backend (preferred) or localStorage (fallback)
      try {
        const { data } = await api.get(`/api/cloud-pcs/${pcId}/open-apps`);
        if (data.open_apps && data.open_apps.length > 0) {
          setOpenApps(data.open_apps);
        } else {
          // Fallback to localStorage
          const saved = localStorage.getItem(`cloudpc_openApps_${pcId}`);
          if (saved) {
            try {
              const restoredApps = JSON.parse(saved);
              setOpenApps(restoredApps);
            } catch (e) {
              console.error('Failed to restore apps from localStorage:', e);
            }
          }
        }
      } catch (err) {
        // If backend fails, try localStorage
        console.warn('Failed to load apps from backend, trying localStorage:', err);
        const saved = localStorage.getItem(`cloudpc_openApps_${pcId}`);
        if (saved) {
          try {
            const restoredApps = JSON.parse(saved);
            setOpenApps(restoredApps);
          } catch (e) {
            console.error('Failed to restore apps from localStorage:', e);
          }
        }
      }
      
      // Enter fullscreen after password verification
      setTimeout(() => {
        enterFullscreen();
      }, 100);
    } catch (err) {
      setPasswordError(err.response?.data?.error || 'Incorrect password');
    }
  };

  const enterFullscreen = () => {
    const element = viewerRef.current;
    if (!element) return;
    
    if (!document.fullscreenElement && 
        !document.webkitFullscreenElement && 
        !document.msFullscreenElement) {
      // Enter fullscreen
      if (element.requestFullscreen) {
        element.requestFullscreen();
      } else if (element.webkitRequestFullscreen) {
        element.webkitRequestFullscreen();
      } else if (element.mozRequestFullScreen) {
        element.mozRequestFullScreen();
      } else if (element.msRequestFullscreen) {
        element.msRequestFullscreen();
      }
    }
  };

  // Save apps to backend and localStorage whenever they change
  useEffect(() => {
    if (pcId && passwordVerified) {
      // Save to localStorage for immediate access
      localStorage.setItem(`cloudpc_openApps_${pcId}`, JSON.stringify(openApps));
      
      // Save to backend for cross-device sync (debounced)
      const saveTimeout = setTimeout(async () => {
        try {
          await api.put(`/api/cloud-pcs/${pcId}/open-apps`, {
            open_apps: openApps
          });
        } catch (err) {
          console.error('Failed to save open apps to backend:', err);
        }
      }, 500); // Debounce by 500ms
      
      return () => clearTimeout(saveTimeout);
    }
  }, [openApps, pcId, passwordVerified]);

  const openApp = (appNameOrAppData) => {
    let appName, appCode, appId, isDownloadedApp;
    
    // Handle both string (app name) and object (app data) formats
    if (typeof appNameOrAppData === 'string') {
      appName = appNameOrAppData;
      appCode = null;
      appId = null;
      isDownloadedApp = false;
    } else {
      appName = appNameOrAppData.name;
      appCode = appNameOrAppData.code;
      appId = appNameOrAppData.appId;
      isDownloadedApp = appNameOrAppData.isDownloadedApp || false;
    }
    
    // Check if app is already open (including minimized)
    const existingApp = openApps.find(app => app.name === appName && (!isDownloadedApp || app.appId === appId));
    if (existingApp) {
      // If minimized, restore it
      if (existingApp.minimized) {
        setOpenApps(openApps.map(app => 
          app.id === existingApp.id ? { ...app, minimized: false } : app
        ));
      }
      setShowStartMenu(false);
      return;
    }
    // Open new app with initial position
    const newApp = {
      name: appName,
      id: Date.now(),
      minimized: false,
      x: Math.random() * 100 + 50, // Random initial position
      y: Math.random() * 100 + 50,
      width: 800,
      height: 600,
      isMaximized: false,
      code: appCode || null,
      appId: appId || null,
      isDownloadedApp: isDownloadedApp
    };
    setOpenApps([...openApps, newApp]);
    setShowStartMenu(false);
    
    // Focus Minecraft iframe when opened
    if (appName === 'Minecraft') {
      setTimeout(() => {
        const iframe = document.querySelector(`iframe[title="Minecraft"]`);
        if (iframe) {
          iframe.focus();
          try {
            if (iframe.contentWindow) {
              iframe.contentWindow.focus();
            }
          } catch (e) {
            // Cross-origin restrictions might prevent this
          }
        }
      }, 300);
    }
  };

  const closeApp = (appId) => {
    setOpenApps(openApps.filter(app => app.id !== appId));
  };

  const minimizeApp = (appId) => {
    setOpenApps(openApps.map(app => 
      app.id === appId ? { ...app, minimized: true } : app
    ));
  };

  const restoreApp = (appId) => {
    setOpenApps(openApps.map(app => 
      app.id === appId ? { ...app, minimized: false } : app
    ));
    
    // Focus Minecraft iframe when restored
    const restoredApp = openApps.find(app => app.id === appId);
    if (restoredApp && restoredApp.name === 'Minecraft') {
      setTimeout(() => {
        const iframe = document.querySelector(`iframe[title="Minecraft"]`);
        if (iframe) {
          iframe.focus();
          try {
            if (iframe.contentWindow) {
              iframe.contentWindow.focus();
            }
          } catch (e) {
            // Cross-origin restrictions might prevent this
          }
        }
      }, 150);
    }
  };

  const handleWindowMouseDown = (e, appId) => {
    const app = openApps.find(a => a.id === appId);
    if (!app || app.isMaximized) return;
    
    // Check if clicking on resize handle
    const resizeHandle = e.target.closest('.resize-handle');
    if (resizeHandle) {
      const edge = resizeHandle.dataset.edge;
      const rect = e.currentTarget.getBoundingClientRect();
      resizingWindowRef.current = appId;
      resizeStartRef.current = {
        x: e.clientX,
        y: e.clientY,
        width: rect.width,
        height: rect.height,
        edge: edge
      };
      e.preventDefault();
      e.stopPropagation();
      return;
    }
    
    // Only start drag if clicking on header
    if (!e.target.closest('.app-window-header')) return;
    
    draggingWindowRef.current = appId;
    const rect = e.currentTarget.getBoundingClientRect();
    dragOffsetRef.current = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  };

  const handleWindowMouseMove = (e) => {
    // Handle window resizing
    if (resizingWindowRef.current) {
      const appId = resizingWindowRef.current;
      const start = resizeStartRef.current;
      const deltaX = e.clientX - start.x;
      const deltaY = e.clientY - start.y;
      const edge = start.edge;
      
      setOpenApps(openApps.map(app => {
        if (app.id === appId) {
          let newWidth = start.width;
          let newHeight = start.height;
          let newX = app.x || 50;
          let newY = app.y || 50;
          
          // Minimum window size
          const minWidth = 300;
          const minHeight = 200;
          
          // Handle different edges
          if (edge.includes('right')) {
            newWidth = Math.max(minWidth, start.width + deltaX);
          }
          if (edge.includes('left')) {
            newWidth = Math.max(minWidth, start.width - deltaX);
            newX = Math.max(0, (app.x || 50) + deltaX);
          }
          if (edge.includes('bottom')) {
            newHeight = Math.max(minHeight, start.height + deltaY);
          }
          if (edge.includes('top')) {
            newHeight = Math.max(minHeight, start.height - deltaY);
            newY = Math.max(0, (app.y || 50) + deltaY);
          }
          
          return {
            ...app,
            width: newWidth,
            height: newHeight,
            x: newX,
            y: newY
          };
        }
        return app;
      }));
      return;
    }
    
    // Handle window dragging
    if (!draggingWindowRef.current) return;
    
    const appId = draggingWindowRef.current;
    const newX = e.clientX - dragOffsetRef.current.x;
    const newY = e.clientY - dragOffsetRef.current.y;
    
    setOpenApps(openApps.map(app => 
      app.id === appId ? { ...app, x: Math.max(0, newX), y: Math.max(0, newY) } : app
    ));
  };

  const handleWindowMouseUp = () => {
    draggingWindowRef.current = null;
    resizingWindowRef.current = null;
  };

  const toggleWindowMaximize = (appId) => {
    setOpenApps(openApps.map(app => {
      if (app.id === appId) {
        return { ...app, isMaximized: !app.isMaximized };
      }
      return app;
    }));
  };

  useEffect(() => {
    document.addEventListener('mousemove', handleWindowMouseMove);
    document.addEventListener('mouseup', handleWindowMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleWindowMouseMove);
      document.removeEventListener('mouseup', handleWindowMouseUp);
    };
  }, [openApps]);

  useEffect(() => {
    const handleMouseMove = (e) => {
      const windowHeight = window.innerHeight;
      const mouseY = e.clientY;
      const threshold = 50; // pixels from bottom
      
      if (windowHeight - mouseY < threshold) {
        // Show taskbar
        setTaskbarVisible(true);
        if (taskbarTimeout) {
          clearTimeout(taskbarTimeout);
          setTaskbarTimeout(null);
        }
      } else {
        // Hide taskbar after delay
        if (taskbarTimeout) {
          clearTimeout(taskbarTimeout);
        }
        const timeout = setTimeout(() => {
          setTaskbarVisible(false);
        }, 1000); // Hide after 1 second
        setTaskbarTimeout(timeout);
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      if (taskbarTimeout) {
        clearTimeout(taskbarTimeout);
      }
    };
  }, [taskbarTimeout]);

  const getAppIcon = (appName) => {
    if (appName === 'File Manager') return '📁';
    if (appName === 'Drawing') return '🎨';
    if (appName === 'AI App Builder') return '🤖';
    if (appName === 'Settings') return '⚙️';
    if (appName === 'App Store') return '🛒';
    if (appName === 'Minecraft') return '⛏️';
    return '📄';
  };

  const toggleFullscreen = () => {
    const element = viewerRef.current;
    if (!element) return;
    
    if (!document.fullscreenElement && 
        !document.webkitFullscreenElement && 
        !document.msFullscreenElement) {
      // Enter fullscreen
      enterFullscreen();
    } else {
      // Exit fullscreen
      if (document.exitFullscreen) {
        document.exitFullscreen();
      } else if (document.webkitExitFullscreen) {
        document.webkitExitFullscreen();
      } else if (document.mozCancelFullScreen) {
        document.mozCancelFullScreen();
      } else if (document.msExitFullscreen) {
        document.msExitFullscreen();
      }
    }
  };

  useEffect(() => {
    const handleFullscreenChange = () => {
      const isFull = !!(
        document.fullscreenElement ||
        document.webkitFullscreenElement ||
        document.msFullscreenElement
      );
      setIsFullscreen(isFull);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('msfullscreenchange', handleFullscreenChange);

    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
      document.removeEventListener('msfullscreenchange', handleFullscreenChange);
    };
  }, []);

  if (loading) {
    return <div className="cloud-pc-viewer loading">Loading Cloud PC...</div>;
  }

  if (!cloudPC) {
    return <div className="cloud-pc-viewer error">Cloud PC not found</div>;
  }

  if (!passwordVerified) {
    return (
      <div className="cloud-pc-viewer password-screen">
        <div className="password-container">
          <div className="welcome-message">
            <h1>Welcome back, {user?.username}!</h1>
            <p>Enter your password to access {cloudPC.name}</p>
          </div>
          <form onSubmit={handlePasswordSubmit} className="password-form">
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              className="password-input"
              autoFocus
            />
            {passwordError && <div className="password-error">{passwordError}</div>}
            <button type="submit" className="password-submit">Login</button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="cloud-pc-viewer" ref={viewerRef}>
      <div className="cloud-pc-desktop">
        <div className={`cloud-pc-taskbar ${taskbarVisible ? 'visible' : 'hidden'}`}>
          <button 
            className="start-button"
            onClick={() => setShowStartMenu(!showStartMenu)}
          >
            <span className="start-icon">🚀</span>
            <span>Start</span>
          </button>
          <div className="taskbar-apps">
            {openApps.map((app) => (
              <div
                key={app.id}
                className={`taskbar-app ${app.minimized ? 'minimized-app' : ''}`}
                onClick={() => app.minimized ? restoreApp(app.id) : null}
                title={app.name}
              >
                <span className="taskbar-app-icon">{getAppIcon(app.name)}</span>
                <span className="taskbar-app-name">{app.name}</span>
              </div>
            ))}
          </div>

          <button 
            className="fullscreen-button"
            onClick={toggleFullscreen}
            title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}
          >
            {isFullscreen ? (
              <>
                <span className="fullscreen-icon">⤓</span>
                <span>Exit Fullscreen</span>
              </>
            ) : (
              <>
                <span className="fullscreen-icon">⤢</span>
                <span>Fullscreen</span>
              </>
            )}
          </button>

          {showStartMenu && (
            <div className="start-menu">
              <div className="start-menu-header">Applications</div>
              <div className="start-menu-items">
                <div className="start-menu-item" onClick={() => openApp('File Manager')}>
                  <span className="menu-icon">📁</span>
                  <span>File Manager</span>
                </div>
                <div className="start-menu-item" onClick={() => openApp('Drawing')}>
                  <span className="menu-icon">🎨</span>
                  <span>Drawing</span>
                </div>
                <div className="start-menu-item" onClick={() => openApp('AI App Builder')}>
                  <span className="menu-icon">🤖</span>
                  <span>AI App Builder</span>
                </div>
                <div className="start-menu-item" onClick={() => openApp('Settings')}>
                  <span className="menu-icon">⚙️</span>
                  <span>Settings</span>
                </div>
                <div className="start-menu-item" onClick={() => openApp('App Store')}>
                  <span className="menu-icon">🛒</span>
                  <span>App Store</span>
                </div>
                <div className="start-menu-item" onClick={() => openApp('Minecraft')}>
                  <span className="menu-icon">⛏️</span>
                  <span>Minecraft</span>
                </div>
                {(() => {
                  const pinnedApps = JSON.parse(localStorage.getItem('cloudpc_pinnedApps') || '[]');
                  return pinnedApps.map((pinnedApp) => (
                    <div key={pinnedApp.id} className="start-menu-item" onClick={() => openApp({
                      name: pinnedApp.name,
                      code: pinnedApp.code,
                      appId: pinnedApp.id,
                      isDownloadedApp: true
                    })}>
                      <span className="menu-icon">📱</span>
                      <span>{pinnedApp.name}</span>
                    </div>
                  ));
                })()}
              </div>
            </div>
          )}
        </div>

        {openApps.map((app) => (
            <div 
              key={app.id} 
              className={`app-window ${app.isMaximized ? 'maximized' : ''} ${app.minimized ? 'minimized' : ''}`}
              style={{
                left: app.isMaximized ? 0 : `${app.x || 50}px`,
                top: app.isMaximized ? 0 : `${app.y || 50}px`,
                width: app.isMaximized ? '100%' : `${app.width || 800}px`,
                height: app.isMaximized ? 'calc(100% - 48px)' : `${app.height || 600}px`,
                zIndex: openApps.indexOf(app) + 1000,
                display: app.minimized ? 'none' : 'flex'
              }}
              onMouseDown={(e) => {
                // Don't start dragging if clicking on Minecraft iframe
                if (app.name === 'Minecraft' && e.target.closest('.downloaded-app-container')) {
                  return;
                }
                handleWindowMouseDown(e, app.id);
              }}
            >
              {!app.isMaximized && (
                <>
                  {/* Corner resize handles */}
                  <div className="resize-handle resize-handle-nw" data-edge="top-left"></div>
                  <div className="resize-handle resize-handle-ne" data-edge="top-right"></div>
                  <div className="resize-handle resize-handle-sw" data-edge="bottom-left"></div>
                  <div className="resize-handle resize-handle-se" data-edge="bottom-right"></div>
                  {/* Edge resize handles */}
                  <div className="resize-handle resize-handle-n" data-edge="top"></div>
                  <div className="resize-handle resize-handle-s" data-edge="bottom"></div>
                  <div className="resize-handle resize-handle-w" data-edge="left"></div>
                  <div className="resize-handle resize-handle-e" data-edge="right"></div>
                </>
              )}
              <div className="app-window-header">
                <span className="app-title">{app.name}</span>
                <div className="app-window-controls">
                  <button 
                    className="app-minimize" 
                    onClick={() => minimizeApp(app.id)}
                    title="Minimize"
                  >
                    −
                  </button>
                  <button 
                    className="app-maximize" 
                    onClick={() => toggleWindowMaximize(app.id)}
                    title={app.isMaximized ? "Restore" : "Maximize"}
                  >
                    {app.isMaximized ? '❐' : '□'}
                  </button>
                  <button 
                    className="app-close" 
                    onClick={() => closeApp(app.id)}
                    title="Close"
                  >
                    ×
                  </button>
                </div>
              </div>
              <div className="app-window-content">
                {app.name === 'File Manager' && (
                  <FileManager pcId={pcId} />
                )}
              {app.name === 'Drawing' && (
                <DrawingApp pcId={pcId} />
              )}
              {app.name === 'AI App Builder' && (
                <AIAppBuilder pcId={pcId} />
              )}
              {app.name === 'Settings' && (
                <Settings pcId={pcId} />
              )}
              {app.name === 'App Store' && (
                <AppStore pcId={pcId} onOpenApp={openApp} />
              )}
              {app.name === 'Minecraft' && (
                <div 
                  className="downloaded-app-container"
                  onClick={(e) => {
                    // Focus the iframe when clicking on Minecraft window
                    e.stopPropagation();
                    const iframe = e.currentTarget.querySelector('iframe');
                    if (iframe) {
                      iframe.focus();
                      try {
                        if (iframe.contentWindow) {
                          iframe.contentWindow.focus();
                        }
                      } catch (err) {
                        // Cross-origin restrictions might prevent this
                      }
                    }
                  }}
                  onMouseDown={(e) => {
                    // Prevent window dragging when clicking on Minecraft content
                    if (e.target.closest('.downloaded-app-container')) {
                      e.stopPropagation();
                    }
                  }}
                >
                  <iframe
                    key={`minecraft-${app.id}`}
                    src="/minecraft.html"
                    title="Minecraft"
                    className="downloaded-app-iframe"
                    sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-modals allow-pointer-lock"
                    tabIndex={0}
                    onLoad={(e) => {
                      // Focus the iframe when it loads
                      const iframe = e.target;
                      if (iframe) {
                        iframe.focus();
                        try {
                          if (iframe.contentWindow) {
                            iframe.contentWindow.focus();
                          }
                        } catch (err) {
                          // Cross-origin restrictions might prevent this
                        }
                      }
                    }}
                    onMouseEnter={() => {
                      // Focus when mouse enters the iframe area
                      const iframe = document.querySelector(`iframe[title="Minecraft"]`);
                      if (iframe) {
                        iframe.focus();
                        try {
                          if (iframe.contentWindow) {
                            iframe.contentWindow.focus();
                          }
                        } catch (err) {
                          // Cross-origin restrictions might prevent this
                        }
                      }
                    }}
                  />
                </div>
              )}
              {app.isDownloadedApp && app.code && (
                <div className="downloaded-app-container">
                  <iframe
                    srcDoc={app.code}
                    title={app.name}
                    className="downloaded-app-iframe"
                    sandbox="allow-scripts allow-same-origin"
                  />
                </div>
              )}
            </div>
          </div>
          ))}
      </div>
    </div>
  );
}

export default CloudPCViewer;

