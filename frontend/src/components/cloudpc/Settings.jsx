import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import './Settings.css';

const THEMES = {
  light: {
    name: 'Light',
    background: '#ffffff',
    text: '#1a1a1a',
    surface: '#f5f5f5',
    border: '#e0e0e0',
    accent: '#667eea'
  },
  dark: {
    name: 'Dark',
    background: '#1a1a1a',
    text: '#ffffff',
    surface: '#2d2d2d',
    border: '#404040',
    accent: '#8b5cf6'
  },
  auto: {
    name: 'Auto (Time-based)',
    background: null, // Will be determined by time
    text: null,
    surface: null,
    border: null,
    accent: '#667eea'
  }
};

const FONTS = [
  { name: 'System Default', value: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' },
  { name: 'Arial', value: 'Arial, sans-serif' },
  { name: 'Helvetica', value: 'Helvetica, sans-serif' },
  { name: 'Times New Roman', value: '"Times New Roman", serif' },
  { name: 'Courier New', value: '"Courier New", monospace' },
  { name: 'Georgia', value: 'Georgia, serif' },
  { name: 'Verdana', value: 'Verdana, sans-serif' },
  { name: 'Comic Sans MS', value: '"Comic Sans MS", cursive' },
  { name: 'Trebuchet MS', value: '"Trebuchet MS", sans-serif' },
  { name: 'Impact', value: 'Impact, sans-serif' }
];

function Settings({ pcId }) {
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('cloudpc_theme') || 'auto';
    return saved;
  });
  
  const [font, setFont] = useState(() => {
    const saved = localStorage.getItem('cloudpc_font') || FONTS[0].value;
    return saved;
  });
  
  const [fontSize, setFontSize] = useState(() => {
    const saved = localStorage.getItem('cloudpc_fontSize') || '16';
    return saved;
  });
  
  const [animations, setAnimations] = useState(() => {
    const saved = localStorage.getItem('cloudpc_animations');
    return saved !== null ? saved === 'true' : true;
  });
  
  const [taskbarPosition, setTaskbarPosition] = useState(() => {
    const saved = localStorage.getItem('cloudpc_taskbarPosition') || 'bottom';
    return saved;
  });

  const [wallpaper, setWallpaper] = useState(() => {
    return localStorage.getItem('cloudpc_wallpaper') || '';
  });

  const [showWallpaperBrowser, setShowWallpaperBrowser] = useState(false);
  const [wallpaperFiles, setWallpaperFiles] = useState([]);
  const [wallpaperPath, setWallpaperPath] = useState('/');
  const [loadingWallpapers, setLoadingWallpapers] = useState(false);
  const [showResetConfirm, setShowResetConfirm] = useState(false);

  useEffect(() => {
    // Apply theme
    applyTheme(theme);
  }, [theme]);

  useEffect(() => {
    // Apply font
    document.documentElement.style.setProperty('--cloudpc-font', font);
  }, [font]);

  useEffect(() => {
    // Apply font size
    document.documentElement.style.setProperty('--cloudpc-font-size', `${fontSize}px`);
  }, [fontSize]);

  useEffect(() => {
    // Apply animations
    const root = document.documentElement;
    if (animations) {
      root.style.setProperty('--cloudpc-animations', '1');
    } else {
      root.style.setProperty('--cloudpc-animations', '0');
    }
  }, [animations]);

  useEffect(() => {
    // Apply taskbar position
    const viewer = document.querySelector('.cloud-pc-viewer');
    if (viewer) {
      viewer.setAttribute('data-taskbar-position', taskbarPosition);
    }
  }, [taskbarPosition]);

  const applyTheme = (themeName) => {
    const root = document.documentElement;
    
    if (themeName === 'auto') {
      // Determine theme based on time of day
      const hour = new Date().getHours();
      const isDark = hour < 6 || hour >= 20; // Dark from 8 PM to 6 AM
      const activeTheme = isDark ? THEMES.dark : THEMES.light;
      
      root.style.setProperty('--cloudpc-bg', activeTheme.background);
      root.style.setProperty('--cloudpc-text', activeTheme.text);
      root.style.setProperty('--cloudpc-surface', activeTheme.surface);
      root.style.setProperty('--cloudpc-border', activeTheme.border);
      root.style.setProperty('--cloudpc-accent', activeTheme.accent);
      // Taskbar colors: match background, opposite text
      root.style.setProperty('--cloudpc-taskbar-bg', activeTheme.background);
      root.style.setProperty('--cloudpc-taskbar-text', activeTheme.text);
    } else {
      const selectedTheme = THEMES[themeName];
      root.style.setProperty('--cloudpc-bg', selectedTheme.background);
      root.style.setProperty('--cloudpc-text', selectedTheme.text);
      root.style.setProperty('--cloudpc-surface', selectedTheme.surface);
      root.style.setProperty('--cloudpc-border', selectedTheme.border);
      root.style.setProperty('--cloudpc-accent', selectedTheme.accent);
      // Taskbar colors: match background, opposite text
      root.style.setProperty('--cloudpc-taskbar-bg', selectedTheme.background);
      root.style.setProperty('--cloudpc-taskbar-text', selectedTheme.text);
    }
    
    localStorage.setItem('cloudpc_theme', themeName);
  };

  const handleThemeChange = (newTheme) => {
    setTheme(newTheme);
    // Dispatch event to notify CloudPCViewer
    window.dispatchEvent(new CustomEvent('cloudpc-settings-changed'));
  };

  const handleFontChange = (newFont) => {
    setFont(newFont);
    localStorage.setItem('cloudpc_font', newFont);
    window.dispatchEvent(new CustomEvent('cloudpc-settings-changed'));
  };

  const handleFontSizeChange = (newSize) => {
    setFontSize(newSize);
    localStorage.setItem('cloudpc_fontSize', newSize);
    window.dispatchEvent(new CustomEvent('cloudpc-settings-changed'));
  };

  const handleAnimationsChange = (enabled) => {
    setAnimations(enabled);
    localStorage.setItem('cloudpc_animations', enabled.toString());
    window.dispatchEvent(new CustomEvent('cloudpc-settings-changed'));
  };

  const handleTaskbarPositionChange = (position) => {
    setTaskbarPosition(position);
    localStorage.setItem('cloudpc_taskbarPosition', position);
    window.dispatchEvent(new CustomEvent('cloudpc-settings-changed'));
  };

  const loadWallpaperFiles = async (path = '/') => {
    if (!pcId) return;
    try {
      setLoadingWallpapers(true);
      const { data } = await api.get(`/api/cloud-pcs/${pcId}/files`, {
        params: { path }
      });
      // Filter for PNG images only
      const pngFiles = (data.files || []).filter(file => 
        file.type === 'file' && file.name.toLowerCase().match(/\.(png|jpg|jpeg|gif|webp|bmp)$/)
      );
      // Include directories for navigation
      const dirs = (data.files || []).filter(file => file.type === 'directory');
      setWallpaperFiles([...dirs, ...pngFiles]);
    } catch (err) {
      console.error('Failed to load wallpaper files:', err);
    } finally {
      setLoadingWallpapers(false);
    }
  };

  const handleWallpaperSelect = async (file) => {
    if (file.type === 'directory') {
      const newPath = wallpaperPath === '/' ? `/${file.name}` : `${wallpaperPath}/${file.name}`;
      setWallpaperPath(newPath);
      await loadWallpaperFiles(newPath);
    } else {
      // Read the image file
      const filePath = wallpaperPath === '/' ? `/${file.name}` : `${wallpaperPath}/${file.name}`;
      try {
        const { data } = await api.get(`/api/cloud-pcs/${pcId}/files/read`, {
          params: { path: filePath }
        });
        
        if (data.is_binary && data.mime_type && data.mime_type.startsWith('image/')) {
          const imageUrl = `data:${data.mime_type};base64,${data.content}`;
          setWallpaper(imageUrl);
          localStorage.setItem('cloudpc_wallpaper', imageUrl);
          // Apply wallpaper immediately
          const viewer = document.querySelector('.cloud-pc-viewer');
          if (viewer) {
            viewer.style.setProperty('--cloudpc-wallpaper', `url(${imageUrl})`);
            // Also update desktop background
            const desktop = viewer.querySelector('.cloud-pc-desktop');
            if (desktop) {
              desktop.style.backgroundImage = `url(${imageUrl})`;
            }
          }
          window.dispatchEvent(new CustomEvent('cloudpc-settings-changed'));
          setShowWallpaperBrowser(false);
        }
      } catch (err) {
        console.error('Failed to load wallpaper:', err);
      }
    }
  };

  const handleWallpaperUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const imageUrl = event.target.result;
        setWallpaper(imageUrl);
        localStorage.setItem('cloudpc_wallpaper', imageUrl);
        // Apply wallpaper immediately
        const viewer = document.querySelector('.cloud-pc-viewer');
        if (viewer) {
          viewer.style.setProperty('--cloudpc-wallpaper', `url(${imageUrl})`);
          // Also update desktop background
          const desktop = viewer.querySelector('.cloud-pc-desktop');
          if (desktop) {
            desktop.style.backgroundImage = `url(${imageUrl})`;
          }
        }
        window.dispatchEvent(new CustomEvent('cloudpc-settings-changed'));
      };
      reader.readAsDataURL(file);
    }
    e.target.value = ''; // Reset input
  };

  const removeWallpaper = () => {
    setWallpaper('');
    localStorage.removeItem('cloudpc_wallpaper');
    const viewer = document.querySelector('.cloud-pc-viewer');
    if (viewer) {
      const defaultWallpaper = 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)';
      viewer.style.setProperty('--cloudpc-wallpaper', defaultWallpaper);
      // Also update desktop background
      const desktop = viewer.querySelector('.cloud-pc-desktop');
      if (desktop) {
        desktop.style.backgroundImage = defaultWallpaper;
      }
    }
    window.dispatchEvent(new CustomEvent('cloudpc-settings-changed'));
  };

  useEffect(() => {
    if (showWallpaperBrowser && pcId) {
      loadWallpaperFiles(wallpaperPath);
    }
  }, [showWallpaperBrowser, wallpaperPath, pcId]);

  const resetSettings = () => {
    setShowResetConfirm(true);
  };

  const confirmResetSettings = () => {
    setShowResetConfirm(false);
    setTheme('auto');
    setFont(FONTS[0].value);
    setFontSize('16');
    setAnimations(true);
    setTaskbarPosition('bottom');
    setWallpaper('');
    localStorage.removeItem('cloudpc_theme');
    localStorage.removeItem('cloudpc_font');
    localStorage.removeItem('cloudpc_fontSize');
    localStorage.removeItem('cloudpc_animations');
    localStorage.removeItem('cloudpc_taskbarPosition');
    localStorage.removeItem('cloudpc_wallpaper');
    const viewer = document.querySelector('.cloud-pc-viewer');
    if (viewer) {
      viewer.style.setProperty('--cloudpc-wallpaper', 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)');
    }
    window.dispatchEvent(new CustomEvent('cloudpc-settings-changed'));
  };

  return (
    <div className="settings-app">
      <div className="settings-header">
        <h1>⚙️ Settings</h1>
        <p>Customize your Cloud PC experience</p>
      </div>

      <div className="settings-content">
        <div className="settings-section">
          <h2>🎨 Appearance</h2>
          
          <div className="setting-item">
            <label>Theme</label>
            <div className="theme-options">
              {Object.entries(THEMES).map(([key, themeOption]) => (
                <button
                  key={key}
                  className={`theme-option ${theme === key ? 'active' : ''}`}
                  onClick={() => handleThemeChange(key)}
                >
                  <div className="theme-preview" data-theme={key}>
                    <div className="theme-preview-bg"></div>
                    <div className="theme-preview-surface"></div>
                  </div>
                  <span>{themeOption.name}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="setting-item">
            <label>Font Family</label>
            <select
              value={font}
              onChange={(e) => handleFontChange(e.target.value)}
              className="setting-select"
            >
              {FONTS.map((fontOption) => (
                <option key={fontOption.value} value={fontOption.value}>
                  {fontOption.name}
                </option>
              ))}
            </select>
            <div className="font-preview" style={{ fontFamily: font }}>
              The quick brown fox jumps over the lazy dog
            </div>
          </div>

          <div className="setting-item">
            <label>Font Size: {fontSize}px</label>
            <input
              type="range"
              min="12"
              max="24"
              value={fontSize}
              onChange={(e) => handleFontSizeChange(e.target.value)}
              className="setting-slider"
            />
            <div className="font-size-preview" style={{ fontSize: `${fontSize}px` }}>
              Preview text at {fontSize}px
            </div>
          </div>
        </div>

        <div className="settings-section">
          <h2>🖥️ Interface</h2>
          
          <div className="setting-item">
            <label>Taskbar Position</label>
            <div className="taskbar-position-options">
              {['bottom', 'top', 'left', 'right'].map((position) => (
                <button
                  key={position}
                  className={`position-option ${taskbarPosition === position ? 'active' : ''}`}
                  onClick={() => handleTaskbarPositionChange(position)}
                >
                  <span className={`position-icon position-${position}`}>
                    {position === 'bottom' && '▬'}
                    {position === 'top' && '▬'}
                    {position === 'left' && '▮'}
                    {position === 'right' && '▮'}
                  </span>
                  <span>{position.charAt(0).toUpperCase() + position.slice(1)}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="setting-item">
            <label>
              <input
                type="checkbox"
                checked={animations}
                onChange={(e) => handleAnimationsChange(e.target.checked)}
                className="setting-checkbox"
              />
              Enable Animations
            </label>
            <p className="setting-description">
              Toggle smooth transitions and animations throughout the interface
            </p>
          </div>
        </div>

        <div className="settings-section">
          <h2>🖼️ Wallpaper</h2>
          
          <div className="setting-item">
            <label>Desktop Background</label>
            {wallpaper && (
              <div className="wallpaper-preview">
                <img src={wallpaper} alt="Current wallpaper" />
                <button
                  onClick={removeWallpaper}
                  className="remove-wallpaper-button"
                  title="Remove wallpaper"
                >
                  ×
                </button>
              </div>
            )}
            <div className="wallpaper-actions">
              <label className="wallpaper-upload-button">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleWallpaperUpload}
                  style={{ display: 'none' }}
                />
                Upload from Computer
              </label>
              {pcId && (
                <button
                  onClick={() => {
                    setShowWallpaperBrowser(true);
                    loadWallpaperFiles('/');
                  }}
                  className="wallpaper-browse-button"
                >
                  Browse Cloud PC Files
                </button>
              )}
            </div>
            <p className="setting-description">
              Choose a custom wallpaper for your desktop background
            </p>
          </div>
        </div>

        <div className="settings-section">
          <h2>🔧 Advanced</h2>
          
          <div className="setting-item">
            <button
              onClick={resetSettings}
              className="reset-button"
            >
              Reset All Settings
            </button>
            <p className="setting-description">
              Restore all settings to their default values
            </p>
          </div>
        </div>

        {showWallpaperBrowser && (
          <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) setShowWallpaperBrowser(false); }}>
            <div className="modal wallpaper-browser-modal" onClick={(e) => e.stopPropagation()}>
              <h3>Select Wallpaper from Cloud PC</h3>
              <div className="wallpaper-browser-content">
                <div className="wallpaper-path">
                  <strong>Path:</strong> {wallpaperPath || '/'}
                </div>
                {loadingWallpapers ? (
                  <div className="wallpaper-loading">Loading images...</div>
                ) : wallpaperFiles.length === 0 ? (
                  <div className="wallpaper-empty">No images found in this folder</div>
                ) : (
                  <div className="wallpaper-grid">
                    {wallpaperFiles.map((file, index) => (
                      <div
                        key={index}
                        className={`wallpaper-item ${file.type === 'directory' ? 'directory' : 'image'}`}
                        onClick={() => handleWallpaperSelect(file)}
                        title={file.name}
                      >
                        {file.type === 'directory' ? (
                          <>
                            <span className="wallpaper-icon">📁</span>
                            <span className="wallpaper-name">{file.name}</span>
                          </>
                        ) : (
                          <>
                            <div className="wallpaper-thumbnail">
                              <img
                                src={`/api/cloud-pcs/${pcId}/files/read?path=${encodeURIComponent(wallpaperPath === '/' ? `/${file.name}` : `${wallpaperPath}/${file.name}`)}`}
                                alt={file.name}
                                onError={(e) => {
                                  e.target.style.display = 'none';
                                  e.target.nextSibling.style.display = 'flex';
                                }}
                              />
                              <div className="wallpaper-fallback" style={{ display: 'none' }}>
                                🖼️
                              </div>
                            </div>
                            <span className="wallpaper-name">{file.name}</span>
                          </>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <div className="modal-actions">
                <button
                  onClick={() => {
                    if (wallpaperPath !== '/') {
                      const parts = wallpaperPath.split('/').filter(p => p);
                      parts.pop();
                      setWallpaperPath(parts.length > 0 ? '/' + parts.join('/') : '/');
                    }
                  }}
                  className="modal-cancel"
                  disabled={wallpaperPath === '/'}
                >
                  ← Back
                </button>
                <button
                  onClick={() => setShowWallpaperBrowser(false)}
                  className="modal-cancel"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}

        {showResetConfirm && (
          <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) setShowResetConfirm(false); }}>
            <div className="modal" onClick={(e) => e.stopPropagation()}>
              <h3>Reset Settings?</h3>
              <p>Are you sure you want to reset all settings to default? This action cannot be undone.</p>
              <div className="modal-actions">
                <button
                  className="modal-submit delete-confirm-button"
                  onClick={confirmResetSettings}
                >
                  Reset
                </button>
                <button
                  className="modal-cancel"
                  onClick={() => setShowResetConfirm(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Settings;

