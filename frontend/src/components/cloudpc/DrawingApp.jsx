import React, { useState, useRef, useEffect } from 'react';
import api from '../../services/api';
import './DrawingApp.css';

function DrawingApp({ pcId }) {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [color, setColor] = useState('#000000');
  const [brushSize, setBrushSize] = useState(5);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [savePath, setSavePath] = useState('/');
  const [saveFileName, setSaveFileName] = useState('drawing.png');
  const [saving, setSaving] = useState(false);
  const [popup, setPopup] = useState(null);
  const fileNameInputRef = useRef(null);

  const handleFileNameChange = (e) => {
    const value = e.target.value;
    const cursorPosition = e.target.selectionStart;
    
    // If user is typing, automatically append .png if not already present
    if (value && !value.toLowerCase().endsWith('.png')) {
      const newValue = value + '.png';
      setSaveFileName(newValue);
      // Restore cursor position (adjust for added .png)
      setTimeout(() => {
        if (fileNameInputRef.current) {
          const newPosition = Math.min(cursorPosition, newValue.length - 4);
          fileNameInputRef.current.setSelectionRange(newPosition, newPosition);
        }
      }, 0);
    } else if (value) {
      setSaveFileName(value);
    } else {
      setSaveFileName('drawing.png');
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    // Fill with white background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  const startDrawing = (e) => {
    setIsDrawing(true);
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    ctx.lineTo(x, y);
    ctx.strokeStyle = color;
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };

  const handleSave = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    setSaving(true);
    try {
      const imageData = canvas.toDataURL('image/png');
      
      // Filename already has .png from handleFileNameChange, but ensure it's there
      let finalFileName = saveFileName.trim();
      if (!finalFileName.toLowerCase().endsWith('.png')) {
        finalFileName += '.png';
      }
      
      await api.post(`/api/cloud-pcs/${pcId}/files/drawing`, {
        path: savePath,
        filename: finalFileName,
        image_data: imageData
      });

      setShowSaveDialog(false);
      setSaveFileName('drawing.png');
      setSavePath('/');
      setPopup({ type: 'success', message: 'Drawing saved successfully!' });
      setTimeout(() => setPopup(null), 3000);
    } catch (err) {
      setPopup({ type: 'error', message: err.response?.data?.error || 'Failed to save drawing' });
      setTimeout(() => setPopup(null), 3000);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="drawing-app">
      <div className="drawing-toolbar">
        <div className="toolbar-group">
          <label>Color:</label>
          <input
            type="color"
            value={color}
            onChange={(e) => setColor(e.target.value)}
            className="color-picker"
          />
        </div>
        <div className="toolbar-group">
          <label>Brush Size: {brushSize}px</label>
          <input
            type="range"
            min="1"
            max="50"
            value={brushSize}
            onChange={(e) => setBrushSize(parseInt(e.target.value))}
            className="brush-slider"
          />
        </div>
        <button onClick={clearCanvas} className="clear-button">🗑️ Clear</button>
        <button onClick={() => setShowSaveDialog(true)} className="save-button">💾 Save Drawing</button>
      </div>

      <div className="drawing-canvas-container">
        <canvas
          ref={canvasRef}
          className="drawing-canvas"
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={(e) => {
            e.preventDefault();
            startDrawing(e.touches[0]);
          }}
          onTouchMove={(e) => {
            e.preventDefault();
            draw(e.touches[0]);
          }}
          onTouchEnd={stopDrawing}
        />
      </div>

      {showSaveDialog && (
        <div className="modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) setShowSaveDialog(false); }}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>Save Drawing</h3>
            <div className="modal-form">
              <div>
                <label>Path:</label>
                <input
                  type="text"
                  value={savePath}
                  onChange={(e) => setSavePath(e.target.value)}
                  placeholder="/"
                  className="modal-input"
                />
              </div>
              <div>
                <label>Filename:</label>
                <input
                  ref={fileNameInputRef}
                  type="text"
                  value={saveFileName}
                  onChange={handleFileNameChange}
                  placeholder="drawing.png"
                  className="modal-input"
                />
              </div>
              <div className="modal-actions">
                <button onClick={handleSave} disabled={saving} className="modal-submit">
                  {saving ? 'Saving...' : 'Save'}
                </button>
                <button onClick={() => setShowSaveDialog(false)} className="modal-cancel">
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {popup && (
        <div className={`popup popup-${popup.type}`}>
          <div className="popup-content">
            <span className="popup-icon">
              {popup.type === 'success' ? '✅' : '❌'}
            </span>
            <span className="popup-message">{popup.message}</span>
            <button className="popup-close" onClick={() => setPopup(null)}>×</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default DrawingApp;

