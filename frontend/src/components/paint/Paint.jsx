import React, { useState, useEffect } from 'react';
import CanvasDraw from 'react-canvas-draw';
import api from '../../services/api';

function Paint() {
  const [paintings, setPaintings] = useState([]);
  const [title, setTitle] = useState('');
  const [canvas, setCanvas] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  
  // Brush settings
  const [brushColor, setBrushColor] = useState('#000000');
  const [brushSize, setBrushSize] = useState(2);
  const [brushShape, setBrushShape] = useState('round'); // round or square
  const [canvasKey, setCanvasKey] = useState(0); // Used only when explicitly forcing reset
  const [selectedPainting, setSelectedPainting] = useState(null); // For viewing full size

  const normalizePainting = (raw = {}) => {
    const name = raw.name || raw.title || 'Untitled Painting';
    let saveData = '';
    
    if (typeof raw.data === 'string') {
      // If data is already a string, use it directly
      saveData = raw.data;
    } else if (raw.data) {
      // If data is an object/array, stringify it
      try {
        saveData = JSON.stringify(raw.data);
      } catch (err) {
        console.error('Error stringifying painting data:', err);
        saveData = JSON.stringify([]);
      }
    } else if (typeof raw.image_data === 'string') {
      // Legacy format - image_data should be JSON string for CanvasDraw
      saveData = raw.image_data;
    } else {
      // Default empty painting
      saveData = JSON.stringify([]);
    }
    
    // Validate saveData is valid JSON array
    if (saveData && typeof saveData === 'string') {
      try {
        const parsed = JSON.parse(saveData);
        if (!Array.isArray(parsed)) {
          console.warn('Painting data is not an array, resetting to empty');
          saveData = JSON.stringify([]);
        }
      } catch (err) {
        console.warn('Invalid JSON in painting data, resetting to empty:', err);
        saveData = JSON.stringify([]);
      }
    }
    
    return {
      ...raw,
      name,
      saveData,
    };
  };

  useEffect(() => {
    fetchPaintings();
  }, []);

  // Update brush shape by patching the canvas component's internal contexts (no DOM getContext calls)
  useEffect(() => {
    if (canvas && canvas.ctx) {
      const lineCapValue = brushShape === 'square' ? 'butt' : 'round';
      const lineJoinValue = brushShape === 'square' ? 'miter' : 'round';
      if (canvas.ctx.temp) {
        canvas.ctx.temp.lineCap = lineCapValue;
        canvas.ctx.temp.lineJoin = lineJoinValue;
      }
      if (canvas.ctx.drawing) {
        canvas.ctx.drawing.lineCap = lineCapValue;
        canvas.ctx.drawing.lineJoin = lineJoinValue;
      }
    }
  }, [brushShape, canvas]);

  const fetchPaintings = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.get('/api/paint');
      const items = response.data?.paintings || [];
      setPaintings(items.map(normalizePainting));
    } catch (err) {
      console.error('Failed to fetch paintings:', err);
      setError('Failed to load paintings');
      setPaintings([]);
    } finally {
      setLoading(false);
    }
  };

  const savePainting = async () => {
    if (!canvas || !title.trim()) return;
    setSaving(true);
    setError(null);
    try {
      const imageData = canvas.getSaveData();
      let strokes;
      try {
        strokes = JSON.parse(imageData);
      } catch (err) {
        strokes = [];
      }
      const response = await api.post('/api/paint', {
        name: title.trim(),
        data: strokes
      });
      const created = response.data?.paint ? normalizePainting(response.data.paint) : null;
      if (created) {
        setPaintings([...paintings, created]);
      } else {
        await fetchPaintings();
      }
      setTitle('');
      canvas.clear();
    } catch (err) {
      console.error('Failed to save painting:', err);
      setError('Failed to save painting');
    } finally {
      setSaving(false);
    }
  };

  const deletePainting = async (paintingId) => {
    const ok = window.confirm('Delete this painting?');
    if (!ok) return;
    try {
      await api.delete(`/api/paint/${paintingId}`);
      setPaintings(paintings.filter(p => p.id !== paintingId));
      if (selectedPainting?.id === paintingId) {
        setSelectedPainting(null);
      }
    } catch (error) {
      console.error('Failed to delete painting:', error);
      setError('Failed to delete painting');
    }
  };

  const loadPaintingToCanvas = (painting) => {
    if (!canvas || !painting) {
      setError('Canvas not ready or painting data invalid');
      return;
    }
    
    if (!painting.saveData) {
      setError('Failed to read file data: No save data found');
      return;
    }

    try {
      // Validate that saveData is valid JSON (CanvasDraw format)
      if (typeof painting.saveData === 'string') {
        // Try to parse it to validate
        try {
          const parsed = JSON.parse(painting.saveData);
          if (!Array.isArray(parsed)) {
            throw new Error('Invalid painting data format');
          }
        } catch (parseErr) {
          setError('Failed to read file data: Invalid painting format. This might be a PNG image, not a painting file.');
          return;
        }
      }

      canvas.loadSaveData(painting.saveData, true);
      setTitle(painting.name);
      setSelectedPainting(null);
      setError(null);
      
      // Update lineCap after loading
      if (canvas.canvas) {
        const ctx = canvas.canvas.getContext('2d');
        if (ctx) {
          ctx.lineCap = brushShape === 'square' ? 'butt' : 'round';
          ctx.lineJoin = brushShape === 'square' ? 'miter' : 'round';
        }
      }
    } catch (err) {
      console.error('Error loading painting:', err);
      setError(`Failed to read file data: ${err.message || 'Unknown error'}`);
    }
  };

  const viewPaintingFullSize = (painting) => {
    setSelectedPainting(normalizePainting(painting));
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Paint</h1>
      {loading && <div style={styles.status}>Loading paintings...</div>}
      {error && <div style={styles.error}>{error}</div>}
      
      <div style={styles.drawingSection}>
        <div style={styles.canvasContainer}>
          <CanvasDraw
            key={canvasKey}
            ref={(canvasDraw) => {
              setCanvas(canvasDraw);
              // Patch internal contexts immediately when canvas is set
              if (canvasDraw && canvasDraw.ctx) {
                const lineCapValue = brushShape === 'square' ? 'butt' : 'round';
                const lineJoinValue = brushShape === 'square' ? 'miter' : 'round';
                if (canvasDraw.ctx.temp) {
                  canvasDraw.ctx.temp.lineCap = lineCapValue;
                  canvasDraw.ctx.temp.lineJoin = lineJoinValue;
                }
                if (canvasDraw.ctx.drawing) {
                  canvasDraw.ctx.drawing.lineCap = lineCapValue;
                  canvasDraw.ctx.drawing.lineJoin = lineJoinValue;
                }
              }
            }}
            brushRadius={brushSize}
            lazyRadius={0}
            brushColor={brushColor}
            canvasWidth={600}
            canvasHeight={400}
            style={styles.canvas}
            onChange={() => {
              // Re-patch contexts on each change to ensure brush shape is applied
              if (canvas && canvas.ctx) {
                const lineCapValue = brushShape === 'square' ? 'butt' : 'round';
                const lineJoinValue = brushShape === 'square' ? 'miter' : 'round';
                if (canvas.ctx.temp) {
                  canvas.ctx.temp.lineCap = lineCapValue;
                  canvas.ctx.temp.lineJoin = lineJoinValue;
                }
                if (canvas.ctx.drawing) {
                  canvas.ctx.drawing.lineCap = lineCapValue;
                  canvas.ctx.drawing.lineJoin = lineJoinValue;
                }
              }
            }}
          />
        </div>
        
        <div style={styles.brushControls}>
          <div style={styles.controlGroup}>
            <label style={styles.label}>Color:</label>
            <div style={styles.colorPickerContainer}>
              <input
                type="color"
                value={brushColor}
                onChange={(e) => setBrushColor(e.target.value)}
                style={styles.colorInput}
              />
              <div style={styles.presetColors}>
                {['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080', '#FFC0CB'].map(color => (
                  <button
                    key={color}
                    type="button"
                    onClick={() => setBrushColor(color)}
                    style={{
                      ...styles.colorButton,
                      backgroundColor: color,
                      border: brushColor === color ? '3px solid #2196f3' : '1px solid #ddd'
                    }}
                  />
                ))}
              </div>
            </div>
          </div>
          
          <div style={styles.controlGroup}>
            <label style={styles.label}>Brush Size: {brushSize}px</label>
            <input
              type="range"
              min="1"
              max="50"
              value={brushSize}
              onChange={(e) => {
                setBrushSize(parseInt(e.target.value));
              }}
              style={styles.slider}
            />
            <div style={styles.sizePresets}>
              {[2, 5, 10, 20, 30].map(size => (
                <button
                  key={size}
                  type="button"
                  onClick={() => {
                    setBrushSize(size);
                  }}
                  style={{
                    ...styles.sizeButton,
                    backgroundColor: brushSize === size ? '#2196f3' : '#f5f5f5',
                    color: brushSize === size ? 'white' : '#333'
                  }}
                >
                  {size}px
                </button>
              ))}
            </div>
          </div>
          
          <div style={styles.controlGroup}>
            <label style={styles.label}>Brush Shape:</label>
            <div style={styles.shapeButtons}>
              <button
                type="button"
                onClick={() => setBrushShape('round')}
                style={{
                  ...styles.shapeButton,
                  backgroundColor: brushShape === 'round' ? '#2196f3' : '#f5f5f5',
                  color: brushShape === 'round' ? 'white' : '#333'
                }}
              >
                ● Round
              </button>
              <button
                type="button"
                onClick={() => setBrushShape('square')}
                style={{
                  ...styles.shapeButton,
                  backgroundColor: brushShape === 'square' ? '#2196f3' : '#f5f5f5',
                  color: brushShape === 'square' ? 'white' : '#333'
                }}
              >
                ■ Square
              </button>
            </div>
          </div>
        </div>
        
        <div style={styles.controls}>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Enter painting title..."
            style={styles.input}
            disabled={saving}
          />
          <button onClick={savePainting} style={styles.saveButton} disabled={saving}>
            {saving ? 'Saving...' : 'Save Painting'}
          </button>
          <button onClick={() => {
            canvas?.clear();
          }} style={styles.clearButton}>
            Clear Canvas
          </button>
        </div>
      </div>

      <div style={styles.gallery}>
        <h2 style={styles.subtitle}>My Paintings</h2>
        <div style={styles.paintingGrid}>
          {paintings.map((painting) => (
            <div key={painting.id} style={styles.paintingCard} className="painting-card">
              <div 
                style={styles.paintingThumbnail}
                className="painting-thumbnail"
                onClick={() => viewPaintingFullSize(painting)}
              >
                <CanvasDraw
                  disabled
                  hideGrid
                  loadTimeOffset={0}
                  saveData={painting.saveData}
                  canvasWidth={200}
                  canvasHeight={150}
                />
              </div>
              <div style={styles.paintingInfo}>
                <span style={styles.paintingTitle}>{painting.name}</span>
                <div style={styles.paintingActions}>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      loadPaintingToCanvas(painting);
                    }}
                    style={styles.loadButton}
                    title="Load to canvas"
                  >
                    Edit
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deletePainting(painting.id);
                    }}
                    style={styles.deleteButton}
                    title="Delete painting"
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {selectedPainting && (
        <div style={styles.modal} onClick={() => setSelectedPainting(null)}>
          <div style={styles.modalContent} onClick={(e) => e.stopPropagation()}>
            <div style={styles.modalHeader}>
              <h3 style={styles.modalTitle}>{selectedPainting.name}</h3>
              <button
                onClick={() => setSelectedPainting(null)}
                style={styles.closeButton}
              >
                ×
              </button>
            </div>
            <div style={styles.modalCanvas}>
              <CanvasDraw
                disabled
                hideGrid
                loadTimeOffset={0}
                saveData={selectedPainting.saveData}
                canvasWidth={800}
                canvasHeight={600}
              />
            </div>
            <div style={styles.modalActions}>
              <button
                onClick={() => {
                  loadPaintingToCanvas(selectedPainting);
                  setSelectedPainting(null);
                }}
                style={styles.modalLoadButton}
              >
                Load to Canvas
              </button>
              <button
                onClick={() => {
                  deletePainting(selectedPainting.id);
                  setSelectedPainting(null);
                }}
                style={styles.modalDeleteButton}
              >
                Delete
              </button>
            </div>
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
  title: {
    fontSize: '2.5rem',
    fontWeight: '700',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
    textAlign: 'center',
    marginBottom: '2rem',
  },
  drawingSection: {
    marginBottom: '3rem',
  },
  canvasContainer: {
    display: 'flex',
    justifyContent: 'center',
    marginBottom: '1rem',
  },
  canvas: {
    border: '2px solid rgba(102, 126, 234, 0.4)',
    borderRadius: '15px',
    backgroundColor: 'white',
    boxShadow: '0 10px 30px rgba(0, 0, 0, 0.2)',
  },
  brushControls: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    padding: '2rem',
    borderRadius: '20px',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    marginBottom: '1rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
    border: '1px solid rgba(102, 126, 234, 0.3)',
  },
  controlGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
  },
  label: {
    fontWeight: '500',
    color: '#333',
    fontSize: '0.9rem',
  },
  colorPickerContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
  },
  colorInput: {
    width: '60px',
    height: '40px',
    border: '2px solid #ddd',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  presetColors: {
    display: 'flex',
    gap: '0.5rem',
    flexWrap: 'wrap',
  },
  colorButton: {
    width: '30px',
    height: '30px',
    border: '1px solid #ddd',
    borderRadius: '4px',
    cursor: 'pointer',
    padding: 0,
  },
  slider: {
    width: '100%',
    height: '6px',
    borderRadius: '3px',
    outline: 'none',
    cursor: 'pointer',
  },
  sizePresets: {
    display: 'flex',
    gap: '0.5rem',
    flexWrap: 'wrap',
  },
  sizeButton: {
    padding: '0.5rem 1rem',
    border: '1px solid #ddd',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '0.85rem',
    transition: 'all 0.2s',
  },
  shapeButtons: {
    display: 'flex',
    gap: '0.5rem',
  },
  shapeButton: {
    padding: '0.5rem 1rem',
    border: '1px solid #ddd',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '0.9rem',
    transition: 'all 0.2s',
  },
  controls: {
    display: 'flex',
    gap: '1rem',
    justifyContent: 'center',
    flexWrap: 'wrap',
  },
  input: {
    padding: '0.75rem',
    fontSize: '1rem',
    border: '1px solid #ddd',
    borderRadius: '4px',
    width: '300px',
  },
  saveButton: {
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
  clearButton: {
    padding: '0.875rem 1.75rem',
    background: 'linear-gradient(135deg, #ff9800 0%, #f57c00 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '12px',
    cursor: 'pointer',
    fontSize: '1rem',
    fontWeight: '600',
    boxShadow: '0 4px 15px rgba(255, 152, 0, 0.4)',
    transition: 'all 0.3s ease',
  },
  gallery: {
    marginTop: '2rem',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    padding: '2rem',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    border: '1px solid rgba(102, 126, 234, 0.3)',
  },
  subtitle: {
    color: '#333',
    marginBottom: '1.5rem',
    fontSize: '1.5rem',
    fontWeight: '600',
  },
  paintingGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
    gap: '1rem',
  },
  paintingCard: {
    backgroundColor: 'white',
    borderRadius: '15px',
    boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)',
    overflow: 'hidden',
    transition: 'all 0.3s ease',
    border: '1px solid rgba(102, 126, 234, 0.4)',
  },
  paintingThumbnail: {
    cursor: 'pointer',
    transition: 'opacity 0.2s',
  },
  paintingInfo: {
    padding: '0.75rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
  },
  paintingTitle: {
    fontWeight: '500',
    color: '#333',
  },
  paintingActions: {
    display: 'flex',
    gap: '0.5rem',
  },
  loadButton: {
    padding: '0.5rem 1rem',
    backgroundColor: '#2196f3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '0.85rem',
    transition: 'background-color 0.2s',
  },
  deleteButton: {
    padding: '0.5rem 1rem',
    backgroundColor: '#ff5252',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '0.85rem',
    transition: 'background-color 0.2s',
  },
  modal: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    backdropFilter: 'blur(5px)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 2000,
    padding: '2rem',
  },
  modalContent: {
    backgroundColor: 'rgba(255, 255, 255, 0.98)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    maxWidth: '90vw',
    maxHeight: '90vh',
    overflow: 'auto',
    display: 'flex',
    flexDirection: 'column',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)',
    border: '1px solid rgba(102, 126, 234, 0.3)',
  },
  modalHeader: {
    padding: '1rem',
    borderBottom: '1px solid #eee',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  modalTitle: {
    margin: 0,
    color: '#333',
  },
  closeButton: {
    background: 'none',
    border: 'none',
    fontSize: '2rem',
    color: '#666',
    cursor: 'pointer',
    padding: '0 0.5rem',
    lineHeight: 1,
    '&:hover': {
      color: '#333',
    },
  },
  modalCanvas: {
    padding: '1rem',
    display: 'flex',
    justifyContent: 'center',
    backgroundColor: '#f5f5f5',
  },
  modalActions: {
    padding: '1rem',
    borderTop: '1px solid #eee',
    display: 'flex',
    gap: '1rem',
    justifyContent: 'center',
  },
  modalLoadButton: {
    padding: '0.75rem 1.5rem',
    backgroundColor: '#2196f3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '1rem',
    fontWeight: '500',
  },
  modalDeleteButton: {
    padding: '0.75rem 1.5rem',
    backgroundColor: '#ff5252',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '1rem',
    fontWeight: '500',
  },
};

export default Paint;