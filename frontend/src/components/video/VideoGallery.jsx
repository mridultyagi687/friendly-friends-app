import React, { useState, useEffect, useRef, useCallback } from 'react';
import api from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';

function VideoGallery() {
  const { user, loading: authLoading } = useAuth();
  const [videos, setVideos] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [title, setTitle] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [thumbnails, setThumbnails] = useState({});
  const [playingVideo, setPlayingVideo] = useState(null);
  const [videoError, setVideoError] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const videoPlayerRef = useRef(null);

  const normalizeVideo = (raw = {}) => ({
    ...raw,
    title: raw.title || raw.filename || 'Untitled Video',
  });

  const getVideoUrl = useCallback((videoId) => {
    // Always use relative URL to go through Vite proxy or current origin
    // This ensures it works with both dev server (proxy) and production (same origin)
    return `/api/videos/${videoId}/stream`;
  }, []);

  useEffect(() => {
    if (authLoading) return;
    if (!user) {
      setVideos([]);
      setLoading(false);
      setError(null);
      return;
    }
    fetchVideos();
  }, [user, authLoading]);

  // Ensure video loads when playingVideo changes
  useEffect(() => {
    if (!playingVideo || !videoPlayerRef.current) return;

    const videoUrl = getVideoUrl(playingVideo.id);
    const videoEl = videoPlayerRef.current;
    
    // Set video source
    if (videoEl.src !== videoUrl) {
      videoEl.src = videoUrl;
      videoEl.load();
    }
  }, [playingVideo, getVideoUrl]);

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('mozfullscreenchange', handleFullscreenChange);
    document.addEventListener('MSFullscreenChange', handleFullscreenChange);

    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
      document.removeEventListener('mozfullscreenchange', handleFullscreenChange);
      document.removeEventListener('MSFullscreenChange', handleFullscreenChange);
    };
  }, []);

  const fetchVideos = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.get('/api/videos');
      const items = response.data?.videos || [];
      const normalized = items.map(normalizeVideo);
      setVideos(normalized);
      // generate thumbnails asynchronously
      generateThumbnails(normalized);
    } catch (err) {
      console.error('Failed to fetch videos:', err);
      if (err.response?.status === 401) {
        setError('Please log in to view videos');
      } else {
        setError('Failed to load videos');
      }
      setVideos([]);
    } finally {
      setLoading(false);
    }
  };

  // Generate thumbnails for videos by capturing a frame
  const generateThumbnails = (videoList) => {
    if (!videoList || !Array.isArray(videoList)) return;
    
    videoList.forEach((v) => {
      if (!v || !v.id) return;
      
      try {
        // Use relative URL to go through proxy
        const videoUrl = `/api/videos/${v.id}/stream`;
        const videoEl = document.createElement('video');
        videoEl.crossOrigin = 'use-credentials';  // Use credentials for CORS
        videoEl.src = videoUrl;
        videoEl.muted = true; // Mute to allow autoplay
        
        // Handle errors silently
        videoEl.addEventListener('error', () => {
          // Silently fail - don't crash the component
        });
        
        // try to capture at 1 second
        videoEl.addEventListener('loadeddata', () => {
          try {
            if (videoEl.duration) {
              videoEl.currentTime = Math.min(1, videoEl.duration / 2 || 0);
            }
          } catch (e) {
            // some browsers may block seeking before metadata; fallback
          }
        });
        
        const capture = () => {
          try {
            const canvas = document.createElement('canvas');
            canvas.width = videoEl.videoWidth || 300;
            canvas.height = videoEl.videoHeight || 200;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL('image/png');
            setThumbnails(prev => ({ ...prev, [v.id]: dataUrl }));
          } catch (e) {
            // drawing may fail due to CORS; ignore
          }
          videoEl.removeEventListener('seeked', capture);
        };
        videoEl.addEventListener('seeked', capture);
        
        // fallback: if cannot seek, try capture after a timeout
        setTimeout(() => {
          try {
            if (!thumbnails[v.id]) {
              const canvas = document.createElement('canvas');
              canvas.width = videoEl.videoWidth || 300;
              canvas.height = videoEl.videoHeight || 200;
              const ctx = canvas.getContext('2d');
              ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
              const dataUrl = canvas.toDataURL('image/png');
              setThumbnails(prev => ({ ...prev, [v.id]: dataUrl }));
            }
          } catch (e) {
            // Silently fail
          }
        }, 1500);
      } catch (e) {
        // Silently fail for thumbnail generation - don't crash the component
        console.error('Error generating thumbnail for video', v.id, e);
      }
    });
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0] || null;
    setSelectedFile(file);
    if (file) {
      const derivedTitle = file.name ? file.name.replace(/\.[^/.]+$/, '') : '';
      setTitle((prev) => (prev && prev.trim().length > 0 ? prev : derivedTitle));
    }
    setError(null);
  };

  const handleTitleInputChange = (event) => {
    setTitle(event.target.value);
    if (error) {
      setError(null);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    const trimmedTitle = title.trim();
    if (!trimmedTitle) {
      setError('Please enter a video title');
      return;
    }
    if (!selectedFile) {
      setError('Please select a video file');
      return;
    }
    const filenameLower = selectedFile.name ? selectedFile.name.toLowerCase() : '';
    if (!filenameLower.endsWith('.mp4')) {
      setError('Only MP4 videos are supported');
      return;
    }

    const formData = new FormData();
    formData.append('title', trimmedTitle);
    formData.append('video', selectedFile);

    setUploading(true);
    setError(null);
    try {
      // Don't set Content-Type header - axios will set it automatically with boundary for FormData
      const response = await api.post('/api/videos', formData, {
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(percent);
          }
        }
      });
      const created = response.data?.video ? normalizeVideo(response.data.video) : null;
      if (created) {
        setVideos(prev => [created, ...prev]);
      } else {
        await fetchVideos(); // Refresh the list
      }
      setSelectedFile(null);
      setTitle('');
      setUploadProgress(0);
      // Reset file input
      const fileInput = document.querySelector('input[type="file"]');
      if (fileInput) fileInput.value = '';
    } catch (error) {
      console.error('Failed to upload video:', error);
      setError(error.response?.data?.error || 'Failed to upload video');
    } finally {
      setUploading(false);
    }
  };

  const deleteVideo = async (videoId) => {
    const ok = window.confirm('Delete this video?');
    if (!ok) return;
    try {
      await api.delete(`/api/videos/${videoId}`);
      setVideos(videos.filter(video => video.id !== videoId));
      setThumbnails(prev => { const copy = { ...prev }; delete copy[videoId]; return copy; });
    } catch (error) {
      console.error('Failed to delete video:', error);
      setError('Failed to delete video');
    }
  };

  const canDeleteVideo = (video) => {
    return user && (user.is_admin || video.owner_id === user.id);
  };

  const playVideo = (video) => {
    setPlayingVideo(null); // Clear first to reset
    setVideoError(null); // Clear any previous errors
    setTimeout(() => {
      setPlayingVideo(video);
    }, 100);
  };

  const closeVideoPlayer = () => {
    // Exit fullscreen if active
    if (isFullscreen) {
      exitFullscreen();
    }
    setPlayingVideo(null);
    setVideoError(null);
    setIsFullscreen(false);
  };

  const toggleFullscreen = () => {
    const element = document.getElementById(`modal-content-${playingVideo.id}`);
    if (!element) return;

    if (!isFullscreen) {
      // Enter fullscreen
      if (element.requestFullscreen) {
        element.requestFullscreen().catch(err => {
          console.error('Error attempting to enable fullscreen:', err);
        });
      } else if (element.webkitRequestFullscreen) {
        element.webkitRequestFullscreen();
      } else if (element.mozRequestFullScreen) {
        element.mozRequestFullScreen();
      } else if (element.msRequestFullscreen) {
        element.msRequestFullscreen();
      }
    } else {
      // Exit fullscreen
      exitFullscreen();
    }
  };

  const exitFullscreen = () => {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    } else if (document.webkitExitFullscreen) {
      document.webkitExitFullscreen();
    } else if (document.mozCancelFullScreen) {
      document.mozCancelFullScreen();
    } else if (document.msExitFullscreen) {
      document.msExitFullscreen();
    }
  };

  // Show loading if auth is still loading
  if (authLoading) {
    return <div style={{ padding: '2rem', textAlign: 'center' }}>Loading...</div>;
  }

  if (!authLoading && !user) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center' }}>
        Please log in to view and upload videos.
      </div>
    );
  }

  const normalizedTitle = title.trim();
  const isUploadDisabled = !selectedFile || uploading || normalizedTitle.length === 0;

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Video Gallery</h1>
      {loading && <div style={styles.status}>Loading videos...</div>}
      {error && <div style={styles.error}>{error}</div>}

      {user && (
        <form onSubmit={handleUpload} style={styles.uploadForm}>
          <input
            type="text"
            value={title}
            onChange={handleTitleInputChange}
            placeholder="Enter video title"
            style={styles.titleInput}
            maxLength={120}
            aria-label="Video title"
            autoComplete="off"
            required
          />
          <label style={styles.fileInputLabel}>
            <input
              type="file"
              onChange={handleFileSelect}
              accept="video/*"
              style={styles.fileInput}
            />
            <span style={styles.fileInputText}>
              {selectedFile ? selectedFile.name : 'Choose File'}
            </span>
          </label>
          <button
            type="submit"
            disabled={isUploadDisabled}
            style={{
              ...styles.uploadButton,
              ...(isUploadDisabled ? styles.uploadButtonDisabled : {})
            }}
          >
            {uploading ? `Uploading... ${uploadProgress}%` : 'Upload Video'}
          </button>
        </form>
      )}

      <div style={styles.videoGrid}>
        {videos.map((video) => (
          <div key={video.id} style={styles.videoCard} className="video-card">
            <div 
              style={styles.videoThumbnailContainer} 
              className="video-thumbnail-container"
              onClick={() => playVideo(video)}
            >
              {thumbnails[video.id] ? (
                <>
                  <img src={thumbnails[video.id]} alt={video.filename || 'Video'} style={styles.thumbnail} />
                  <div style={styles.playButtonOverlay} className="play-button-overlay">
                    <div style={styles.playButton} className="play-button">
                      ▶
                    </div>
                  </div>
                </>
              ) : (
                <video
                  style={styles.video}
                  src={getVideoUrl(video.id)}
                  muted
                  onMouseEnter={(e) => {
                    e.target.play().catch(() => {});
                  }}
                  onMouseLeave={(e) => {
                    e.target.pause();
                    e.target.currentTime = 0;
                  }}
                />
              )}
            </div>
            <div style={styles.videoInfo}>
              <div style={styles.videoDetails}>
                <span style={styles.videoTitle}>{video.title}</span>
              </div>
              <div style={styles.videoActions}>
                <button
                  onClick={() => playVideo(video)}
                  style={styles.playButtonSmall}
                  title="Play video"
                >
                  ▶ Play
                </button>
                {canDeleteVideo(video) && (
                  <button
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      deleteVideo(video.id);
                    }}
                    style={styles.deleteButton}
                    title="Delete video"
                  >
                    🗑️
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {playingVideo && (
        <div 
          style={styles.modal} 
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              closeVideoPlayer();
            }
          }}
          onKeyDown={(e) => {
            if (e.key === 'Escape') {
              if (isFullscreen) {
                exitFullscreen();
              } else {
                closeVideoPlayer();
              }
            } else if (e.key === 'F11') {
              e.preventDefault();
              toggleFullscreen();
            }
          }}
          tabIndex={0}
        >
          <div 
            id={`modal-content-${playingVideo.id}`}
            style={styles.modalContent} 
            onClick={(e) => e.stopPropagation()}
          >
            <div style={styles.modalHeader}>
              <h3 style={styles.modalTitle}>{playingVideo.title}</h3>
              <div style={styles.headerButtons}>
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    toggleFullscreen();
                  }}
                  style={styles.fullscreenButton}
                  title={isFullscreen ? "Exit Fullscreen (F11 or ESC)" : "Enter Fullscreen (F11)"}
                >
                  {isFullscreen ? '🗗' : '🗖'}
                </button>
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    closeVideoPlayer();
                  }}
                  style={styles.closeButton}
                  title="Close (ESC)"
                >
                  ×
                </button>
              </div>
            </div>
            <div 
              id={`video-container-${playingVideo.id}`}
              style={styles.videoPlayerContainer}
            >
              {videoError && (
                <div style={styles.videoError}>
                  <p>Error loading video: {videoError}</p>
                  <button onClick={() => {
                    setVideoError(null);
                    const videoEl = videoPlayerRef.current;
                    if (videoEl) {
                      const videoUrl = getVideoUrl(playingVideo.id);
                      videoEl.src = videoUrl;
                      videoEl.load();
                    }
                  }} style={styles.retryButton}>
                    Retry
                  </button>
                </div>
              )}
              <video
                ref={videoPlayerRef}
                key={`video-${playingVideo.id}`}
                controls
                autoPlay
                playsInline
                preload="metadata"
                crossOrigin="use-credentials"
                style={{
                  ...styles.videoPlayer,
                  display: videoError ? 'none' : 'block'
                }}
                onError={(e) => {
                  const videoEl = e.target;
                  let errorMessage = 'Failed to load video';
                  if (videoEl?.error) {
                    switch (videoEl.error.code) {
                      case 1:
                        errorMessage = 'Video loading was aborted';
                        break;
                      case 2:
                        errorMessage = 'Network error while loading video';
                        break;
                      case 3:
                        errorMessage = 'Video decoding error. The file may be corrupted';
                        break;
                      case 4:
                        errorMessage = 'Video format not supported or file not found';
                        break;
                      default:
                        errorMessage = videoEl.error.message || 'Video playback error';
                    }
                  } else {
                    // Check if it's a network/CORS issue
                    const videoUrl = getVideoUrl(playingVideo.id);
                    console.warn('Video error:', videoUrl, videoEl.error);
                    errorMessage = 'Unable to load video. Please check your connection.';
                  }
                  setVideoError(errorMessage);
                }}
                onLoadedData={() => setVideoError(null)}
                onCanPlay={() => setVideoError(null)}
              >
                Your browser does not support the video tag.
              </video>
            </div>
            <div style={styles.modalActions}>
              {canDeleteVideo(playingVideo) && (
                <button
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    deleteVideo(playingVideo.id);
                    closeVideoPlayer();
                  }}
                  style={styles.modalDeleteButton}
                >
                  Delete Video
                </button>
              )}
            </div>
          </div>
        </div>
      )}
      {uploading && (
        <div style={{padding: '1rem'}}>
          Uploading: {uploadProgress}%
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
  status: {
    textAlign: 'center',
    color: '#666',
    margin: '0.5rem 0',
  },
  error: {
    textAlign: 'center',
    color: '#c62828',
    margin: '0.5rem 0',
    padding: '0.75rem',
    backgroundColor: '#ffebee',
    borderRadius: '4px',
  },
  uploadForm: {
    display: 'flex',
    gap: '1rem',
    marginBottom: '2rem',
    flexWrap: 'wrap',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    padding: '1.5rem',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    border: '1px solid rgba(102, 126, 234, 0.3)',
  },
  fileInput: {
    display: 'none',
  },
  fileInputLabel: {
    flex: '1 1 200px',
    padding: '0.75rem',
    backgroundColor: '#f5f5f5',
    border: '2px dashed #ddd',
    borderRadius: '4px',
    cursor: 'pointer',
    textAlign: 'center',
    transition: 'all 0.3s',
    '&:hover': {
      backgroundColor: '#e8e8e8',
      borderColor: '#2196f3',
    },
  },
  fileInputText: {
    display: 'block',
    color: '#666',
    fontSize: '0.9rem',
  },
  titleInput: {
    flex: '1 1 200px',
    padding: '0.75rem',
    fontSize: '1rem',
    border: '1px solid #ddd',
    borderRadius: '4px',
  },
  uploadButton: {
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
  uploadButtonDisabled: {
    backgroundColor: '#ccc',
    cursor: 'not-allowed',
  },
  videoGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
    gap: '1rem',
  },
  videoCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '15px',
    boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)',
    overflow: 'hidden',
    transition: 'all 0.3s ease',
    border: '1px solid rgba(102, 126, 234, 0.3)',
  },
  videoThumbnailContainer: {
    position: 'relative',
    cursor: 'pointer',
    overflow: 'hidden',
    backgroundColor: '#000',
  },
  thumbnail: {
    width: '100%',
    height: 'auto',
    display: 'block',
    objectFit: 'cover',
    transition: 'transform 0.3s',
  },
  playButtonOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.4)',
    transition: 'background-color 0.3s',
  },
  playButton: {
    width: '60px',
    height: '60px',
    borderRadius: '50%',
    backgroundColor: 'rgba(33, 150, 243, 0.9)',
    color: 'white',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    fontSize: '24px',
    fontWeight: 'bold',
    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.3)',
    transition: 'transform 0.3s',
    transform: 'scale(1)',
  },
  video: {
    width: '100%',
    height: 'auto',
    display: 'block',
  },
  videoInfo: {
    padding: '0.75rem',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: '1rem',
  },
  videoActions: {
    display: 'flex',
    gap: '0.5rem',
    alignItems: 'center',
  },
  playButtonSmall: {
    padding: '0.5rem 1rem',
    backgroundColor: '#2196f3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '0.9rem',
    fontWeight: '500',
    transition: 'background-color 0.2s',
  },
  videoDetails: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.25rem',
    flex: 1,
  },
  videoTitle: {
    fontWeight: '500',
    color: '#333',
  },
  videoOwner: {
    fontSize: '0.85rem',
    color: '#666',
  },
  deleteButton: {
    padding: '0.5rem 1rem',
    backgroundColor: '#ff5252',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '0.9rem',
  },
  modal: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 2000,
    padding: '2rem',
  },
  modalContent: {
    backgroundColor: 'rgba(26, 26, 26, 0.98)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    maxWidth: '90vw',
    maxHeight: '90vh',
    overflow: 'auto',
    display: 'flex',
    flexDirection: 'column',
    width: '100%',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)',
    border: '1px solid rgba(102, 126, 234, 0.15)',
  },
  modalHeader: {
    padding: '1rem 1.5rem',
    borderBottom: '1px solid #333',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: '1rem',
    flexWrap: 'wrap',
  },
  headerButtons: {
    display: 'flex',
    gap: '0.5rem',
    alignItems: 'center',
  },
  fullscreenButton: {
    background: 'rgba(102, 126, 234, 0.15)',
    border: '1px solid rgba(102, 126, 234, 0.2)',
    borderRadius: '4px',
    fontSize: '1.3rem',
    color: '#fff',
    cursor: 'pointer',
    padding: '0.5rem 1rem',
    lineHeight: 1,
    transition: 'all 0.2s',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: '40px',
    '&:hover': {
      background: 'rgba(102, 126, 234, 0.2)',
    },
  },
  modalTitle: {
    margin: 0,
    color: 'white',
    fontSize: '1.2rem',
    flex: 1,
  },
  modalOwner: {
    color: '#aaa',
    fontSize: '0.9rem',
  },
  closeButton: {
    background: 'none',
    border: 'none',
    fontSize: '2rem',
    color: '#ccc',
    cursor: 'pointer',
    padding: '0 0.5rem',
    lineHeight: 1,
    transition: 'color 0.2s',
  },
  videoPlayerContainer: {
    padding: '1rem',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
    minHeight: '400px',
    width: '100%',
  },
  videoPlayer: {
    maxWidth: '100%',
    maxHeight: '70vh',
    width: 'auto',
    height: 'auto',
    backgroundColor: '#000',
    display: 'block',
  },
  videoError: {
    padding: '2rem',
    textAlign: 'center',
    color: '#fff',
  },
  retryButton: {
    marginTop: '1rem',
    padding: '0.75rem 1.5rem',
    backgroundColor: '#2196f3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '1rem',
    fontWeight: '500',
  },
  modalActions: {
    padding: '1rem 1.5rem',
    borderTop: '1px solid #333',
    display: 'flex',
    gap: '1rem',
    justifyContent: 'flex-end',
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

export default VideoGallery;