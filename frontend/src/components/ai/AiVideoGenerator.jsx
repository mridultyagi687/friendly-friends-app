import React, { useCallback, useEffect, useMemo, useState } from 'react';
import api from '../../services/api';

const styles = {
  container: {
    maxWidth: '1100px',
    margin: '0 auto',
    padding: '2rem 1.5rem 4rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '2rem',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: '1rem',
  },
  title: {
    fontSize: '2.2rem',
    fontWeight: 700,
    color: '#0f172a',
    margin: 0,
  },
  subtitle: {
    fontSize: '1rem',
    color: '#475569',
    maxWidth: '600px',
  },
  form: {
    backgroundColor: '#ffffff',
    borderRadius: '16px',
    border: '1px solid #e2e8f0',
    padding: '1.75rem',
    boxShadow: '0 20px 40px rgba(15, 23, 42, 0.08)',
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
  },
  label: {
    display: 'block',
    fontWeight: 600,
    color: '#1f2937',
    marginBottom: '0.5rem',
  },
  textarea: {
    width: '100%',
    minHeight: '120px',
    borderRadius: '12px',
    border: '1px solid #cbd5f5',
    padding: '1rem',
    fontSize: '1rem',
    backgroundColor: '#f8fafc',
    color: '#0f172a',
    resize: 'vertical',
    lineHeight: 1.5,
  },
  controlsRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '1rem',
    alignItems: 'center',
  },
  select: {
    minWidth: '200px',
    padding: '0.65rem 1rem',
    borderRadius: '999px',
    border: '1px solid #cbd5f5',
    fontSize: '0.95rem',
    backgroundColor: '#f8fafc',
    color: '#0f172a',
  },
  toggleWrapper: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.75rem',
    backgroundColor: '#f1f5f9',
    borderRadius: '999px',
    padding: '0.35rem 0.75rem',
  },
  toggleLabel: {
    fontWeight: 600,
    color: '#1e293b',
  },
  switch: {
    position: 'relative',
    width: '48px',
    height: '26px',
    backgroundColor: '#cbd5e1',
    borderRadius: '999px',
    cursor: 'pointer',
    transition: 'background-color 0.2s ease',
  },
  switchOn: {
    backgroundColor: '#2563eb',
  },
  switchKnob: (active) => ({
    position: 'absolute',
    top: '3px',
    left: active ? '24px' : '4px',
    width: '20px',
    height: '20px',
    borderRadius: '50%',
    backgroundColor: '#ffffff',
    boxShadow: '0 2px 6px rgba(15, 23, 42, 0.2)',
    transition: 'left 0.2s ease',
  }),
  submitRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
  },
  submitButton: {
    backgroundColor: '#2563eb',
    color: '#ffffff',
    border: 'none',
    padding: '0.75rem 2rem',
    borderRadius: '999px',
    fontWeight: 600,
    fontSize: '1rem',
    cursor: 'pointer',
    boxShadow: '0 15px 25px rgba(37, 99, 235, 0.25)',
    transition: 'transform 0.15s ease, box-shadow 0.15s ease',
  },
  submitDisabled: {
    backgroundColor: '#94a3b8',
    cursor: 'not-allowed',
    boxShadow: 'none',
  },
  status: {
    fontSize: '0.95rem',
    color: '#64748b',
  },
  error: {
    color: '#dc2626',
    fontSize: '0.95rem',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
    gap: '1.5rem',
  },
  card: {
    backgroundColor: '#ffffff',
    borderRadius: '14px',
    border: '1px solid #e2e8f0',
    boxShadow: '0 12px 30px rgba(15, 23, 42, 0.08)',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
  },
  cardBody: {
    padding: '1rem 1.25rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
  },
  badgeRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '0.5rem',
  },
  badge: {
    fontSize: '0.75rem',
    padding: '0.35rem 0.6rem',
    borderRadius: '999px',
    backgroundColor: '#eff6ff',
    color: '#1d4ed8',
    fontWeight: 600,
  },
  badgeSecondary: {
    backgroundColor: '#f8fafc',
    color: '#334155',
  },
  statusBadge: {
    backgroundColor: '#e0f2fe',
    color: '#0369a1',
  },
  statusBadgeReady: {
    backgroundColor: '#dcfce7',
    color: '#15803d',
  },
  preview: {
    width: '100%',
    aspectRatio: '16 / 9',
    backgroundColor: '#0f172a',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#94a3b8',
    fontWeight: 600,
  },
  emptyState: {
    textAlign: 'center',
    padding: '3rem 1rem',
    border: '2px dashed #e2e8f0',
    borderRadius: '14px',
    color: '#64748b',
    backgroundColor: '#f8fafc',
  },
};

const videoStyles = [
  { value: 'cinematic', label: 'Cinematic' },
  { value: 'modern', label: 'Modern Motion' },
  { value: 'dreamy', label: 'Dreamy Soft Focus' },
  { value: 'retro', label: 'Retro Future' },
];

function AiVideoGenerator() {
    const [prompt, setPrompt] = useState('');
    const [style, setStyle] = useState('cinematic');
    const [audioEnabled, setAudioEnabled] = useState(true);
    const [videos, setVideos] = useState([]);
    const [loading, setLoading] = useState(true);
    const [generating, setGenerating] = useState(false);
    const [error, setError] = useState(null);

    const isSubmitDisabled = useMemo(() => generating || !prompt.trim(), [generating, prompt]);

    const fetchVideos = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const { data } = await api.get('/api/ai/videos');
            setVideos(Array.isArray(data?.videos) ? data.videos : []);
        } catch (err) {
            const message = err.response?.data?.error || 'Unable to load AI videos right now.';
            setError(message);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchVideos();
    }, [fetchVideos]);

    const handleSubmit = useCallback(async (event) => {
        event.preventDefault();
        if (isSubmitDisabled) return;
        setGenerating(true);
        setError(null);

        try {
            const payload = {
                prompt: prompt.trim(),
                style,
                audioEnabled,
            };
            const { data } = await api.post('/api/ai/videos', payload);
            if (data?.video) {
                setVideos((prev) => [data.video, ...prev.filter((video) => video.id !== data.video.id)]);
            }
            setPrompt('');
        } catch (err) {
            const message = err.response?.data?.error || 'Video generation failed. Please try again shortly.';
            setError(message);
        } finally {
            setGenerating(false);
        }
    }, [audioEnabled, isSubmitDisabled, prompt, style]);

    const renderStatusBadge = (status) => {
        const base = { ...styles.badge, ...styles.statusBadge };
        if (status === 'ready') return { ...base, ...styles.statusBadgeReady };
        if (status === 'failed') return { ...base, backgroundColor: '#fee2e2', color: '#b91c1c' };
        if (status === 'generating') return { ...base, backgroundColor: '#fef9c3', color: '#b45309' };
        return base;
    };

    return (
        <div style={styles.container}>
            <header style={styles.header}>
                <div>
                    <h1 style={styles.title}>🎥 Friendly Friends AI Video Lab</h1>
                    <p style={styles.subtitle}>
                        Generate short highlight clips powered by OpenAI. Describe the scene and choose whether to lay down our modern backing track.
                    </p>
                </div>
            </header>

            <form style={styles.form} onSubmit={handleSubmit}>
                <div>
                    <label htmlFor="prompt" style={styles.label}>Your video idea</label>
                    <textarea
                        id="prompt"
                        style={styles.textarea}
                        placeholder="Example: A time-lapse of flowers blooming under a neon night sky with a gentle camera orbit."
                        value={prompt}
                        onChange={(event) => setPrompt(event.target.value)}
                        disabled={generating}
                    />
                </div>

                <div style={styles.controlsRow}>
                    <div>
                        <label htmlFor="style" style={styles.label}>Visual style</label>
                        <select
                            id="style"
                            style={styles.select}
                            value={style}
                            onChange={(event) => setStyle(event.target.value)}
                            disabled={generating}
                        >
                            {videoStyles.map((option) => (
                                <option key={option.value} value={option.value}>{option.label}</option>
                            ))}
                        </select>
                    </div>

                    <div style={styles.toggleWrapper} onClick={() => setAudioEnabled((prev) => !prev)}>
                        <span style={styles.toggleLabel}>Modern audio</span>
                        <div style={{
                            ...styles.switch,
                            ...(audioEnabled ? styles.switchOn : {}),
                        }}>
                            <span style={styles.switchKnob(audioEnabled)} />
                        </div>
                        <span style={{ fontWeight: 600, color: audioEnabled ? '#2563eb' : '#64748b' }}>
                            {audioEnabled ? 'On' : 'Off'}
                        </span>
                    </div>
                </div>

                <div style={styles.submitRow}>
                    <button
                        type="submit"
                        style={{
                            ...styles.submitButton,
                            ...(isSubmitDisabled ? styles.submitDisabled : {}),
                        }}
                        disabled={isSubmitDisabled}
                    >
                        {generating ? 'Rendering…' : 'Generate video'}
                    </button>
                    <span style={styles.status}>
                        {generating
                            ? 'Hang tight! This usually takes a few moments.'
                            : 'Videos render with MP4 playback in your browser.'}
                    </span>
                </div>
                {error && <div style={styles.error}>⚠️ {error}</div>}
            </form>

            {loading ? (
                <div style={styles.emptyState}>Loading your AI generated videos…</div>
            ) : videos.length === 0 ? (
                <div style={styles.emptyState}>
                    No AI videos yet. Describe something magical above and let Friendly Friends AI bring it to life!
                </div>
            ) : (
                <div style={styles.grid}>
                    {videos.map((video) => {
                        const badges = [
                            { label: video.style ? video.style : 'Classic', secondary: true },
                            { label: video.audioEnabled ? 'Modern audio on' : 'Audio muted', secondary: true },
                        ];
                        return (
                            <div key={video.id} style={styles.card}>
                                {video.status === 'ready' && video.streamUrl ? (
                                    <video
                                        style={{ width: '100%', display: 'block' }}
                                        controls
                                        preload="metadata"
                                    >
                                        <source src={video.streamUrl} type="video/mp4" />
                                        Your browser does not support the video tag.
                                    </video>
                                ) : (
                                    <div style={styles.preview}>
                                        {video.status === 'generating' && 'Rendering…'}
                                        {video.status === 'failed' && 'Generation failed'}
                                        {video.status !== 'generating' && video.status !== 'failed' && 'Preparing…'}
                                    </div>
                                )}
                                <div style={styles.cardBody}>
                                    <span style={{ ...styles.badge, ...renderStatusBadge(video.status) }}>{video.status}</span>
                                    <div style={styles.badgeRow}>
                                        {badges.map((item) => (
                                            <span
                                                key={item.label}
                                                style={{
                                                    ...styles.badge,
                                                    ...(item.secondary ? styles.badgeSecondary : {}),
                                                }}
                                            >
                                                {item.label}
                                            </span>
                                        ))}
                                    </div>
                                    <p style={{ margin: '0.5rem 0', color: '#0f172a', fontWeight: 500 }}>
                                        {video.prompt}
                                    </p>
                                    <p style={{ margin: 0, fontSize: '0.8rem', color: '#94a3b8' }}>
                                        {video.createdAt ? `Created ${new Date(video.createdAt).toLocaleString()}` : ''}
                                    </p>
                                    {video.meta?.error && (
                                        <p style={{ ...styles.error, margin: 0 }}>
                                            ⚠️ {video.meta.error}
                                        </p>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}

export default AiVideoGenerator;
