import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import api from '../../services/api';

const styles = {
  container: {
    maxWidth: '1100px',
    margin: '0 auto',
    padding: '2rem 1.5rem 4rem',
    display: 'flex',
    flexDirection: 'column',
    minHeight: '100vh',
  },
  header: {
    marginBottom: '1.75rem',
  },
  title: {
    fontSize: '2rem',
    fontWeight: '700',
    color: '#1f2937',
    margin: 0,
  },
  subtitle: {
    marginTop: '0.5rem',
    color: '#4b5563',
    fontSize: '1rem',
  },
  body: {
    flex: 1,
    display: 'flex',
    flexWrap: 'wrap',
    gap: '1.5rem',
  },
  sidebar: {
    flex: '1 1 260px',
    minWidth: '240px',
    maxWidth: '320px',
    background: 'linear-gradient(180deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
    borderRadius: '16px',
    border: '1px solid rgba(102, 126, 234, 0.2)',
    boxShadow: '0 12px 32px rgba(102, 126, 234, 0.15)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    backdropFilter: 'blur(10px)',
  },
  chatArea: {
    flex: '3 1 520px',
    minWidth: '280px',
    background: 'linear-gradient(180deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%)',
    borderRadius: '16px',
    border: '1px solid rgba(102, 126, 234, 0.2)',
    boxShadow: '0 12px 32px rgba(102, 126, 234, 0.15)',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    backdropFilter: 'blur(10px)',
  },
  chatListHeader: {
    padding: '1.25rem',
    borderBottom: '1px solid rgba(102, 126, 234, 0.2)',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
  },
  chatListTitle: {
    margin: 0,
    fontSize: '1rem',
    fontWeight: 600,
    color: '#111827',
  },
  newChatButton: {
    backgroundColor: '#2563eb',
    color: '#ffffff',
    border: 'none',
    borderRadius: '999px',
    padding: '0.45rem 1rem',
    fontSize: '0.85rem',
    fontWeight: 600,
    cursor: 'pointer',
    boxShadow: '0 8px 20px rgba(37, 99, 235, 0.25)',
  },
  newChatButtonDisabled: {
    backgroundColor: '#9ca3af',
    boxShadow: 'none',
    cursor: 'not-allowed',
  },
  chatList: {
    flex: 1,
    overflowY: 'auto',
    padding: '0.5rem',
  },
  chatListEmpty: {
    padding: '1.5rem 1rem',
    color: '#6b7280',
    fontSize: '0.9rem',
  },
  chatListItem: {
    borderRadius: '12px',
    padding: '0.85rem 1rem',
    marginBottom: '0.5rem',
    cursor: 'pointer',
    background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(240, 245, 255, 0.9) 100%)',
    border: '1px solid rgba(102, 126, 234, 0.2)',
    transition: 'transform 0.15s ease, box-shadow 0.15s ease, border 0.15s ease',
    boxShadow: '0 10px 20px rgba(102, 126, 234, 0.1)',
  },
  chatListItemActive: {
    background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%)',
    border: '1px solid rgba(102, 126, 234, 0.4)',
    boxShadow: '0 8px 18px rgba(102, 126, 234, 0.25)',
  },
  chatListItemHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '0.75rem',
  },
  chatListItemText: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.35rem',
    flex: 1,
    minWidth: 0,
  },
  chatListItemTitle: {
    margin: 0,
    fontSize: '0.95rem',
    fontWeight: 600,
    color: '#1f2937',
    lineHeight: 1.3,
  },
  chatListItemMeta: {
    marginTop: '0.35rem',
    fontSize: '0.8rem',
    color: '#64748b',
  },
  chatDeleteButton: {
    backgroundColor: 'transparent',
    border: 'none',
    color: '#ef4444',
    fontSize: '1.15rem',
    cursor: 'pointer',
    padding: '0.25rem 0.4rem',
    borderRadius: '8px',
    minWidth: '38px',
    minHeight: '36px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'background-color 0.15s ease, color 0.15s ease',
  },
  chatHeader: {
    padding: '1.25rem',
    borderBottom: '1px solid rgba(102, 126, 234, 0.2)',
    background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
  },
  chatTitle: {
    margin: 0,
    fontSize: '1.125rem',
    fontWeight: 600,
    color: '#111827',
  },
  chatSubtitle: {
    marginTop: '0.35rem',
    color: '#6b7280',
    fontSize: '0.9rem',
  },
  transcript: {
    flex: 1,
    padding: '1.5rem',
    overflowY: 'auto',
    background: 'linear-gradient(180deg, rgba(248, 250, 252, 0.5) 0%, rgba(240, 245, 255, 0.5) 100%)',
    position: 'relative',
  },
  loadingOverlay: {
    position: 'absolute',
    top: '0',
    left: '0',
    right: '0',
    bottom: '0',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(248, 250, 252, 0.85)',
    fontSize: '0.95rem',
    color: '#4b5563',
  },
  messageWrapper: {
    marginBottom: '1rem',
    display: 'flex',
  },
  userBubble: {
    marginLeft: 'auto',
    background: '#2563eb',
    color: 'white',
  },
  assistantBubble: {
    marginRight: 'auto',
    background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(240, 245, 255, 0.95) 100%)',
    color: '#1f2937',
    border: '1px solid rgba(102, 126, 234, 0.2)',
  },
  bubbleBase: {
    padding: '0.85rem 1rem',
    borderRadius: '14px',
    maxWidth: '75%',
    lineHeight: 1.5,
    fontSize: '0.95rem',
    boxShadow: '0 4px 12px rgba(15, 23, 42, 0.08)',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  },
  composer: {
    borderTop: '1px solid rgba(102, 126, 234, 0.2)',
    padding: '1rem',
    background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(240, 245, 255, 0.95) 100%)',
  },
  textarea: {
    width: '100%',
    minHeight: '80px',
    resize: 'vertical',
    padding: '0.75rem 1rem',
    fontSize: '1rem',
    borderRadius: '10px',
    border: '1px solid #cbd5f5',
    outline: 'none',
    backgroundColor: '#f9fafb',
    color: '#111827',
  },
  actions: {
    marginTop: '0.75rem',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: '0.75rem',
  },
  sendButton: {
    backgroundColor: '#2563eb',
    color: '#ffffff',
    border: 'none',
    padding: '0.65rem 1.6rem',
    borderRadius: '999px',
    fontWeight: '600',
    fontSize: '0.95rem',
    cursor: 'pointer',
    display: 'inline-flex',
    alignItems: 'center',
    gap: '0.5rem',
    boxShadow: '0 10px 20px rgba(37, 99, 235, 0.25)',
    transition: 'transform 0.15s ease, box-shadow 0.15s ease',
  },
  sendButtonDisabled: {
    backgroundColor: '#9ca3af',
    boxShadow: 'none',
    cursor: 'not-allowed',
  },
  status: {
    fontSize: '0.85rem',
    color: '#6b7280',
  },
  error: {
    color: '#dc2626',
    fontSize: '0.9rem',
    marginTop: '0.5rem',
  },
  searchToggle: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    fontSize: '0.85rem',
    color: '#475569',
    marginTop: '0.75rem',
  },
  searchToggleInput: {
    width: '18px',
    height: '18px',
  },
  searchResultsBox: {
    marginBottom: '1rem',
    padding: '1rem',
    backgroundColor: '#f1f5f9',
    borderRadius: '12px',
    border: '1px solid #cbd5e1',
    boxShadow: '0 6px 16px rgba(15, 23, 42, 0.1)',
  },
  searchResultsHeader: {
    fontWeight: 600,
    fontSize: '0.95rem',
    color: '#1f2937',
    marginBottom: '0.5rem',
  },
  searchResultsSection: {
    marginBottom: '0.75rem',
  },
  searchResultsSource: {
    fontWeight: 600,
    fontSize: '0.85rem',
    color: '#2563eb',
    marginBottom: '0.4rem',
  },
  searchResultsList: {
    listStyle: 'disc',
    paddingLeft: '1.25rem',
    margin: 0,
    display: 'flex',
    flexDirection: 'column',
    gap: '0.5rem',
  },
  searchResultItem: {
    fontSize: '0.85rem',
    color: '#1f2937',
  },
  searchResultLink: {
    color: '#1d4ed8',
    textDecoration: 'none',
    fontWeight: 600,
  },
  searchResultSnippet: {
    marginTop: '0.2rem',
    color: '#475569',
  },
};

const initialAssistantMessage = {
  role: 'assistant',
  content:
    'Hi! I am Friendly Friends AI. Ask me anything about science, wellness, art or the Friendly Friends AI community and I will do my best to help 🌟',
};

function AiChat() {
  const [chats, setChats] = useState([]);
  const [selectedChat, setSelectedChat] = useState(null);
  const [messages, setMessages] = useState([]);
  const [loadingChats, setLoadingChats] = useState(true);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [input, setInput] = useState('');
  const [error, setError] = useState(null);
  const [latestSearchResults, setLatestSearchResults] = useState(null);
  const transcriptEndRef = useRef(null);

  const displayMessages = useMemo(() => {
    if (messages && messages.length > 0) {
      return messages;
    }
    return [initialAssistantMessage];
  }, [messages]);

  const scrollToBottom = useCallback(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [displayMessages, sending, scrollToBottom]);

  const loadChatMessages = useCallback(async (chatId) => {
    if (!chatId) {
      setSelectedChat(null);
      setMessages([]);
      return;
    }
    setMessagesLoading(true);
    setError(null);
    try {
      const { data } = await api.get(`/api/ai/chats/${chatId}/messages`);
      // Update selected chat and messages
      if (data?.chat) {
        setSelectedChat(data.chat);
      }
      setMessages(Array.isArray(data?.messages) ? data.messages : []);
    } catch (err) {
      const serverMessage =
        err.response?.data?.error || 'Unable to load this chat right now. Please try again shortly.';
      setError(serverMessage);
      // Clear messages on error
      setMessages([]);
    } finally {
      setMessagesLoading(false);
    }
  }, []);

  const fetchChats = useCallback(async () => {
    setLoadingChats(true);
    setError(null);
    try {
      const { data } = await api.get('/api/ai/chats');
      const chatList = Array.isArray(data?.chats) ? data.chats : [];
      setChats(chatList);

      if (chatList.length === 0) {
        setSelectedChat(null);
        setMessages([]);
        return;
      }

      if (!selectedChat) {
        await loadChatMessages(chatList[0].id);
        return;
      }

      const stillExists = chatList.some((chat) => chat.id === selectedChat.id);
      if (!stillExists) {
        await loadChatMessages(chatList[0].id);
      }
    } catch (err) {
      const serverMessage =
        err.response?.data?.error || 'Unable to load your chats right now. Please try again later.';
      setError(serverMessage);
    } finally {
      setLoadingChats(false);
    }
  }, [loadChatMessages, selectedChat]);

  useEffect(() => {
    fetchChats();
  }, [fetchChats]);

  const handleSelectChat = useCallback(
    async (chat) => {
      if (!chat || sending) {
        return;
      }
      // Set selected chat immediately for visual feedback
      setSelectedChat(chat);
      // Clear search results
      setLatestSearchResults(null);
      setError(null);
      // Load messages for the selected chat (always reload to ensure fresh data)
      await loadChatMessages(chat.id);
    },
    [loadChatMessages, sending],
  );

  const handleCreateChat = useCallback(async () => {
    if (sending) {
      return;
    }
    setError(null);
    try {
      const { data } = await api.post('/api/ai/chats', {});
      const chat = data?.chat;
      if (chat) {
        setChats((prev) => [chat, ...prev.filter((item) => item.id !== chat.id)]);
        await loadChatMessages(chat.id);
        setLatestSearchResults(null);
      }
    } catch (err) {
      const serverMessage =
        err.response?.data?.error || 'Unable to start a new chat right now. Please try again shortly.';
      setError(serverMessage);
    }
  }, [loadChatMessages, sending]);

  const handleDeleteChat = useCallback(async (chat) => {
    if (!chat) {
      return;
    }
    const confirmDelete = window.confirm('Delete this conversation? This will remove its AI messages.');
    if (!confirmDelete) {
      return;
    }

    setError(null);
    try {
      const response = await api.delete(`/api/ai/chats/${chat.id}`);
      
      // Remove the deleted chat from the list
      const remaining = chats.filter((item) => item.id !== chat.id);
      setChats(remaining);

      // If the deleted chat was selected, switch to another chat or clear
      if (selectedChat?.id === chat.id) {
        if (remaining.length > 0) {
          // Switch to the first remaining chat
          await loadChatMessages(remaining[0].id);
        } else {
          // No chats left, clear selection
          setSelectedChat(null);
          setMessages([]);
        }
      }
      
      // Clear any error messages on success
      setError(null);
    } catch (err) {
      console.error('Error deleting chat:', err);
      const serverMessage = err.response?.data?.error || 'Unable to delete this chat right now.';
      setError(serverMessage);
      // Show error to user
      alert(`Failed to delete chat: ${serverMessage}`);
    }
  }, [chats, loadChatMessages, selectedChat]);

  const sendMessage = useCallback(async () => {
    const trimmed = input.trim();
    if (!trimmed || sending) {
      return;
    }

    const optimisticMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: trimmed,
      created_at: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, optimisticMessage]);
    setInput('');
    setError(null);
    setSending(true);
    setLatestSearchResults(null);

    try {
      const payload = { message: trimmed };
      if (selectedChat?.id) {
        payload.chat_id = selectedChat.id;
      } else {
        payload.title = trimmed.slice(0, 60);
      }
      payload.use_search = true;

      const { data } = await api.post('/api/ai/chat', payload);
      const chat = data?.chat;
      const serverMessages = Array.isArray(data?.messages) ? data.messages : null;
      setLatestSearchResults(data?.search || null);

      if (chat) {
        setChats((prev) => [chat, ...prev.filter((item) => item.id !== chat.id)]);
        setSelectedChat(chat);
      }

      if (serverMessages) {
        setMessages(serverMessages);
      } else {
        const reply =
          data?.reply || 'I had trouble replying just now, but you can ask me again!';
        setMessages((prev) => [
          ...prev,
          {
            id: `assistant-${Date.now()}`,
            role: 'assistant',
            content: reply,
            created_at: new Date().toISOString(),
          },
        ]);
      }
      if (!data?.search) {
        setLatestSearchResults(null);
      }
    } catch (err) {
      const serverMessage =
        err.response?.data?.error ||
        'The AI service is unavailable right now. Please try again in a little while.';
      setError(serverMessage);
      setMessages((prev) => [
        ...prev,
        {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: `⚠️ ${serverMessage}`,
          created_at: new Date().toISOString(),
        },
      ]);
      setLatestSearchResults(null);
    } finally {
      setSending(false);
    }
  }, [input, selectedChat, sending]);

  const handleKeyDown = useCallback(
    (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    },
    [sendMessage],
  );

  const formatTimestamp = useCallback((value) => {
    if (!value) {
      return '';
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return '';
    }
    return date.toLocaleString();
  }, []);

  const isSendDisabled = useMemo(() => sending || input.trim().length === 0, [input, sending]);

  const searchResultContent = useMemo(() => {
    if (!latestSearchResults || typeof latestSearchResults !== 'object') {
      return null;
    }
    const entries = Object.entries(latestSearchResults).filter(([, value]) => Array.isArray(value) && value.length > 0);
    if (entries.length === 0) {
      return null;
    }
    return (
      <div style={styles.searchResultsBox}>
        <div style={styles.searchResultsHeader}>🌐 Web insights (Wikipedia • Reddit • Friendly Friends AI Blog)</div>
        {entries.map(([source, items]) => {
          const labelMap = {
            wikipedia: 'Wikipedia',
            reddit: 'Reddit',
            blog: 'Friendly Friends AI Blog'
          };
          return (
            <div key={source} style={styles.searchResultsSection}>
              <div style={styles.searchResultsSource}>{labelMap[source] || source}</div>
              <ul style={styles.searchResultsList}>
                {items.map((item, index) => (
                  <li key={`${source}-${index}`} style={styles.searchResultItem}>
                    <a href={item.url} target="_blank" rel="noreferrer" style={styles.searchResultLink}>
                      {item.title || 'View result'}
                    </a>
                    {item.snippet && <div style={styles.searchResultSnippet}>{item.snippet}</div>}
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
      </div>
    );
  }, [latestSearchResults]);

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>🤖 Friendly Friends AI Chat</h1>
        <p style={styles.subtitle}>
          Ask questions, brainstorm ideas, or get a friendly boost. Your chats stay tied to your
          account so you can continue the conversation from any device.
        </p>
      </header>

      <div style={styles.body}>
        <aside style={styles.sidebar}>
          <div style={styles.chatListHeader}>
            <h2 style={styles.chatListTitle}>Conversations</h2>
            <button
              type="button"
              onClick={handleCreateChat}
              disabled={sending || messagesLoading}
              style={{
                ...styles.newChatButton,
                ...((sending || messagesLoading) ? styles.newChatButtonDisabled : {}),
              }}
            >
              New chat
            </button>
          </div>

          <div style={styles.chatList}>
            {loadingChats ? (
              <div style={styles.chatListEmpty}>Loading chats…</div>
            ) : chats.length === 0 ? (
              <div style={styles.chatListEmpty}>
                No conversations yet. Start a new chat to get personalised answers.
              </div>
            ) : (
              chats.map((chat) => {
                const isActive = selectedChat?.id === chat.id;
                const itemStyle = {
                  ...styles.chatListItem,
                  ...(isActive ? styles.chatListItemActive : {}),
                };
                const meta = formatTimestamp(chat.created_at) || 'Just now';
                return (
                  <div
                    key={chat.id}
                    style={itemStyle}
                    onClick={() => handleSelectChat(chat)}
                    role="button"
                    tabIndex={0}
                    onKeyPress={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        handleSelectChat(chat);
                      }
                    }}
                  >
                    <div style={styles.chatListItemHeader}>
                      <div style={styles.chatListItemText}>
                        <h3 style={styles.chatListItemTitle}>{chat.title || 'Untitled chat'}</h3>
                        <div style={styles.chatListItemMeta}>{meta}</div>
                      </div>
                      <button
                        type="button"
                        style={styles.chatDeleteButton}
                        title="Delete chat"
                        aria-label="Delete chat"
                        onClick={(event) => {
                          event.stopPropagation();
                          handleDeleteChat(chat);
                        }}
                      >
                        🗑️
                      </button>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </aside>

        <section style={styles.chatArea}>
          <div style={styles.chatHeader}>
            <h3 style={styles.chatTitle}>{selectedChat?.title || 'Start a conversation'}</h3>
            <p style={styles.chatSubtitle}>
              {selectedChat
                ? `Created ${formatTimestamp(selectedChat.created_at) || 'just now'}`
                : 'Pick a chat from the left or start a new one to begin.'}
            </p>
          </div>

          <div style={styles.transcript}>
            {searchResultContent}
            {messagesLoading && <div style={styles.loadingOverlay}>Loading messages…</div>}
            {displayMessages.map((message, index) => {
              const isUser = message.role === 'user';
              const bubbleStyle = {
                ...styles.bubbleBase,
                ...(isUser ? styles.userBubble : styles.assistantBubble),
              };
              const wrapperStyle = {
                ...styles.messageWrapper,
                justifyContent: isUser ? 'flex-end' : 'flex-start',
              };
              const key = message.id || `${message.role}-${index}`;

              return (
                <div key={key} style={wrapperStyle}>
                  <div style={bubbleStyle}>{message.content}</div>
                </div>
              );
            })}
            <div ref={transcriptEndRef} />
          </div>

          <div style={styles.composer}>
            <textarea
              style={styles.textarea}
              placeholder="Ask Friendly Friends AI anything..."
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              disabled={sending || messagesLoading}
            />
            <div style={styles.actions}>
              <span style={styles.status}>
                {sending ? 'Thinking…' : 'Press Enter to send, Shift + Enter for newline.'}
              </span>
              <button
                type="button"
                onClick={sendMessage}
                disabled={isSendDisabled}
                style={{
                  ...styles.sendButton,
                  ...(isSendDisabled ? styles.sendButtonDisabled : {}),
                }}
              >
                {sending ? 'Sending…' : 'Send'}
              </button>
            </div>
            {error && <div style={styles.error}>{error}</div>}
          </div>
        </section>
      </div>
    </div>
  );
}

export default AiChat;

