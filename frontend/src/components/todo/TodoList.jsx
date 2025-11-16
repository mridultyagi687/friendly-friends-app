import React, { useState, useEffect } from 'react';
import api from '../../services/api';

function TodoList() {
  const [todos, setTodos] = useState([]);
  const [newTodo, setNewTodo] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [editingId, setEditingId] = useState(null);
  const [editText, setEditText] = useState('');

  useEffect(() => {
    fetchTodos();
  }, []);

  const fetchTodos = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.get('/api/todos');
      setTodos(response.data?.todos || []);
    } catch (err) {
      console.error('Failed to fetch todos:', err);
      setError('Failed to load todos');
      setTodos([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!newTodo.trim()) {
      setError('Please enter a todo title');
      return;
    }

    try {
      const response = await api.post('/api/todos', { title: newTodo.trim() });
      const created = response.data?.todo;
      if (created) {
        setTodos([...todos, created]);
        setNewTodo('');
        setError(null);
      }
    } catch (error) {
      console.error('Failed to add todo:', error);
      setError(error.response?.data?.error || 'Failed to add todo');
    }
  };

  const toggleTodo = async (todoId, isCompleted) => {
    try {
      await api.put(`/api/todos/${todoId}`, { is_completed: !isCompleted });
      setTodos(todos.map(todo =>
        todo.id === todoId ? { ...todo, is_completed: !isCompleted } : todo
      ));
    } catch (error) {
      console.error('Failed to update todo:', error);
    }
  };

  const deleteTodo = async (todoId) => {
    const ok = window.confirm('Delete this todo?');
    if (!ok) return;
    try {
      await api.delete(`/api/todos/${todoId}`);
      setTodos(todos.filter(todo => todo.id !== todoId));
    } catch (error) {
      console.error('Failed to delete todo:', error);
      setError('Failed to delete todo');
    }
  };

  const startEdit = (todo) => {
    setEditingId(todo.id);
    setEditText(todo.title);
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditText('');
  };

  const saveEdit = async (todoId) => {
    if (!editText.trim()) {
      setError('Todo title cannot be empty');
      return;
    }
    try {
      const response = await api.put(`/api/todos/${todoId}`, { title: editText.trim() });
      const updated = response.data?.todo;
      if (updated) {
        setTodos(todos.map(todo => 
          todo.id === todoId ? updated : todo
        ));
        setEditingId(null);
        setEditText('');
        setError(null);
      }
    } catch (error) {
      console.error('Failed to update todo:', error);
      setError(error.response?.data?.error || 'Failed to update todo');
    }
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>My Todo List</h1>
      {loading && <div style={styles.status}>Loading todos...</div>}
      {error && (
        <div style={styles.error}>
          <span style={styles.errorIcon}>⚠️</span>
          {error}
        </div>
      )}
      
      <div style={styles.formContainer}>
        <form onSubmit={handleSubmit} style={styles.form}>
          <input
            type="text"
            value={newTodo}
            onChange={(e) => setNewTodo(e.target.value)}
            placeholder="Add a new todo..."
            style={styles.input}
            disabled={loading}
          />
          <button type="submit" style={styles.addButton} disabled={loading}>
            ➕ Add
          </button>
        </form>
      </div>

      <div style={styles.todoList}>
        {todos.length === 0 && !loading ? (
          <div style={styles.emptyState}>
            <span style={styles.emptyIcon}>📝</span>
            <p>No todos yet. Add one above!</p>
          </div>
        ) : (
          todos.map((todo) => (
            <div key={todo.id} style={styles.todoItem}>
              <input
                type="checkbox"
                checked={todo.is_completed}
                onChange={() => toggleTodo(todo.id, todo.is_completed)}
                style={styles.checkbox}
                disabled={editingId === todo.id}
              />
              {editingId === todo.id ? (
                <>
                  <input
                    type="text"
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        saveEdit(todo.id);
                      } else if (e.key === 'Escape') {
                        cancelEdit();
                      }
                    }}
                    style={styles.editInput}
                    autoFocus
                  />
                  <div style={styles.editActions}>
                    <button
                      onClick={() => saveEdit(todo.id)}
                      style={styles.saveButton}
                      title="Save (Enter)"
                    >
                      ✓
                    </button>
                    <button
                      onClick={cancelEdit}
                      style={styles.cancelButton}
                      title="Cancel (Esc)"
                    >
                      ✕
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <span 
                    style={{
                      ...styles.todoText,
                      textDecoration: todo.is_completed ? 'line-through' : 'none',
                      opacity: todo.is_completed ? 0.6 : 1,
                      cursor: todo.is_completed ? 'default' : 'pointer'
                    }}
                    onClick={() => !todo.is_completed && startEdit(todo)}
                    title={todo.is_completed ? '' : 'Click to edit'}
                  >
                    {todo.title}
                  </span>
                  <div style={styles.actions}>
                    {!todo.is_completed && (
                      <button
                        onClick={() => startEdit(todo)}
                        style={styles.editButton}
                        title="Edit"
                      >
                        ✏️
                      </button>
                    )}
                    <button
                      onClick={() => deleteTodo(todo.id)}
                      style={styles.deleteButton}
                      title="Delete"
                    >
                      🗑️
                    </button>
                  </div>
                </>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: '700px',
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
    color: 'rgba(255, 255, 255, 0.9)',
    margin: '1rem 0',
    fontSize: '1.1rem',
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
  formContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    padding: '1.5rem',
    marginBottom: '2rem',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    border: '1px solid rgba(102, 126, 234, 0.3)',
  },
  form: {
    display: 'flex',
    gap: '1rem',
  },
  input: {
    flex: 1,
    padding: '0.875rem 1rem',
    fontSize: '1rem',
    border: '2px solid rgba(102, 126, 234, 0.3)',
    borderRadius: '12px',
    transition: 'all 0.3s ease',
    backgroundColor: 'white',
  },
  addButton: {
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
    whiteSpace: 'nowrap',
  },
  todoList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
  },
  emptyState: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    padding: '3rem',
    textAlign: 'center',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
    border: '1px solid rgba(102, 126, 234, 0.3)',
  },
  emptyIcon: {
    fontSize: '3rem',
    display: 'block',
    marginBottom: '1rem',
  },
  todoItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
    padding: '1.25rem',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(10px)',
    borderRadius: '15px',
    boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)',
    border: '1px solid rgba(102, 126, 234, 0.3)',
    transition: 'all 0.3s ease',
  },
  checkbox: {
    width: '20px',
    height: '20px',
    cursor: 'pointer',
    accentColor: '#667eea',
  },
  todoText: {
    flex: 1,
    cursor: 'pointer',
    padding: '0.5rem',
    borderRadius: '8px',
    transition: 'all 0.2s',
    fontSize: '1rem',
    fontWeight: '500',
    color: '#333',
  },
  editInput: {
    flex: 1,
    padding: '0.75rem',
    fontSize: '1rem',
    border: '2px solid #667eea',
    borderRadius: '10px',
    outline: 'none',
    backgroundColor: 'white',
  },
  editActions: {
    display: 'flex',
    gap: '0.5rem',
  },
  saveButton: {
    padding: '0.5rem 1rem',
    background: 'linear-gradient(135deg, #4caf50 0%, #45a049 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '1.2rem',
    lineHeight: 1,
    boxShadow: '0 2px 8px rgba(76, 175, 80, 0.3)',
  },
  cancelButton: {
    padding: '0.5rem 1rem',
    backgroundColor: '#9e9e9e',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '1.2rem',
    lineHeight: 1,
  },
  actions: {
    display: 'flex',
    gap: '0.5rem',
  },
  editButton: {
    padding: '0.5rem 1rem',
    background: 'linear-gradient(135deg, #ff9800 0%, #f57c00 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '1rem',
    boxShadow: '0 2px 8px rgba(255, 152, 0, 0.3)',
  },
  deleteButton: {
    padding: '0.5rem 1rem',
    background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '1rem',
    boxShadow: '0 2px 8px rgba(245, 87, 108, 0.3)',
  },
};

export default TodoList;
