import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import api from '../services/api';
import { useAuth } from './AuthContext';

const CallContext = createContext(null);

export function CallProvider({ children }) {
  const { user } = useAuth();
  const [incomingCall, setIncomingCall] = useState(null);
  const [callStatus, setCallStatus] = useState(null);
  const [currentCall, setCurrentCall] = useState(null);
  const callCheckInterval = useRef(null);

  useEffect(() => {
    if (!user || !user.id) return;

    const checkIncomingCalls = async () => {
      try {
        const res = await api.get('/api/calls/pending');
        if (res.data && res.data.pending !== false && res.data.status === 'pending') {
          if (res.data.receiver_id === user.id) {
            if (!incomingCall || incomingCall.id !== res.data.id) {
              setIncomingCall(res.data);
            }
          }
        } else if (incomingCall) {
          setIncomingCall(null);
        }
      } catch (e) {
        // Silently ignore 404 errors (endpoint might not exist or feature disabled)
        // Only log unexpected errors
        if (e.response && e.response.status !== 404) {
          console.warn('Error checking for calls:', e.response?.status, e.response?.data);
        }
      }
    };

    checkIncomingCalls();
    callCheckInterval.current = setInterval(checkIncomingCalls, 5000); // Reduced frequency to every 5 seconds
    return () => {
      if (callCheckInterval.current) clearInterval(callCheckInterval.current);
    };
  }, [user, incomingCall]);

  const acceptCall = async () => {
    if (!incomingCall) return;
    try {
      await api.post(`/api/calls/${incomingCall.id}/accept`);
      setCurrentCall(incomingCall);
      setIncomingCall(null);
      setCallStatus('active');
    } catch (e) {
      // ignore
    }
  };

  const rejectCall = async () => {
    if (!incomingCall) return;
    try {
      await api.post(`/api/calls/${incomingCall.id}/reject`);
      setIncomingCall(null);
    } catch (e) {
      // ignore
    }
  };

  const endCall = async () => {
    try {
      if (currentCall?.id) {
        await api.post(`/api/calls/${currentCall.id}/end`);
      }
    } catch (e) {
      // ignore
    } finally {
      setCallStatus(null);
      setCurrentCall(null);
    }
  };

  return (
    <CallContext.Provider value={{ incomingCall, currentCall, callStatus, setCallStatus, setCurrentCall, acceptCall, rejectCall, endCall }}>
      {/* Global Incoming Call Modal */}
      {incomingCall && incomingCall.receiver_id === user?.id && (
        <div style={styles.callModal}>
          <div style={styles.callModalContent}>
            <h2 style={styles.callTitle}>📞 Incoming Call</h2>
            <p style={styles.callFrom}>{incomingCall.caller_username || 'Someone'} is calling you</p>
            <div style={styles.callButtons}>
              <button onClick={acceptCall} style={styles.acceptButton}>✓ Accept</button>
              <button onClick={rejectCall} style={styles.rejectButton}>✗ Reject</button>
            </div>
          </div>
        </div>
      )}

      {children}
    </CallContext.Provider>
  );
}

export const useCall = () => useContext(CallContext);

const styles = {
  callModal: { position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10000 },
  callModalContent: { background: 'white', padding: '2rem', borderRadius: 12, textAlign: 'center', boxShadow: '0 4px 20px rgba(0,0,0,0.3)' },
  callTitle: { margin: '0 0 1rem 0', fontSize: '1.5rem', color: '#333' },
  callFrom: { margin: '0 0 1.5rem 0', fontSize: '1.1rem', color: '#666' },
  callButtons: { display: 'flex', gap: '1rem', justifyContent: 'center' },
  acceptButton: { padding: '0.75rem 2rem', background: '#4caf50', color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: '1rem', fontWeight: '600' },
  rejectButton: { padding: '0.75rem 2rem', background: '#f44336', color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: '1rem', fontWeight: '600' }
};
