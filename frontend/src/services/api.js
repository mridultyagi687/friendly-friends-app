import axios from 'axios';

// When using Vite proxy, use relative URLs (Vite will proxy /api to backend)
// If VITE_API_URL is set, use that instead (for production builds)
const DEFAULT_API_ORIGIN = 'https://jerilyn-nonobligated-punningly.ngrok-free.dev';
const BASE_URL =
  import.meta.env.VITE_API_URL ||
  import.meta.env.REACT_APP_API_URL ||
  (typeof window !== 'undefined' && window.location.origin
    ? window.location.origin
    : DEFAULT_API_ORIGIN);

const api = axios.create({
  baseURL: BASE_URL,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to log requests and ensure correct URL
api.interceptors.request.use(
  (config) => {
    // Ensure we're using relative URLs for proxy (should be empty BASE_URL)
    // Fix any localhost:5000 or localhost:5001 URLs to use proxy instead
    if (config.baseURL && (config.baseURL.includes('localhost:5000') || config.baseURL.includes('localhost:5001'))) {
      // Force relative URL for proxy
      config.baseURL = '';
    }
    
    // Remove Content-Type header for FormData - axios will set it automatically with boundary
    if (config.data instanceof FormData) {
      delete config.headers['Content-Type'];
    }
    
    // Only log in development (not in production)
    if (import.meta.env.DEV) {
      console.log('Making request to:', config.url);
      if (typeof window !== 'undefined') {
        console.log('Full URL will be:', window.location.origin + config.baseURL + config.url);
      }
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add error interceptor for better debugging
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Only log detailed errors in development
    if (import.meta.env.DEV) {
      if (error.response) {
        // Only log 500 errors or other serious errors, skip 401/404 in production-like scenarios
        if (error.response.status >= 500) {
          console.error('API Error Details:', {
            message: error.message,
            code: error.code,
            response: error.response?.data || '(empty response)',
            status: error.response?.status,
            statusText: error.response?.statusText,
          });
        }
      } else {
        // Network error - no response from server
        if (error.code === 'ERR_NETWORK' || error.message === 'Network Error') {
          console.error('Network Error:', {
            message: error.message,
            code: error.code,
            backendURL: BASE_URL || 'using proxy',
          });
        }
      }
    }
    return Promise.reject(error);
  }
);

export default api;
