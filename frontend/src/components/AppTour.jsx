import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

function AppTour() {
  const { user } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [showTour, setShowTour] = useState(false);

  // Check if user has completed the tour
  useEffect(() => {
    if (!user) {
      setShowTour(false);
      setIsRunning(false);
      return;
    }

    const tourCompleted = localStorage.getItem(`tour_completed_${user.id}`);
    const shouldStartTour = localStorage.getItem(`start_tour_${user.id}`) === 'true';
    
    if (shouldStartTour) {
      localStorage.removeItem(`start_tour_${user.id}`);
      setShowTour(true);
      setIsRunning(true);
      setCurrentStep(0);
    } else if (!tourCompleted && user) {
      // Show tour after a short delay for new users
      const timer = setTimeout(() => {
        setShowTour(true);
        setIsRunning(true);
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [user]);

  // Tour steps configuration
  const tourSteps = [
    {
      id: 'sidebar',
      title: '🎯 Navigation Sidebar',
      content: 'This is your main navigation. Click any option to explore different features like Videos, Messages, Paint, and more!',
      target: '[data-tour="sidebar"]',
      position: 'right',
      route: '/members',
    },
    {
      id: 'members',
      title: '👥 Members',
      content: 'View all community members here. You can see who\'s online and connect with other users.',
      target: '[data-tour="members"]',
      position: 'bottom',
      route: '/members',
    },
    {
      id: 'videos',
      title: '🎥 Video Sharing',
      content: 'Upload and share MP4 videos with the community. Click here to browse and upload videos!',
      target: '[data-tour="videos"]',
      position: 'bottom',
      route: '/videos',
    },
    {
      id: 'messages',
      title: '💬 Messaging',
      content: 'Send messages and files to other users. Real-time communication with desktop notifications!',
      target: '[data-tour="messages"]',
      position: 'bottom',
      route: '/messages',
    },
    {
      id: 'paint',
      title: '🎨 Paint & Draw',
      content: 'Create beautiful artwork with our collaborative drawing tool. Your paintings are saved automatically!',
      target: '[data-tour="paint"]',
      position: 'bottom',
      route: '/paint',
    },
    {
      id: 'todos',
      title: '✅ Todo Lists',
      content: 'Manage your tasks and stay organized with personal todo lists.',
      target: '[data-tour="todos"]',
      position: 'bottom',
      route: '/todos',
    },
    {
      id: 'ai-chat',
      title: '🤖 AI Chat',
      content: 'Chat with our AI assistant powered by OpenAI. Get help, ask questions, and have conversations!',
      target: '[data-tour="ai-chat"]',
      position: 'bottom',
      route: '/ai-chat',
    },
    {
      id: 'docs',
      title: '📄 My Docs & Images',
      content: 'Generate AI-powered documents and create JPEG images using DALL-E. All your content is saved automatically!',
      target: '[data-tour="docs"]',
      position: 'bottom',
      route: '/docs',
    },
    {
      id: 'blog',
      title: '📝 Blog',
      content: 'Read and create blog posts with images. Share your thoughts with the community!',
      target: '[data-tour="blog"]',
      position: 'bottom',
      route: '/blog',
    },
  ];

  // Admin-only steps
  const adminSteps = [
    {
      id: 'admin',
      title: '⚙️ Admin Dashboard',
      content: 'Manage users, view system stats, and access admin-only features.',
      target: '[data-tour="admin"]',
      position: 'bottom',
      route: '/admin',
    },
    {
      id: 'ai-training',
      title: '🎓 AI Training',
      content: 'Train the AI with custom prompts and responses. This helps improve AI responses for all users.',
      target: '[data-tour="ai-training"]',
      position: 'bottom',
      route: '/admin/ai-training',
    },
  ];

  const allSteps = user?.is_admin ? [...tourSteps, ...adminSteps] : tourSteps;

  const [targetFound, setTargetFound] = useState(false);

  useEffect(() => {
    if (!isRunning || !showTour) return;

    const step = allSteps[currentStep];
    if (!step) {
      finishTour();
      return;
    }

    // Navigate to the step's route if needed
    if (location.pathname !== step.route) {
      navigate(step.route);
      setTargetFound(false);
    }

    // Wait for navigation and DOM update, then highlight the target
    const timer = setTimeout(() => {
      const element = document.querySelector(step.target);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        setTargetFound(true);
      } else {
        // If target not found, skip to next step after delay
        setTimeout(() => {
          if (currentStep < allSteps.length - 1) {
            setCurrentStep(currentStep + 1);
          } else {
            finishTour();
          }
        }, 500);
      }
    }, 800);

    return () => clearTimeout(timer);
  }, [currentStep, isRunning, showTour, location.pathname, navigate, allSteps]);

  const nextStep = () => {
    if (currentStep < allSteps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      finishTour();
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const skipTour = () => {
    finishTour();
  };

  const finishTour = () => {
    setIsRunning(false);
    setShowTour(false);
    if (user) {
      localStorage.setItem(`tour_completed_${user.id}`, 'true');
    }
  };

  const startTour = () => {
    setCurrentStep(0);
    setIsRunning(true);
    setShowTour(true);
  };

  if (!user || !showTour || !isRunning) {
    return null;
  }

  const step = allSteps[currentStep];
  if (!step) {
    finishTour();
    return null;
  }

  // Wait a bit for DOM to update after navigation
  const targetElement = document.querySelector(step.target);
  if (!targetElement) {
    // If target not found, return null (will be handled by useEffect)
    return null;
  }

  const rect = targetElement.getBoundingClientRect();
  const scrollY = window.scrollY || window.pageYOffset;
  const scrollX = window.scrollX || window.pageXOffset;

  // Get viewport dimensions
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  const tooltipWidth = Math.min(400, viewportWidth - 40); // Max 400px, but ensure 20px margin on each side
  const tooltipHeight = 250; // Approximate tooltip height

  // Calculate position based on step.position, but ensure it fits on screen
  let top = rect.top + scrollY;
  let left = rect.left + scrollX;
  let tooltipPosition = 'bottom';
  let finalTop = 0;
  let finalLeft = 0;

  if (step.position === 'right') {
    // Try right side first
    left = rect.right + scrollX + 20;
    top = rect.top + scrollY + rect.height / 2;
    tooltipPosition = 'left';
    
    // Check if tooltip fits on right side
    if (left + tooltipWidth > viewportWidth + scrollX) {
      // Doesn't fit on right, try left side
      left = rect.left + scrollX - tooltipWidth - 20;
      tooltipPosition = 'right';
    }
    
    // Ensure tooltip doesn't go off left edge
    if (left < scrollX + 20) {
      left = scrollX + 20;
    }
    
    // Center vertically if possible
    finalTop = top - tooltipHeight / 2;
    finalLeft = left;
    
    // Ensure tooltip doesn't go off top or bottom
    if (finalTop < scrollY + 20) {
      finalTop = scrollY + 20;
    } else if (finalTop + tooltipHeight > scrollY + viewportHeight - 20) {
      finalTop = scrollY + viewportHeight - tooltipHeight - 20;
    }
  } else if (step.position === 'left') {
    // Try left side first
    left = rect.left + scrollX - tooltipWidth - 20;
    top = rect.top + scrollY + rect.height / 2;
    tooltipPosition = 'right';
    
    // Check if tooltip fits on left side
    if (left < scrollX + 20) {
      // Doesn't fit on left, try right side
      left = rect.right + scrollX + 20;
      tooltipPosition = 'left';
    }
    
    // Ensure tooltip doesn't go off right edge
    if (left + tooltipWidth > viewportWidth + scrollX) {
      left = viewportWidth + scrollX - tooltipWidth - 20;
    }
    
    // Center vertically if possible
    finalTop = top - tooltipHeight / 2;
    finalLeft = left;
    
    // Ensure tooltip doesn't go off top or bottom
    if (finalTop < scrollY + 20) {
      finalTop = scrollY + 20;
    } else if (finalTop + tooltipHeight > scrollY + viewportHeight - 20) {
      finalTop = scrollY + viewportHeight - tooltipHeight - 20;
    }
  } else if (step.position === 'top') {
    // Try top side
    top = rect.top + scrollY - tooltipHeight - 20;
    left = rect.left + scrollX + rect.width / 2;
    tooltipPosition = 'bottom';
    
    // Center horizontally
    finalLeft = left - tooltipWidth / 2;
    finalTop = top;
    
    // Check if tooltip fits on top
    if (finalTop < scrollY + 20) {
      // Doesn't fit on top, try bottom
      finalTop = rect.bottom + scrollY + 20;
      tooltipPosition = 'top';
    }
    
    // Ensure tooltip doesn't go off left or right edges
    if (finalLeft < scrollX + 20) {
      finalLeft = scrollX + 20;
    } else if (finalLeft + tooltipWidth > viewportWidth + scrollX - 20) {
      finalLeft = viewportWidth + scrollX - tooltipWidth - 20;
    }
    
    // Ensure tooltip doesn't go off bottom
    if (finalTop + tooltipHeight > scrollY + viewportHeight - 20) {
      finalTop = scrollY + viewportHeight - tooltipHeight - 20;
    }
  } else {
    // Default: bottom position
    top = rect.bottom + scrollY + 20;
    left = rect.left + scrollX + rect.width / 2;
    tooltipPosition = 'top';
    
    // Center horizontally
    finalLeft = left - tooltipWidth / 2;
    finalTop = top;
    
    // Check if tooltip fits on bottom
    if (finalTop + tooltipHeight > scrollY + viewportHeight - 20) {
      // Doesn't fit on bottom, try top
      finalTop = rect.top + scrollY - tooltipHeight - 20;
      tooltipPosition = 'bottom';
    }
    
    // Ensure tooltip doesn't go off left or right edges
    if (finalLeft < scrollX + 20) {
      finalLeft = scrollX + 20;
    } else if (finalLeft + tooltipWidth > viewportWidth + scrollX - 20) {
      finalLeft = viewportWidth + scrollX - tooltipWidth - 20;
    }
    
    // Ensure tooltip doesn't go off top
    if (finalTop < scrollY + 20) {
      finalTop = scrollY + 20;
    }
  }

  return (
    <>
      {/* Overlay */}
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          zIndex: 9998,
          pointerEvents: 'none',
        }}
      />

      {/* Highlight box */}
      <div
        style={{
          position: 'absolute',
          top: rect.top + scrollY - 4,
          left: rect.left + scrollX - 4,
          width: rect.width + 8,
          height: rect.height + 8,
          border: '3px solid #667eea',
          borderRadius: '8px',
          boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.7), 0 0 20px rgba(102, 126, 234, 0.6)',
          zIndex: 9999,
          pointerEvents: 'none',
          transition: 'all 0.3s ease',
        }}
      />

      {/* Tooltip */}
      <div
        style={{
          position: 'fixed',
          top: `${finalTop - scrollY}px`,
          left: `${finalLeft - scrollX}px`,
          width: `${tooltipWidth}px`,
          maxWidth: 'calc(100vw - 40px)',
          maxHeight: 'calc(100vh - 40px)',
          zIndex: 10000,
          pointerEvents: 'auto',
        }}
      >
        <div style={styles.tooltip}>
          <div style={styles.tooltipHeader}>
            <h3 style={styles.tooltipTitle}>{step.title}</h3>
            <button onClick={skipTour} style={styles.closeButton} title="Skip tour">
              ✕
            </button>
          </div>
          <p style={styles.tooltipContent}>{step.content}</p>
          <div style={styles.tooltipFooter}>
            <div style={styles.progress}>
              Step {currentStep + 1} of {allSteps.length}
            </div>
            <div style={styles.buttons}>
              {currentStep > 0 && (
                <button onClick={prevStep} style={styles.prevButton}>
                  ← Previous
                </button>
              )}
              <button onClick={nextStep} style={styles.nextButton}>
                {currentStep === allSteps.length - 1 ? 'Finish' : 'Next →'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

// Export function to start tour manually
export function useAppTour() {
  const { user } = useAuth();
  
  const startTour = () => {
    if (user) {
      localStorage.setItem(`start_tour_${user.id}`, 'true');
      window.location.reload();
    }
  };

  const hasCompletedTour = () => {
    if (!user) return true;
    return localStorage.getItem(`tour_completed_${user.id}`) === 'true';
  };

  return { startTour, hasCompletedTour };
}

const styles = {
  tooltip: {
    background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 250, 252, 0.98) 100%)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    padding: '1.5rem',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(102, 126, 234, 0.2)',
    border: '2px solid rgba(102, 126, 234, 0.4)',
    maxHeight: 'calc(100vh - 40px)',
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
  },
  tooltipHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: '1rem',
    flexShrink: 0,
  },
  tooltipTitle: {
    margin: 0,
    fontSize: 'clamp(1.2rem, 4vw, 1.5rem)',
    fontWeight: '700',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
    flex: 1,
    lineHeight: '1.3',
  },
  closeButton: {
    background: 'transparent',
    border: 'none',
    fontSize: '1.5rem',
    color: '#999',
    cursor: 'pointer',
    padding: '0',
    width: '30px',
    height: '30px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: '50%',
    transition: 'all 0.2s ease',
    flexShrink: 0,
  },
  closeButtonHover: {
    background: 'rgba(102, 126, 234, 0.1)',
    color: '#667eea',
  },
  tooltipContent: {
    margin: 0,
    marginBottom: '1.5rem',
    fontSize: 'clamp(0.9rem, 3vw, 1rem)',
    lineHeight: '1.6',
    color: '#333',
    flex: 1,
    overflowY: 'auto',
  },
  tooltipFooter: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: '1rem',
    flexWrap: 'wrap',
    flexShrink: 0,
    marginTop: 'auto',
  },
  progress: {
    fontSize: 'clamp(0.8rem, 2.5vw, 0.9rem)',
    color: '#666',
    fontWeight: '500',
  },
  buttons: {
    display: 'flex',
    gap: '0.75rem',
    flexWrap: 'wrap',
  },
  prevButton: {
    padding: 'clamp(0.6rem, 2vw, 0.75rem) clamp(1rem, 3vw, 1.5rem)',
    background: 'rgba(102, 126, 234, 0.1)',
    color: '#667eea',
    border: '2px solid rgba(102, 126, 234, 0.3)',
    borderRadius: '10px',
    fontSize: 'clamp(0.8rem, 2.5vw, 0.9rem)',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    whiteSpace: 'nowrap',
  },
  prevButtonHover: {
    background: 'rgba(102, 126, 234, 0.2)',
    borderColor: 'rgba(102, 126, 234, 0.5)',
    transform: 'translateY(-1px)',
  },
  nextButton: {
    padding: 'clamp(0.6rem, 2vw, 0.75rem) clamp(1rem, 3vw, 1.5rem)',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    color: 'white',
    border: 'none',
    borderRadius: '10px',
    fontSize: 'clamp(0.8rem, 2.5vw, 0.9rem)',
    fontWeight: '600',
    cursor: 'pointer',
    boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
    transition: 'all 0.3s ease',
    whiteSpace: 'nowrap',
  },
  nextButtonHover: {
    transform: 'translateY(-1px)',
    boxShadow: '0 6px 20px rgba(102, 126, 234, 0.5)',
  },
};

export default AppTour;

