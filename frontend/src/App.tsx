import React, { useState, useEffect, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@material-ui/core/styles';
import { CssBaseline, Snackbar, Alert } from '@material-ui/core';
import { io, Socket } from 'socket.io-client';

// Components
import Layout from './components/Layout/Layout';
import Dashboard from './pages/Dashboard/Dashboard';
import GDPCalculator from './pages/GDPCalculator/GDPCalculator';
import Forecasting from './pages/Forecasting/Forecasting';
import DataIntegration from './pages/DataIntegration/DataIntegration';
import Visualization from './pages/Visualization/Visualization';
import AIChat from './pages/AIChat/AIChat';
import Login from './pages/Auth/Login';
import Register from './pages/Auth/Register';
import Profile from './pages/Profile/Profile';
import Admin from './pages/Admin/Admin';
import Loading from './components/Common/Loading';

// Contexts
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ThemeProvider as CustomThemeProvider } from './contexts/ThemeContext';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { NotificationProvider, useNotification } from './contexts/NotificationContext';

// Services
import { initializeMonitoring } from './services/monitoring';

// Styles
import './App.css';

// Create Material-UI theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
      light: '#ff5983',
      dark: '#9a0036',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 8,
  },
  spacing: 8,
});

// Protected Route Component
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();
  
  if (loading) {
    return <Loading />;
  }
  
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" />;
};

// Main App Component
const App: React.FC = () => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const { showNotification } = useNotification();

  useEffect(() => {
    // Initialize WebSocket connection
    const newSocket = io(process.env.REACT_APP_WS_URL || 'ws://localhost:8000');
    setSocket(newSocket);

    // Initialize monitoring
    initializeMonitoring();

    // WebSocket event handlers
    newSocket.on('connect', () => {
      console.log('Connected to server');
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from server');
    });

    newSocket.on('gdp_update', (data: any) => {
      showNotification('GDP data updated', 'success');
    });

    newSocket.on('forecast_complete', (data: any) => {
      showNotification('Forecast calculation completed', 'success');
    });

    newSocket.on('error', (error: any) => {
      showNotification(`Error: ${error.message}`, 'error');
    });

    return () => {
      newSocket.close();
    };
  }, [showNotification]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AuthProvider>
        <CustomThemeProvider>
          <NotificationProvider>
            <WebSocketProvider socket={socket}>
              <Router>
                <div className="App">
                  <Suspense fallback={<Loading />}>
                    <Routes>
                      {/* Public Routes */}
                      <Route path="/login" element={<Login />} />
                      <Route path="/register" element={<Register />} />
                      
                      {/* Protected Routes */}
                      <Route path="/" element={
                        <ProtectedRoute>
                          <Layout>
                            <Dashboard />
                          </Layout>
                        </ProtectedRoute>
                      } />
                      
                      <Route path="/gdp-calculator" element={
                        <ProtectedRoute>
                          <Layout>
                            <GDPCalculator />
                          </Layout>
                        </ProtectedRoute>
                      } />
                      
                      <Route path="/forecasting" element={
                        <ProtectedRoute>
                          <Layout>
                            <Forecasting />
                          </Layout>
                        </ProtectedRoute>
                      } />
                      
                      <Route path="/data-integration" element={
                        <ProtectedRoute>
                          <Layout>
                            <DataIntegration />
                          </Layout>
                        </ProtectedRoute>
                      } />
                      
                      <Route path="/visualization" element={
                        <ProtectedRoute>
                          <Layout>
                            <Visualization />
                          </Layout>
                        </ProtectedRoute>
                      } />
                      
                      <Route path="/ai-chat" element={
                        <ProtectedRoute>
                          <Layout>
                            <AIChat />
                          </Layout>
                        </ProtectedRoute>
                      } />
                      
                      <Route path="/profile" element={
                        <ProtectedRoute>
                          <Layout>
                            <Profile />
                          </Layout>
                        </ProtectedRoute>
                      } />
                      
                      <Route path="/admin" element={
                        <ProtectedRoute>
                          <Layout>
                            <Admin />
                          </Layout>
                        </ProtectedRoute>
                      } />
                      
                      {/* Catch all route */}
                      <Route path="*" element={<Navigate to="/" />} />
                    </Routes>
                  </Suspense>
                </div>
              </Router>
            </WebSocketProvider>
          </NotificationProvider>
        </CustomThemeProvider>
      </AuthProvider>
    </ThemeProvider>
  );
};

export default App;