import React, { useState, useEffect, useRef } from 'react';
import {
  Fab,
  Dialog,
  DialogTitle,
  DialogContent,
  Box,
  Typography,
  Button,
  Paper,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Chip,
  LinearProgress,
  Snackbar,
  Alert,
} from '@material-ui/core';
import {
  Mic,
  MicOff,
  VolumeUp,
  Settings,
  Send,
  Clear,
  QuestionAnswer,
} from '@material-ui/icons';
import { makeStyles } from '@material-ui/core/styles';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';

// Services
import { aiService } from '../../services/aiService';
import { useNotification } from '../../contexts/NotificationContext';

interface VoiceCommand {
  command: string;
  response: string;
  timestamp: Date;
  confidence: number;
}

interface VoiceInterfaceProps {
  onQueryResult?: (result: any) => void;
}

const useStyles = makeStyles((theme) => ({
  fab: {
    position: 'fixed',
    bottom: theme.spacing(8),
    right: theme.spacing(2),
    zIndex: 1000,
  },
  dialog: {
    '& .MuiDialog-paper': {
      width: '80%',
      maxWidth: 600,
      height: '70%',
    },
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
  },
  transcriptArea: {
    flex: 1,
    padding: theme.spacing(2),
    backgroundColor: theme.palette.grey[50],
    borderRadius: theme.spacing(1),
    marginBottom: theme.spacing(2),
    minHeight: 200,
    overflow: 'auto',
  },
  commandsList: {
    flex: 1,
    overflow: 'auto',
    maxHeight: 300,
  },
  micButton: {
    width: 80,
    height: 80,
    margin: theme.spacing(2, 'auto'),
    display: 'block',
  },
  activeListening: {
    backgroundColor: theme.palette.secondary.main,
    '&:hover': {
      backgroundColor: theme.palette.secondary.dark,
    },
  },
  controls: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: theme.spacing(1),
    borderTop: `1px solid ${theme.palette.divider}`,
  },
  waveform: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    height: 60,
    margin: theme.spacing(1, 0),
  },
  wave: {
    width: 4,
    margin: '0 1px',
    backgroundColor: theme.palette.primary.main,
    borderRadius: 2,
    animation: '$wave 1s ease-in-out infinite',
  },
  '@keyframes wave': {
    '0%, 100%': { height: 10 },
    '50%': { height: 30 },
  },
}));

const VoiceInterface: React.FC<VoiceInterfaceProps> = ({ onQueryResult }) => {
  const classes = useStyles();
  const { showNotification } = useNotification();
  
  // Speech recognition
  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition,
  } = useSpeechRecognition();

  // State
  const [open, setOpen] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentResponse, setCurrentResponse] = useState('');
  const [commandHistory, setCommandHistory] = useState<VoiceCommand[]>([]);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [volume, setVolume] = useState(1);
  const [language, setLanguage] = useState('en-US');
  
  // Refs
  const speechSynthesisRef = useRef<SpeechSynthesisUtterance | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // Check browser support
  useEffect(() => {
    if (!browserSupportsSpeechRecognition) {
      showNotification(
        'Speech recognition is not supported in this browser',
        'warning'
      );
    }
  }, [browserSupportsSpeechRecognition, showNotification]);

  // Handle transcript changes
  useEffect(() => {
    if (transcript && !listening && transcript.length > 10) {
      handleVoiceCommand(transcript);
    }
  }, [transcript, listening]);

  // Voice command processing
  const handleVoiceCommand = async (command: string) => {
    if (!command.trim()) return;

    setIsProcessing(true);
    
    try {
      // Process the voice command through AI service
      const response = await aiService.processNaturalLanguageQuery({
        query: command,
        include_visualization: true,
        context: { source: 'voice_interface' },
      });

      // Add to command history
      const newCommand: VoiceCommand = {
        command: command.trim(),
        response: response.answer,
        timestamp: new Date(),
        confidence: response.confidence,
      };

      setCommandHistory(prev => [newCommand, ...prev.slice(0, 9)]); // Keep last 10
      setCurrentResponse(response.answer);

      // Speak the response
      if (response.answer) {
        await speakText(response.answer);
      }

      // Trigger callback with result
      onQueryResult?.(response);

      // Clear transcript
      resetTranscript();

    } catch (error) {
      console.error('Error processing voice command:', error);
      const errorMessage = 'Sorry, I encountered an error processing your request.';
      setCurrentResponse(errorMessage);
      await speakText(errorMessage);
      
      showNotification('Error processing voice command', 'error');
    } finally {
      setIsProcessing(false);
    }
  };

  // Text-to-speech
  const speakText = async (text: string): Promise<void> => {
    return new Promise((resolve) => {
      if ('speechSynthesis' in window) {
        // Cancel any ongoing speech
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = language;
        utterance.volume = volume;
        utterance.rate = 0.9;
        utterance.pitch = 1;

        utterance.onstart = () => setIsSpeaking(true);
        utterance.onend = () => {
          setIsSpeaking(false);
          resolve();
        };
        utterance.onerror = () => {
          setIsSpeaking(false);
          resolve();
        };

        speechSynthesisRef.current = utterance;
        window.speechSynthesis.speak(utterance);
      } else {
        resolve();
      }
    });
  };

  // Start/stop listening
  const toggleListening = () => {
    if (listening) {
      SpeechRecognition.stopListening();
    } else {
      resetTranscript();
      SpeechRecognition.startListening({
        continuous: false,
        language: language,
      });
    }
  };

  // Stop speaking
  const stopSpeaking = () => {
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    setIsSpeaking(false);
  };

  // Clear history
  const clearHistory = () => {
    setCommandHistory([]);
    setCurrentResponse('');
    resetTranscript();
  };

  // Quick commands
  const quickCommands = [
    'Show GDP for United States',
    'Compare GDP between USA and China',
    'What is the GDP growth rate for India?',
    'Show global GDP trends',
    'Forecast GDP for Japan',
  ];

  const handleQuickCommand = (command: string) => {
    resetTranscript();
    handleVoiceCommand(command);
  };

  // Render waveform animation
  const renderWaveform = () => {
    if (!listening) return null;

    return (
      <Box className={classes.waveform}>
        {[...Array(8)].map((_, i) => (
          <div
            key={i}
            className={classes.wave}
            style={{
              animationDelay: `${i * 0.1}s`,
              height: Math.random() * 20 + 10,
            }}
          />
        ))}
      </Box>
    );
  };

  if (!browserSupportsSpeechRecognition) {
    return null;
  }

  return (
    <>
      {/* Floating Action Button */}
      <Fab
        color="primary"
        className={classes.fab}
        onClick={() => setOpen(true)}
      >
        <QuestionAnswer />
      </Fab>

      {/* Voice Interface Dialog */}
      <Dialog open={open} onClose={() => setOpen(false)} className={classes.dialog}>
        <DialogTitle>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            Voice Assistant
            <IconButton onClick={() => setOpen(false)}>
              <Clear />
            </IconButton>
          </Box>
        </DialogTitle>
        
        <DialogContent className={classes.content}>
          {/* Current Transcript */}
          <Paper className={classes.transcriptArea}>
            <Typography variant="h6" gutterBottom>
              {listening ? 'Listening...' : 'Say something about GDP data'}
            </Typography>
            
            {renderWaveform()}
            
            <Typography variant="body1" style={{ marginTop: 16 }}>
              {transcript || 'Your speech will appear here...'}
            </Typography>
            
            {isProcessing && (
              <Box mt={2}>
                <LinearProgress />
                <Typography variant="caption" color="textSecondary">
                  Processing your request...
                </Typography>
              </Box>
            )}
          </Paper>

          {/* Current Response */}
          {currentResponse && (
            <Paper style={{ padding: 16, marginBottom: 16 }}>
              <Box display="flex" justifyContent="space-between" alignItems="flex-start">
                <Typography variant="body1">
                  {currentResponse}
                </Typography>
                <IconButton
                  size="small"
                  onClick={() => speakText(currentResponse)}
                  disabled={isSpeaking}
                >
                  <VolumeUp />
                </IconButton>
              </Box>
            </Paper>
          )}

          {/* Quick Commands */}
          <Typography variant="subtitle2" gutterBottom>
            Quick Commands:
          </Typography>
          <Box mb={2}>
            {quickCommands.map((command, index) => (
              <Chip
                key={index}
                label={command}
                onClick={() => handleQuickCommand(command)}
                style={{ margin: 4 }}
                size="small"
              />
            ))}
          </Box>

          {/* Command History */}
          <Typography variant="subtitle2" gutterBottom>
            Recent Commands:
          </Typography>
          <List className={classes.commandsList}>
            {commandHistory.map((cmd, index) => (
              <ListItem key={index} divider>
                <ListItemText
                  primary={cmd.command}
                  secondary={
                    <Box>
                      <Typography variant="caption" color="textSecondary">
                        {cmd.timestamp.toLocaleTimeString()} • 
                        Confidence: {(cmd.confidence * 100).toFixed(0)}%
                      </Typography>
                      <Typography variant="body2" style={{ marginTop: 4 }}>
                        {cmd.response.substring(0, 100)}...
                      </Typography>
                    </Box>
                  }
                />
                <IconButton
                  edge="end"
                  onClick={() => speakText(cmd.response)}
                  disabled={isSpeaking}
                >
                  <VolumeUp />
                </IconButton>
              </ListItem>
            ))}
          </List>

          {/* Controls */}
          <Box className={classes.controls}>
            <Box>
              <Button
                variant="outlined"
                size="small"
                onClick={clearHistory}
                startIcon={<Clear />}
              >
                Clear
              </Button>
            </Box>

            <Fab
              color={listening ? "secondary" : "primary"}
              className={listening ? classes.activeListening : ''}
              onClick={toggleListening}
              disabled={isProcessing}
            >
              {listening ? <MicOff /> : <Mic />}
            </Fab>

            <Box>
              {isSpeaking && (
                <Button
                  variant="outlined"
                  size="small"
                  onClick={stopSpeaking}
                  startIcon={<MicOff />}
                >
                  Stop
                </Button>
              )}
            </Box>
          </Box>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default VoiceInterface;