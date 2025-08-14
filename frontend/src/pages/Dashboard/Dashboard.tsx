import React, { useState, useEffect, useCallback } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  Menu,
  MenuItem,
  Chip,
  LinearProgress,
  Fab,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Select,
  FormControl,
  InputLabel,
} from '@material-ui/core';
import {
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  Add as AddIcon,
  TrendingUp,
  Assessment,
  Public,
  Timeline,
} from '@material-ui/icons';
import { makeStyles } from '@material-ui/core/styles';

// Custom Components
import GDPOverviewChart from '../../components/Charts/GDPOverviewChart';
import CountryComparisonChart from '../../components/Charts/CountryComparisonChart';
import ForecastChart from '../../components/Charts/ForecastChart';
import WorldMap from '../../components/Maps/WorldMap';
import KPICard from '../../components/Cards/KPICard';
import RecentActivity from '../../components/Activity/RecentActivity';
import AIInsights from '../../components/AI/AIInsights';
import VoiceInterface from '../../components/Voice/VoiceInterface';

// Hooks and Services
import { useAuth } from '../../contexts/AuthContext';
import { useWebSocket } from '../../contexts/WebSocketContext';
import { useNotification } from '../../contexts/NotificationContext';
import { gdpService } from '../../services/gdpService';
import { dashboardService } from '../../services/dashboardService';

// Types
interface DashboardData {
  overview: {
    totalCountries: number;
    latestGDP: number;
    growthRate: number;
    lastUpdate: string;
  };
  recentCalculations: any[];
  forecasts: any[];
  alerts: any[];
  topCountries: any[];
}

interface WidgetConfig {
  id: string;
  type: string;
  title: string;
  position: { x: number; y: number };
  size: { width: number; height: number };
  config: any;
}

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1,
    padding: theme.spacing(3),
  },
  paper: {
    padding: theme.spacing(2),
    textAlign: 'center',
    color: theme.palette.text.secondary,
    height: '100%',
  },
  kpiCard: {
    background: 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)',
    borderRadius: 3,
    border: 0,
    color: 'white',
    height: 120,
    padding: '0 30px',
    boxShadow: '0 3px 5px 2px rgba(255, 105, 135, .3)',
  },
  chartContainer: {
    height: 400,
    padding: theme.spacing(2),
  },
  mapContainer: {
    height: 500,
    padding: theme.spacing(2),
  },
  activityContainer: {
    height: 300,
    overflow: 'auto',
    padding: theme.spacing(2),
  },
  fab: {
    position: 'fixed',
    bottom: theme.spacing(2),
    right: theme.spacing(2),
  },
  dialogContent: {
    minWidth: 400,
  },
}));

const Dashboard: React.FC = () => {
  const classes = useStyles();
  const { user } = useAuth();
  const { socket } = useWebSocket();
  const { showNotification } = useNotification();

  // State
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [widgets, setWidgets] = useState<WidgetConfig[]>([]);
  const [selectedCountries, setSelectedCountries] = useState<string[]>(['USA', 'CHN', 'JPN', 'DEU', 'IND']);
  const [refreshing, setRefreshing] = useState(false);
  
  // Dialog states
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [addWidgetOpen, setAddWidgetOpen] = useState(false);
  
  // Menu states
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  // Load dashboard data
  const loadDashboardData = useCallback(async () => {
    try {
      setLoading(true);
      const data = await dashboardService.getDashboardData(selectedCountries);
      setDashboardData(data);
    } catch (error) {
      console.error('Error loading dashboard data:', error);
      showNotification('Error loading dashboard data', 'error');
    } finally {
      setLoading(false);
    }
  }, [selectedCountries, showNotification]);

  // Refresh data
  const handleRefresh = async () => {
    setRefreshing(true);
    await loadDashboardData();
    setRefreshing(false);
    showNotification('Dashboard data refreshed', 'success');
  };

  // WebSocket event handlers
  useEffect(() => {
    if (socket) {
      socket.on('dashboard_update', (data: any) => {
        setDashboardData(prev => ({ ...prev, ...data }));
      });

      socket.on('gdp_calculation_complete', (data: any) => {
        showNotification('New GDP calculation available', 'info');
        loadDashboardData();
      });
    }
  }, [socket, loadDashboardData, showNotification]);

  // Load initial data
  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  // Load user widget configuration
  useEffect(() => {
    const loadWidgets = async () => {
      try {
        const userWidgets = await dashboardService.getUserWidgets(user?.id);
        setWidgets(userWidgets);
      } catch (error) {
        console.error('Error loading widgets:', error);
      }
    };

    if (user) {
      loadWidgets();
    }
  }, [user]);

  const handleCountrySelection = (countries: string[]) => {
    setSelectedCountries(countries);
  };

  const handleAddWidget = async (widgetType: string) => {
    try {
      const newWidget: WidgetConfig = {
        id: `widget_${Date.now()}`,
        type: widgetType,
        title: `New ${widgetType} Widget`,
        position: { x: 0, y: 0 },
        size: { width: 12, height: 4 },
        config: {},
      };

      await dashboardService.saveWidget(user?.id, newWidget);
      setWidgets(prev => [...prev, newWidget]);
      setAddWidgetOpen(false);
      showNotification('Widget added successfully', 'success');
    } catch (error) {
      showNotification('Error adding widget', 'error');
    }
  };

  if (loading) {
    return (
      <Box className={classes.root}>
        <LinearProgress />
        <Typography variant="h6" align="center" style={{ marginTop: 20 }}>
          Loading dashboard...
        </Typography>
      </Box>
    );
  }

  return (
    <div className={classes.root}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          GDP Analytics Dashboard
        </Typography>
        <Box>
          <IconButton 
            onClick={handleRefresh} 
            disabled={refreshing}
            color="primary"
          >
            <RefreshIcon />
          </IconButton>
          <IconButton 
            onClick={(e) => setAnchorEl(e.currentTarget)}
            color="primary"
          >
            <SettingsIcon />
          </IconButton>
        </Box>
      </Box>

      {/* Settings Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
      >
        <MenuItem onClick={() => setSettingsOpen(true)}>
          Dashboard Settings
        </MenuItem>
        <MenuItem onClick={() => setAddWidgetOpen(true)}>
          Add Widget
        </MenuItem>
      </Menu>

      {/* KPI Cards */}
      <Grid container spacing={3} style={{ marginBottom: 24 }}>
        <Grid item xs={12} sm={6} md={3}>
          <KPICard
            title="Total Countries"
            value={dashboardData?.overview.totalCountries || 0}
            icon={<Public />}
            color="primary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <KPICard
            title="Latest GDP (USD Trillion)"
            value={dashboardData?.overview.latestGDP || 0}
            icon={<TrendingUp />}
            color="secondary"
            format="currency"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <KPICard
            title="Growth Rate (%)"
            value={dashboardData?.overview.growthRate || 0}
            icon={<Assessment />}
            color="success"
            format="percentage"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <KPICard
            title="Active Models"
            value={dashboardData?.forecasts?.length || 0}
            icon={<Timeline />}
            color="info"
          />
        </Grid>
      </Grid>

      {/* Main Charts */}
      <Grid container spacing={3}>
        {/* GDP Overview Chart */}
        <Grid item xs={12} lg={8}>
          <Paper className={classes.paper}>
            <Box className={classes.chartContainer}>
              <Typography variant="h6" gutterBottom>
                GDP Trends Overview
              </Typography>
              <GDPOverviewChart
                countries={selectedCountries}
                data={dashboardData?.recentCalculations || []}
                onCountrySelect={handleCountrySelection}
              />
            </Box>
          </Paper>
        </Grid>

        {/* AI Insights */}
        <Grid item xs={12} lg={4}>
          <Paper className={classes.paper}>
            <Box className={classes.activityContainer}>
              <Typography variant="h6" gutterBottom>
                AI Insights
              </Typography>
              <AIInsights 
                data={dashboardData}
                countries={selectedCountries}
              />
            </Box>
          </Paper>
        </Grid>

        {/* World Map */}
        <Grid item xs={12}>
          <Paper className={classes.paper}>
            <Box className={classes.mapContainer}>
              <Typography variant="h6" gutterBottom>
                Global GDP Visualization
              </Typography>
              <WorldMap
                data={dashboardData?.topCountries || []}
                selectedCountries={selectedCountries}
                onCountryClick={handleCountrySelection}
              />
            </Box>
          </Paper>
        </Grid>

        {/* Country Comparison */}
        <Grid item xs={12} md={8}>
          <Paper className={classes.paper}>
            <Box className={classes.chartContainer}>
              <Typography variant="h6" gutterBottom>
                Country Comparison
              </Typography>
              <CountryComparisonChart
                countries={selectedCountries}
                data={dashboardData?.recentCalculations || []}
              />
            </Box>
          </Paper>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} md={4}>
          <Paper className={classes.paper}>
            <Box className={classes.activityContainer}>
              <Typography variant="h6" gutterBottom>
                Recent Activity
              </Typography>
              <RecentActivity
                activities={dashboardData?.recentCalculations || []}
              />
            </Box>
          </Paper>
        </Grid>

        {/* Forecast Chart */}
        <Grid item xs={12}>
          <Paper className={classes.paper}>
            <Box className={classes.chartContainer}>
              <Typography variant="h6" gutterBottom>
                GDP Forecasts
              </Typography>
              <ForecastChart
                forecasts={dashboardData?.forecasts || []}
                countries={selectedCountries}
              />
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Voice Interface */}
      <VoiceInterface />

      {/* Floating Action Button */}
      <Fab
        color="primary"
        aria-label="add"
        className={classes.fab}
        onClick={() => setAddWidgetOpen(true)}
      >
        <AddIcon />
      </Fab>

      {/* Add Widget Dialog */}
      <Dialog open={addWidgetOpen} onClose={() => setAddWidgetOpen(false)}>
        <DialogTitle>Add New Widget</DialogTitle>
        <DialogContent className={classes.dialogContent}>
          <FormControl fullWidth margin="normal">
            <InputLabel>Widget Type</InputLabel>
            <Select>
              <MenuItem value="chart">Chart Widget</MenuItem>
              <MenuItem value="map">Map Widget</MenuItem>
              <MenuItem value="table">Data Table</MenuItem>
              <MenuItem value="kpi">KPI Card</MenuItem>
              <MenuItem value="ai">AI Insights</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddWidgetOpen(false)}>Cancel</Button>
          <Button onClick={() => handleAddWidget('chart')} color="primary">
            Add Widget
          </Button>
        </DialogActions>
      </Dialog>

      {/* Settings Dialog */}
      <Dialog open={settingsOpen} onClose={() => setSettingsOpen(false)}>
        <DialogTitle>Dashboard Settings</DialogTitle>
        <DialogContent className={classes.dialogContent}>
          <TextField
            fullWidth
            margin="normal"
            label="Refresh Interval (seconds)"
            type="number"
            defaultValue={30}
          />
          <FormControl fullWidth margin="normal">
            <InputLabel>Default Theme</InputLabel>
            <Select defaultValue="light">
              <MenuItem value="light">Light</MenuItem>
              <MenuItem value="dark">Dark</MenuItem>
              <MenuItem value="auto">Auto</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSettingsOpen(false)}>Cancel</Button>
          <Button onClick={() => setSettingsOpen(false)} color="primary">
            Save Settings
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
};

export default Dashboard;