import axios, { AxiosResponse } from 'axios';

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

interface MetricsData {
  gdp_trends: any[];
  country_rankings: any[];
  forecast_accuracy: any[];
  data_quality_scores: any[];
}

class DashboardService {
  private baseURL: string;
  private apiClient: any;

  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
    
    this.apiClient = axios.create({
      baseURL: this.baseURL,
      timeout: 15000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add auth interceptor
    this.apiClient.interceptors.request.use(
      (config: any) => {
        const token = localStorage.getItem('authToken');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error: any) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.apiClient.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error: any) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('authToken');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Get dashboard overview data
  async getDashboardData(countries: string[] = []): Promise<DashboardData> {
    try {
      const response = await this.apiClient.get('/api/v1/dashboard/overview', {
        params: { countries: countries.join(',') }
      });
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch dashboard data');
    }
  }

  // Get real-time metrics
  async getMetrics(timeRange: string = '24h'): Promise<MetricsData> {
    try {
      const response = await this.apiClient.get('/api/v1/dashboard/metrics', {
        params: { time_range: timeRange }
      });
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch metrics');
    }
  }

  // Get user widgets configuration
  async getUserWidgets(userId: string): Promise<WidgetConfig[]> {
    try {
      const response = await this.apiClient.get(`/api/v1/dashboard/widgets/${userId}`);
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch user widgets');
    }
  }

  // Save widget configuration
  async saveWidget(userId: string, widget: WidgetConfig): Promise<void> {
    try {
      await this.apiClient.post(`/api/v1/dashboard/widgets/${userId}`, widget);
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to save widget');
    }
  }

  // Update widget configuration
  async updateWidget(userId: string, widgetId: string, updates: Partial<WidgetConfig>): Promise<void> {
    try {
      await this.apiClient.patch(`/api/v1/dashboard/widgets/${userId}/${widgetId}`, updates);
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to update widget');
    }
  }

  // Delete widget
  async deleteWidget(userId: string, widgetId: string): Promise<void> {
    try {
      await this.apiClient.delete(`/api/v1/dashboard/widgets/${userId}/${widgetId}`);
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to delete widget');
    }
  }

  // Get alerts and notifications
  async getAlerts(userId: string, limit: number = 10): Promise<any[]> {
    try {
      const response = await this.apiClient.get(`/api/v1/dashboard/alerts/${userId}`, {
        params: { limit }
      });
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch alerts');
    }
  }

  // Mark alert as read
  async markAlertAsRead(alertId: string): Promise<void> {
    try {
      await this.apiClient.patch(`/api/v1/dashboard/alerts/${alertId}/read`);
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to mark alert as read');
    }
  }

  // Get system health status
  async getSystemHealth(): Promise<any> {
    try {
      const response = await this.apiClient.get('/api/v1/dashboard/health');
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch system health');
    }
  }

  // Export dashboard data
  async exportDashboard(format: 'pdf' | 'excel' | 'json', options: any = {}): Promise<Blob> {
    try {
      const response = await this.apiClient.post('/api/v1/dashboard/export', {
        format,
        options
      }, {
        responseType: 'blob'
      });
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to export dashboard');
    }
  }

  // Get country suggestions based on user input
  async getCountrySuggestions(query: string): Promise<any[]> {
    try {
      const response = await this.apiClient.get('/api/v1/dashboard/countries/suggest', {
        params: { q: query }
      });
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch country suggestions');
    }
  }

  // Get available data sources
  async getDataSources(): Promise<any[]> {
    try {
      const response = await this.apiClient.get('/api/v1/dashboard/data-sources');
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch data sources');
    }
  }

  // Trigger data refresh
  async refreshData(sources: string[] = []): Promise<any> {
    try {
      const response = await this.apiClient.post('/api/v1/dashboard/refresh', {
        sources
      });
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to refresh data');
    }
  }

  // Get calculation history
  async getCalculationHistory(userId: string, limit: number = 20): Promise<any[]> {
    try {
      const response = await this.apiClient.get(`/api/v1/dashboard/calculations/${userId}`, {
        params: { limit }
      });
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch calculation history');
    }
  }

  // Save dashboard layout
  async saveDashboardLayout(userId: string, layout: any): Promise<void> {
    try {
      await this.apiClient.post(`/api/v1/dashboard/layout/${userId}`, layout);
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to save dashboard layout');
    }
  }

  // Get dashboard layout
  async getDashboardLayout(userId: string): Promise<any> {
    try {
      const response = await this.apiClient.get(`/api/v1/dashboard/layout/${userId}`);
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch dashboard layout');
    }
  }

  // Get performance metrics
  async getPerformanceMetrics(timeRange: string = '24h'): Promise<any> {
    try {
      const response = await this.apiClient.get('/api/v1/dashboard/performance', {
        params: { time_range: timeRange }
      });
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch performance metrics');
    }
  }

  // Subscribe to real-time updates
  async subscribeToUpdates(topics: string[]): Promise<void> {
    try {
      await this.apiClient.post('/api/v1/dashboard/subscribe', {
        topics
      });
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to subscribe to updates');
    }
  }

  // Get usage statistics
  async getUsageStatistics(userId: string, period: string = 'month'): Promise<any> {
    try {
      const response = await this.apiClient.get(`/api/v1/dashboard/usage/${userId}`, {
        params: { period }
      });
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch usage statistics');
    }
  }
}

// Create singleton instance
export const dashboardService = new DashboardService();
export default dashboardService;