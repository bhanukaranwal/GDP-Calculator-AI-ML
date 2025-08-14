import axios, { AxiosResponse } from 'axios';

// Types
interface GDPCalculationRequest {
  country_code: string;
  period: string;
  method: 'expenditure' | 'income' | 'output';
  data: any;
  apply_ai_corrections: boolean;
  include_uncertainty: boolean;
}

interface GDPCalculationResult {
  gdp_value: number;
  components: any;
  country_code: string;
  period: string;
  method: string;
  confidence_interval?: [number, number];
  quality_score: number;
  anomaly_flags: { [key: string]: boolean };
  metadata: any;
  calculation_timestamp: string;
}

interface HistoricalDataParams {
  start_date?: string;
  end_date?: string;
  method?: string;
  include_components?: boolean;
}

interface ComparisonParams {
  country_codes: string[];
  period?: string;
  start_date?: string;
  end_date?: string;
  normalize?: boolean;
}

class GDPService {
  private baseURL: string;
  private apiClient: any;

  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
    
    // Create axios instance with default config
    this.apiClient = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add request interceptor for auth token
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

    // Add response interceptor for error handling
    this.apiClient.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error: any) => {
        if (error.response?.status === 401) {
          // Token expired, redirect to login
          localStorage.removeItem('authToken');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // GDP Calculation
  async calculateGDP(request: GDPCalculationRequest): Promise<GDPCalculationResult> {
    try {
      const response = await this.apiClient.post('/api/v1/gdp/calculate', request);
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to calculate GDP');
    }
  }

  // Get historical GDP data
  async getHistoricalData(
    countryCode: string,
    params?: HistoricalDataParams
  ): Promise<any> {
    try {
      const response = await this.apiClient.get(
        `/api/v1/gdp/historical/${countryCode}`,
        { params }
      );
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch historical data');
    }
  }

  // Compare countries
  async compareCountries(params: ComparisonParams): Promise<any> {
    try {
      const response = await this.apiClient.get('/api/v1/gdp/compare', {
        params: {
          country_codes: params.country_codes.join(','),
          ...params
        }
      });
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to compare countries');
    }
  }

  // Get GDP components
  async getComponents(
    countryCode: string,
    period: string,
    method: string = 'expenditure'
  ): Promise<any> {
    try {
      const response = await this.apiClient.get(
        `/api/v1/gdp/components/${countryCode}/${period}`,
        { params: { method } }
      );
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch components');
    }
  }

  // Validate GDP data
  async validateData(
    data: any,
    countryCode: string,
    method: string
  ): Promise<any> {
    try {
      const response = await this.apiClient.post('/api/v1/gdp/validate', data, {
        params: { country_code: countryCode, method }
      });
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to validate data');
    }
  }

  // Get data quality report
  async getQualityReport(
    countryCode: string,
    periodRange?: string
  ): Promise<any> {
    try {
      const response = await this.apiClient.get(
        `/api/v1/gdp/quality-report/${countryCode}`,
        { params: { period_range: periodRange } }
      );
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch quality report');
    }
  }

  // Bulk calculation
  async bulkCalculate(requests: GDPCalculationRequest[]): Promise<any> {
    try {
      const response = await this.apiClient.post('/api/v1/gdp/bulk-calculate', requests);
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to start bulk calculation');
    }
  }

  // Get GDP trends
  async getTrends(
    countryCode: string,
    lookbackPeriods: number = 12,
    includeForecast: boolean = false
  ): Promise<any> {
    try {
      const response = await this.apiClient.get(
        `/api/v1/gdp/trends/${countryCode}`,
        {
          params: {
            lookback_periods: lookbackPeriods,
            include_forecast: includeForecast
          }
        }
      );
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch trends');
    }
  }

  // Detect anomalies
  async detectAnomalies(
    countryCode: string,
    sensitivity: number = 0.05,
    method?: string
  ): Promise<any> {
    try {
      const response = await this.apiClient.get(
        `/api/v1/gdp/anomalies/${countryCode}`,
        {
          params: { sensitivity, method }
        }
      );
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to detect anomalies');
    }
  }

  // Save calculation result
  async saveCalculation(result: GDPCalculationResult): Promise<void> {
    try {
      await this.apiClient.post('/api/v1/gdp/save', result);
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to save calculation');
    }
  }

  // Get forecasts
  async getForecastsKnow(
    countryCode: string,
    horizon: number = 4
  ): Promise<any> {
    try {
      const response = await this.apiClient.get(
        `/api/v1/forecasting/predict/${countryCode}`,
        { params: { forecast_horizon: horizon } }
      );
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to get forecasts');
    }
  }

  // Get standards compliance info
  async getStandardsCompliance(standard: string = 'SNA2008'): Promise<any> {
    try {
      const response = await this.apiClient.get('/api/v1/gdp/metadata/standards', {
        params: { standard }
      });
      return response.data.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch standards info');
    }
  }
}

// Create singleton instance
export const gdpService = new GDPService();
export default gdpService;