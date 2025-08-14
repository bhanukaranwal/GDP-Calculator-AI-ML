import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Stepper,
  Step,
  StepLabel,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Card,
  CardContent,
  CardActions,
  Chip,
  LinearProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tabs,
  Tab,
  CircularProgress,
} from '@material-ui/core';
import {
  ExpandMore as ExpandMoreIcon,
  Calculate as CalculateIcon,
  TrendingUp,
  Assessment,
  Public,
  Save as SaveIcon,
  Download as DownloadIcon,
} from '@material-ui/icons';
import { makeStyles } from '@material-ui/core/styles';

// Components
import ExpenditureForm from '../../components/Forms/ExpenditureForm';
import IncomeForm from '../../components/Forms/IncomeForm';
import OutputForm from '../../components/Forms/OutputForm';
import ResultsVisualization from '../../components/Visualization/ResultsVisualization';
import DataValidation from '../../components/Validation/DataValidation';
import CountrySelector from '../../components/Selectors/CountrySelector';

// Services
import { gdpService } from '../../services/gdpService';
import { useNotification } from '../../contexts/NotificationContext';

// Types
interface GDPCalculationData {
  country_code: string;
  period: string;
  method: 'expenditure' | 'income' | 'output';
  data: any;
  apply_ai_corrections: boolean;
  include_uncertainty: boolean;
}

interface CalculationResult {
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

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1,
    padding: theme.spacing(3),
  },
  paper: {
    padding: theme.spacing(3),
    textAlign: 'center',
    color: theme.palette.text.secondary,
  },
  stepper: {
    marginBottom: theme.spacing(3),
  },
  formContainer: {
    padding: theme.spacing(3),
  },
  resultCard: {
    marginTop: theme.spacing(2),
    background: 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)',
    color: 'white',
  },
  chip: {
    margin: theme.spacing(0.5),
  },
  loadingContainer: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: 200,
  },
}));

const steps = ['Select Method', 'Enter Data', 'Validate', 'Calculate', 'Results'];

const GDPCalculator: React.FC = () => {
  const classes = useStyles();
  const { showNotification } = useNotification();

  // State
  const [activeStep, setActiveStep] = useState(0);
  const [calculationData, setCalculationData] = useState<GDPCalculationData>({
    country_code: '',
    period: '',
    method: 'expenditure',
    data: {},
    apply_ai_corrections: true,
    include_uncertainty: true,
  });
  const [result, setResult] = useState<CalculationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [validationResults, setValidationResults] = useState<any>(null);
  const [tabValue, setTabValue] = useState(0);

  // Handle next step
  const handleNext = async () => {
    if (activeStep === 2) {
      // Validation step
      await validateData();
    } else if (activeStep === 3) {
      // Calculation step
      await calculateGDP();
    } else {
      setActiveStep((prevActiveStep) => prevActiveStep + 1);
    }
  };

  // Handle back step
  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  // Handle reset
  const handleReset = () => {
    setActiveStep(0);
    setResult(null);
    setValidationResults(null);
    setCalculationData({
      country_code: '',
      period: '',
      method: 'expenditure',
      data: {},
      apply_ai_corrections: true,
      include_uncertainty: true,
    });
  };

  // Validate data
  const validateData = async () => {
    try {
      setLoading(true);
      const validation = await gdpService.validateData(
        calculationData.data,
        calculationData.country_code,
        calculationData.method
      );
      setValidationResults(validation);
      
      if (validation.valid) {
        setActiveStep(3);
        showNotification('Data validation successful', 'success');
      } else {
        showNotification('Data validation failed. Please check the issues.', 'warning');
      }
    } catch (error) {
      showNotification('Error validating data', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Calculate GDP
  const calculateGDP = async () => {
    try {
      setLoading(true);
      const result = await gdpService.calculateGDP(calculationData);
      setResult(result);
      setActiveStep(4);
      showNotification('GDP calculation completed successfully', 'success');
    } catch (error) {
      showNotification('Error calculating GDP', 'error');
      console.error('GDP calculation error:', error);
    } finally {
      setLoading(false);
    }
  };

  // Save results
  const saveResults = async () => {
    if (!result) return;
    
    try {
      await gdpService.saveCalculation(result);
      showNotification('Results saved successfully', 'success');
    } catch (error) {
      showNotification('Error saving results', 'error');
    }
  };

  // Export results
  const exportResults = () => {
    if (!result) return;
    
    const dataStr = JSON.stringify(result, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `gdp_calculation_${result.country_code}_${result.period}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  // Get step content
  const getStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <Box className={classes.formContainer}>
            <Typography variant="h6" gutterBottom>
              Select Calculation Method
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <CountrySelector
                  value={calculationData.country_code}
                  onChange={(country) => 
                    setCalculationData(prev => ({ ...prev, country_code: country }))
                  }
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Period"
                  placeholder="YYYY-Q1 or YYYY-MM or YYYY"
                  value={calculationData.period}
                  onChange={(e) => 
                    setCalculationData(prev => ({ ...prev, period: e.target.value }))
                  }
                  helperText="Enter the time period (e.g., 2024-Q1, 2024-03, 2024)"
                />
              </Grid>
              
              <Grid item xs={12}>
                <FormControl fullWidth>
                  <InputLabel>Calculation Method</InputLabel>
                  <Select
                    value={calculationData.method}
                    onChange={(e) => 
                      setCalculationData(prev => ({ 
                        ...prev, 
                        method: e.target.value as 'expenditure' | 'income' | 'output' 
                      }))
                    }
                  >
                    <MenuItem value="expenditure">Expenditure Approach (C + I + G + NX)</MenuItem>
                    <MenuItem value="income">Income Approach (Wages + Profits + Rents + Interest)</MenuItem>
                    <MenuItem value="output">Output Approach (Sectoral GVA)</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Box>
        );
        
      case 1:
        return (
          <Box className={classes.formContainer}>
            <Typography variant="h6" gutterBottom>
              Enter GDP Data
            </Typography>
            
            <Tabs value={tabValue} onChange={(e, value) => setTabValue(value)}>
              <Tab label="Data Entry" />
              <Tab label="AI Settings" />
            </Tabs>
            
            <Box hidden={tabValue !== 0}>
              {calculationData.method === 'expenditure' && (
                <ExpenditureForm
                  data={calculationData.data}
                  onChange={(data) => 
                    setCalculationData(prev => ({ ...prev, data }))
                  }
                />
              )}
              {calculationData.method === 'income' && (
                <IncomeForm
                  data={calculationData.data}
                  onChange={(data) => 
                    setCalculationData(prev => ({ ...prev, data }))
                  }
                />
              )}
              {calculationData.method === 'output' && (
                <OutputForm
                  data={calculationData.data}
                  onChange={(data) => 
                    setCalculationData(prev => ({ ...prev, data }))
                  }
                />
              )}
            </Box>
            
            <Box hidden={tabValue !== 1} mt={2}>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <FormControl component="fieldset">
                    <Typography variant="subtitle1">AI Enhancement Options</Typography>
                    <Box mt={1}>
                      <Button
                        variant={calculationData.apply_ai_corrections ? "contained" : "outlined"}
                        color="primary"
                        onClick={() => 
                          setCalculationData(prev => ({ 
                            ...prev, 
                            apply_ai_corrections: !prev.apply_ai_corrections 
                          }))
                        }
                      >
                        Apply AI Corrections
                      </Button>
                    </Box>
                    <Box mt={1}>
                      <Button
                        variant={calculationData.include_uncertainty ? "contained" : "outlined"}
                        color="secondary"
                        onClick={() => 
                          setCalculationData(prev => ({ 
                            ...prev, 
                            include_uncertainty: !prev.include_uncertainty 
                          }))
                        }
                      >
                        Include Uncertainty Analysis
                      </Button>
                    </Box>
                  </FormControl>
                </Grid>
              </Grid>
            </Box>
          </Box>
        );
        
      case 2:
        return (
          <Box className={classes.formContainer}>
            <Typography variant="h6" gutterBottom>
              Data Validation
            </Typography>
            
            {loading ? (
              <Box className={classes.loadingContainer}>
                <CircularProgress />
                <Typography variant="body1" style={{ marginLeft: 16 }}>
                  Validating data...
                </Typography>
              </Box>
            ) : (
              <DataValidation
                data={calculationData.data}
                validationResults={validationResults}
                method={calculationData.method}
              />
            )}
          </Box>
        );
        
      case 3:
        return (
          <Box className={classes.formContainer}>
            <Typography variant="h6" gutterBottom>
              Calculate GDP
            </Typography>
            
            {loading ? (
              <Box className={classes.loadingContainer}>
                <CircularProgress />
                <Typography variant="body1" style={{ marginLeft: 16 }}>
                  Calculating GDP with AI enhancements...
                </Typography>
              </Box>
            ) : (
              <Card>
                <CardContent>
                  <Typography variant="h6">Ready to Calculate</Typography>
                  <Typography variant="body2" color="textSecondary">
                    Country: {calculationData.country_code}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Period: {calculationData.period}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Method: {calculationData.method}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    AI Corrections: {calculationData.apply_ai_corrections ? 'Enabled' : 'Disabled'}
                  </Typography>
                </CardContent>
                <CardActions>
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={<CalculateIcon />}
                    onClick={calculateGDP}
                    disabled={loading}
                  >
                    Calculate GDP
                  </Button>
                </CardActions>
              </Card>
            )}
          </Box>
        );
        
      case 4:
        return (
          <Box className={classes.formContainer}>
            <Typography variant="h6" gutterBottom>
              Calculation Results
            </Typography>
            
            {result && (
              <>
                <Card className={classes.resultCard}>
                  <CardContent>
                    <Typography variant="h4" component="h2">
                      ${result.gdp_value.toLocaleString()} Billion
                    </Typography>
                    <Typography variant="subtitle1">
                      GDP for {result.country_code} - {result.period}
                    </Typography>
                    {result.confidence_interval && (
                      <Typography variant="body2">
                        95% Confidence Interval: ${result.confidence_interval[0].toLocaleString()} - ${result.confidence_interval[1].toLocaleString()} Billion
                      </Typography>
                    )}
                  </CardContent>
                </Card>
                
                <Box mt={2}>
                  <Grid container spacing={2}>
                    <Grid item>
                      <Chip 
                        label={`Quality Score: ${(result.quality_score * 100).toFixed(1)}%`}
                        color="primary"
                        className={classes.chip}
                      />
                    </Grid>
                    <Grid item>
                      <Chip 
                        label={`Method: ${result.method}`}
                        color="secondary"
                        className={classes.chip}
                      />
                    </Grid>
                    {Object.entries(result.anomaly_flags).map(([key, value]) => (
                      value && (
                        <Grid item key={key}>
                          <Chip 
                            label={`Anomaly: ${key}`}
                            color="default"
                            className={classes.chip}
                          />
                        </Grid>
                      )
                    ))}
                  </Grid>
                </Box>
                
                <Box mt={3}>
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography>Component Breakdown</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <ResultsVisualization result={result} />
                    </AccordionDetails>
                  </Accordion>
                </Box>
                
                <Box mt={2}>
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={<SaveIcon />}
                    onClick={saveResults}
                    style={{ marginRight: 16 }}
                  >
                    Save Results
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<DownloadIcon />}
                    onClick={exportResults}
                  >
                    Export Results
                  </Button>
                </Box>
              </>
            )}
          </Box>
        );
        
      default:
        return 'Unknown step';
    }
  };

  return (
    <div className={classes.root}>
      <Typography variant="h4" component="h1" gutterBottom>
        GDP Calculator
      </Typography>
      
      <Paper className={classes.paper}>
        <Stepper activeStep={activeStep} className={classes.stepper}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
        
        <div>
          {getStepContent(activeStep)}
          
          <Box mt={3}>
            <Button
              disabled={activeStep === 0}
              onClick={handleBack}
            >
              Back
            </Button>
            <Button
              variant="contained"
              color="primary"
              onClick={handleNext}
              disabled={
                loading || 
                (activeStep === 0 && (!calculationData.country_code || !calculationData.period)) ||
                (activeStep === 1 && Object.keys(calculationData.data).length === 0) ||
                (activeStep === 2 && validationResults && !validationResults.valid)
              }
              style={{ marginLeft: 8 }}
            >
              {activeStep === steps.length - 1 ? 'Finish' : 'Next'}
            </Button>
            {activeStep === steps.length - 1 && (
              <Button
                onClick={handleReset}
                style={{ marginLeft: 8 }}
              >
                Reset
              </Button>
            )}
          </Box>
        </div>
      </Paper>
    </div>
  );
};

export default GDPCalculator;