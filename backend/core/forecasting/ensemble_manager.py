"""
Advanced Ensemble Manager for GDP Forecasting
Implements meta-learning and reinforcement learning for optimal model combinations
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import mlflow
import stable_baselines3 as sb3
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env

from core.models.forecasting_models import ForecastResult, ModelPrediction
from core.forecasting.base_models import (
    ARIMAModel, LSTMModel, TransformerModel, 
    XGBoostModel, ProphetModel
)
from core.ai_models.meta_learner import MetaLearner
from core.monitoring.model_monitor import ModelMonitor

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods"""
    models: List[str] = field(default_factory=lambda: [
        'arima', 'lstm', 'transformer', 'xgboost', 'prophet'
    ])
    ensemble_method: str = 'weighted_average'  # voting, stacking, weighted_average, rl_optimized
    optimization_method: str = 'bayesian'  # grid_search, random_search, bayesian, rl
    validation_method: str = 'time_series_split'
    n_splits: int = 5
    lookback_window: int = 12
    forecast_horizon: int = 4
    uncertainty_quantification: bool = True
    auto_retrain: bool = True
    performance_threshold: float = 0.05  # Retrain if MAPE > 5%


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    mse: float
    mae: float
    mape: float
    r2: float
    prediction_time: float
    last_updated: datetime
    stability_score: float


class EnsembleEnvironment:
    """RL Environment for ensemble weight optimization"""
    
    def __init__(self, models: List[Any], validation_data: pd.DataFrame):
        self.models = models
        self.validation_data = validation_data
        self.current_step = 0
        self.max_steps = len(validation_data) - 1
        self.weights = np.ones(len(models)) / len(models)
        
    def reset(self):
        """Reset environment"""
        self.current_step = 0
        self.weights = np.ones(len(self.models)) / len(self.models)
        return self._get_state()
    
    def step(self, action):
        """Take action and return next state, reward, done"""
        # Update weights based on action
        self.weights = self._action_to_weights(action)
        
        # Get predictions from current step
        predictions = self._get_ensemble_prediction()
        actual = self.validation_data.iloc[self.current_step]['target']
        
        # Calculate reward (negative MSE)
        reward = -((predictions - actual) ** 2)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_state(), reward, done, {}
    
    def _get_state(self):
        """Get current state"""
        if self.current_step >= len(self.validation_data):
            return np.zeros(len(self.models) + 1)
        
        # State includes model predictions and current weights
        predictions = [
            model.predict(self.validation_data.iloc[self.current_step])
            for model in self.models
        ]
        return np.array(predictions + [self.current_step / self.max_steps])
    
    def _action_to_weights(self, action):
        """Convert action to ensemble weights"""
        # Softmax to ensure weights sum to 1
        return np.exp(action) / np.sum(np.exp(action))
    
    def _get_ensemble_prediction(self):
        """Get weighted ensemble prediction"""
        predictions = [
            model.predict(self.validation_data.iloc[self.current_step])
            for model in self.models
        ]
        return np.dot(self.weights, predictions)


class EnsembleManager:
    """
    Advanced ensemble manager with meta-learning and RL optimization
    """
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.models = {}
        self.ensemble_weights = {}
        self.performance_history = {}
        self.meta_learner = MetaLearner()
        self.model_monitor = ModelMonitor()
        
        # Initialize base models
        self._initialize_models()
        
        # RL agent for weight optimization
        self.rl_agent = None
        self.rl_env = None
        
    def _initialize_models(self):
        """Initialize base forecasting models"""
        model_classes = {
            'arima': ARIMAModel,
            'lstm': LSTMModel,
            'transformer': TransformerModel,
            'xgboost': XGBoostModel,
            'prophet': ProphetModel
        }
        
        for model_name in self.config.models:
            if model_name in model_classes:
                self.models[model_name] = model_classes[model_name]()
                self.ensemble_weights[model_name] = 1.0 / len(self.config.models)
        
        logger.info(f"Initialized {len(self.models)} base models")
    
    async def train_ensemble(
        self,
        train_data: pd.DataFrame,
        target_column: str = 'gdp',
        feature_columns: List[str] = None
    ) -> Dict[str, Any]:
        """Train ensemble with all base models"""
        
        logger.info("Training ensemble models...")
        
        training_results = {}
        
        # Train each base model
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name}...")
                
                result = await model.train(
                    train_data,
                    target_column=target_column,
                    feature_columns=feature_columns
                )
                
                training_results[model_name] = result
                
                # Track model performance
                performance = ModelPerformance(
                    model_name=model_name,
                    mse=result.get('mse', float('inf')),
                    mae=result.get('mae', float('inf')),
                    mape=result.get('mape', float('inf')),
                    r2=result.get('r2', -float('inf')),
                    prediction_time=result.get('training_time', 0),
                    last_updated=datetime.utcnow(),
                    stability_score=result.get('stability_score', 0.5)
                )
                
                self.performance_history[model_name] = performance
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                training_results[model_name] = {'error': str(e)}
        
        # Optimize ensemble weights
        await self._optimize_ensemble_weights(train_data, target_column)
        
        # Train meta-learner
        await self._train_meta_learner(train_data, target_column)
        
        logger.info("Ensemble training completed")
        return training_results
    
    async def _optimize_ensemble_weights(
        self,
        data: pd.DataFrame,
        target_column: str
    ):
        """Optimize ensemble weights using selected method"""
        
        if self.config.optimization_method == 'rl':
            await self._optimize_weights_rl(data, target_column)
        elif self.config.optimization_method == 'bayesian':
            await self._optimize_weights_bayesian(data, target_column)
        else:
            await self._optimize_weights_cross_validation(data, target_column)
    
    async def _optimize_weights_rl(
        self,
        data: pd.DataFrame,
        target_column: str
    ):
        """Optimize weights using reinforcement learning"""
        
        logger.info("Optimizing ensemble weights with RL...")
        
        # Prepare validation data
        split_idx = int(len(data) * 0.8)
        validation_data = data[split_idx:].copy()
        
        # Create RL environment
        self.rl_env = EnsembleEnvironment(
            list(self.models.values()),
            validation_data
        )
        
        # Train RL agent
        self.rl_agent = PPO(
            'MlpPolicy',
            self.rl_env,
            learning_rate=0.001,
            n_steps=2048,
            batch_size=64,
            verbose=1
        )
        
        self.rl_agent.learn(total_timesteps=10000)
        
        # Get optimized weights
        state = self.rl_env.reset()
        action, _ = self.rl_agent.predict(state)
        optimized_weights = self.rl_env._action_to_weights(action)
        
        # Update ensemble weights
        for i, model_name in enumerate(self.models.keys()):
            self.ensemble_weights[model_name] = optimized_weights[i]
        
        logger.info(f"RL-optimized weights: {self.ensemble_weights}")
    
    async def _optimize_weights_bayesian(
        self,
        data: pd.DataFrame,
        target_column: str
    ):
        """Optimize weights using Bayesian optimization"""
        
        logger.info("Optimizing ensemble weights with Bayesian optimization...")
        
        def objective(trial):
            # Sample weights for each model
            weights = {}
            weight_sum = 0
            
            for model_name in self.models.keys():
                weight = trial.suggest_float(f'weight_{model_name}', 0.0, 1.0)
                weights[model_name] = weight
                weight_sum += weight
            
            # Normalize weights
            for model_name in weights:
                weights[model_name] /= weight_sum
            
            # Evaluate ensemble with these weights
            mse = self._evaluate_ensemble_weights(weights, data, target_column)
            return mse
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        
        # Extract best weights
        best_params = study.best_params
        best_weights = {}
        weight_sum = 0
        
        for model_name in self.models.keys():
            weight = best_params.get(f'weight_{model_name}', 1.0)
            best_weights[model_name] = weight
            weight_sum += weight
        
        # Normalize and update
        for model_name in best_weights:
            self.ensemble_weights[model_name] = best_weights[model_name] / weight_sum
        
        logger.info(f"Bayesian-optimized weights: {self.ensemble_weights}")
    
    async def _optimize_weights_cross_validation(
        self,
        data: pd.DataFrame,
        target_column: str
    ):
        """Optimize weights using cross-validation"""
        
        logger.info("Optimizing ensemble weights with cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        weight_combinations = []
        scores = []
        
        # Generate weight combinations
        for i in range(100):  # Try 100 random weight combinations
            weights = np.random.dirichlet(np.ones(len(self.models)))
            weight_dict = dict(zip(self.models.keys(), weights))
            weight_combinations.append(weight_dict)
        
        # Evaluate each combination
        for weights in weight_combinations:
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(data):
                train_data = data.iloc[train_idx]
                val_data = data.iloc[val_idx]
                
                mse = self._evaluate_ensemble_weights(weights, val_data, target_column)
                cv_scores.append(mse)
            
            scores.append(np.mean(cv_scores))
        
        # Select best weights
        best_idx = np.argmin(scores)
        self.ensemble_weights = weight_combinations[best_idx]
        
        logger.info(f"CV-optimized weights: {self.ensemble_weights}")
    
    def _evaluate_ensemble_weights(
        self,
        weights: Dict[str, float],
        data: pd.DataFrame,
        target_column: str
    ) -> float:
        """Evaluate ensemble with given weights"""
        
        predictions = []
        actuals = data[target_column].values
        
        for idx in range(len(data)):
            # Get predictions from each model
            model_preds = {}
            for model_name, model in self.models.items():
                try:
                    pred = model.predict_single(data.iloc[idx])
                    model_preds[model_name] = pred
                except:
                    model_preds[model_name] = np.mean(actuals)  # Fallback
            
            # Weighted ensemble prediction
            ensemble_pred = sum(
                weights[name] * pred
                for name, pred in model_preds.items()
            )
            predictions.append(ensemble_pred)
        
        # Calculate MSE
        mse = mean_squared_error(actuals, predictions)
        return mse
    
    async def _train_meta_learner(
        self,
        data: pd.DataFrame,
        target_column: str
    ):
        """Train meta-learner for model selection"""
        
        logger.info("Training meta-learner...")
        
        # Prepare meta-features and targets
        meta_features = []
        meta_targets = []
        
        for idx in range(len(data)):
            # Extract meta-features (data characteristics)
            features = self._extract_meta_features(data.iloc[max(0, idx-10):idx+1])
            
            # Get best model for this instance
            best_model = self._get_best_model_for_instance(data.iloc[idx], target_column)
            
            meta_features.append(features)
            meta_targets.append(best_model)
        
        # Train meta-learner
        await self.meta_learner.train(
            np.array(meta_features),
            np.array(meta_targets)
        )
        
        logger.info("Meta-learner training completed")
    
    def _extract_meta_features(self, data_slice: pd.DataFrame) -> List[float]:
        """Extract meta-features from data slice"""
        if len(data_slice) == 0:
            return [0.0] * 10
        
        features = []
        
        # Statistical features
        features.append(data_slice['gdp'].mean())
        features.append(data_slice['gdp'].std())
        features.append(data_slice['gdp'].skew())
        features.append(data_slice['gdp'].kurt())
        
        # Trend features
        if len(data_slice) > 1:
            features.append(np.polyfit(range(len(data_slice)), data_slice['gdp'], 1)[0])
        else:
            features.append(0.0)
        
        # Seasonality features (simplified)
        features.append(np.var(data_slice['gdp']))
        
        # Volatility
        if len(data_slice) > 1:
            returns = data_slice['gdp'].pct_change().dropna()
            features.append(returns.std() if len(returns) > 0 else 0.0)
        else:
            features.append(0.0)
        
        # Autocorrelation
        if len(data_slice) > 2:
            features.append(data_slice['gdp'].autocorr(lag=1))
        else:
            features.append(0.0)
        
        # Missing values ratio
        features.append(data_slice.isnull().sum().sum() / (len(data_slice) * len(data_slice.columns)))
        
        # Data recency (dummy feature)
        features.append(1.0)
        
        return features
    
    def _get_best_model_for_instance(
        self,
        instance: pd.Series,
        target_column: str
    ) -> str:
        """Get best performing model for specific instance"""
        
        # Simplified: return model with best recent performance
        if not self.performance_history:
            return list(self.models.keys())[0]
        
        best_model = min(
            self.performance_history.keys(),
            key=lambda x: self.performance_history[x].mse
        )
        
        return best_model
    
    async def predict(
        self,
        input_data: pd.DataFrame,
        forecast_horizon: int = None,
        return_uncertainty: bool = True
    ) -> ForecastResult:
        """Generate ensemble prediction"""
        
        forecast_horizon = forecast_horizon or self.config.forecast_horizon
        
        logger.info(f"Generating ensemble forecast for horizon {forecast_horizon}")
        
        # Get predictions from all models
        model_predictions = {}
        prediction_times = {}
        
        for model_name, model in self.models.items():
            try:
                start_time = datetime.utcnow()
                
                predictions = await model.predict(
                    input_data,
                    forecast_horizon=forecast_horizon
                )
                
                end_time = datetime.utcnow()
                prediction_times[model_name] = (end_time - start_time).total_seconds()
                
                model_predictions[model_name] = predictions
                
            except Exception as e:
                logger.error(f"Error in {model_name} prediction: {str(e)}")
                # Use fallback prediction
                model_predictions[model_name] = [0.0] * forecast_horizon
                prediction_times[model_name] = 0.0
        
        # Meta-learning model selection (if applicable)
        if self.config.ensemble_method == 'meta_learning':
            selected_models = await self._meta_learning_selection(input_data)
        else:
            selected_models = list(self.models.keys())
        
        # Generate ensemble predictions
        ensemble_predictions = []
        ensemble_uncertainties = []
        
        for step in range(forecast_horizon):
            step_predictions = []
            step_weights = []
            
            for model_name in selected_models:
                if model_name in model_predictions:
                    pred = model_predictions[model_name][step]
                    weight = self.ensemble_weights.get(model_name, 0.0)
                    
                    step_predictions.append(pred)
                    step_weights.append(weight)
            
            if step_predictions:
                # Normalize weights
                total_weight = sum(step_weights)
                if total_weight > 0:
                    step_weights = [w / total_weight for w in step_weights]
                else:
                    step_weights = [1.0 / len(step_predictions)] * len(step_predictions)
                
                # Weighted average
                ensemble_pred = np.average(step_predictions, weights=step_weights)
                ensemble_predictions.append(ensemble_pred)
                
                # Uncertainty estimation
                if return_uncertainty:
                    pred_std = np.std(step_predictions)
                    weight_entropy = -sum(w * np.log(w + 1e-8) for w in step_weights)
                    uncertainty = pred_std * (1 + weight_entropy)
                    ensemble_uncertainties.append(uncertainty)
                else:
                    ensemble_uncertainties.append(0.0)
            else:
                ensemble_predictions.append(0.0)
                ensemble_uncertainties.append(1.0)
        
        # Calculate confidence intervals
        confidence_intervals = []
        for i, (pred, unc) in enumerate(zip(ensemble_predictions, ensemble_uncertainties)):
            lower = pred - 1.96 * unc
            upper = pred + 1.96 * unc
            confidence_intervals.append((lower, upper))
        
        # Compile model info
        model_info = {}
        for model_name in self.models.keys():
            if model_name in self.performance_history:
                perf = self.performance_history[model_name]
                model_info[model_name] = {
                    'weight': self.ensemble_weights.get(model_name, 0.0),
                    'mse': perf.mse,
                    'mae': perf.mae,
                    'prediction_time': prediction_times.get(model_name, 0.0)
                }
        
        result = ForecastResult(
            predictions=ensemble_predictions,
            timestamps=[
                datetime.utcnow() + timedelta(days=30*i)
                for i in range(forecast_horizon)
            ],
            confidence_intervals=confidence_intervals,
            model_info=model_info,
            ensemble_method=self.config.ensemble_method,
            forecast_horizon=forecast_horizon,
            uncertainty_estimates=ensemble_uncertainties if return_uncertainty else None,
            metadata={
                'ensemble_weights': self.ensemble_weights,
                'selected_models': selected_models,
                'prediction_timestamp': datetime.utcnow().isoformat(),
                'config': self.config.__dict__
            }
        )
        
        logger.info("Ensemble prediction completed")
        return result
    
    async def _meta_learning_selection(
        self,
        input_data: pd.DataFrame
    ) -> List[str]:
        """Select best models using meta-learning"""
        
        # Extract meta-features
        meta_features = self._extract_meta_features(input_data.tail(10))
        
        # Get model recommendations
        recommended_models = await self.meta_learner.predict([meta_features])
        
        return recommended_models[0] if recommended_models else list(self.models.keys())
    
    async def evaluate_ensemble(
        self,
        test_data: pd.DataFrame,
        target_column: str = 'gdp'
    ) -> Dict[str, float]:
        """Evaluate ensemble performance"""
        
        logger.info("Evaluating ensemble performance...")
        
        predictions = []
        actuals = test_data[target_column].values
        
        # Generate predictions for test data
        for idx in range(len(test_data)):
            input_slice = test_data.iloc[max(0, idx-10):idx+1]
            
            result = await self.predict(
                input_slice,
                forecast_horizon=1,
                return_uncertainty=False
            )
            
            predictions.append(result.predictions[0])
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # MAPE
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # Directional accuracy
        actual_directions = np.diff(actuals) > 0
        pred_directions = np.diff(predictions) > 0
        directional_accuracy = np.mean(actual_directions == pred_directions) * 100
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
        
        logger.info(f"Ensemble evaluation completed: {metrics}")
        return metrics
    
    async def retrain_if_needed(
        self,
        recent_data: pd.DataFrame,
        target_column: str = 'gdp'
    ) -> bool:
        """Check if retraining is needed and retrain if so"""
        
        if not self.config.auto_retrain:
            return False
        
        # Evaluate current performance on recent data
        current_metrics = await self.evaluate_ensemble(recent_data, target_column)
        
        # Check if performance degraded beyond threshold
        current_mape = current_metrics.get('mape', float('inf'))
        
        if current_mape > self.config.performance_threshold * 100:
            logger.info(f"Performance degraded (MAPE: {current_mape:.2f}%), retraining...")
            
            # Retrain ensemble
            await self.train_ensemble(recent_data, target_column)
            
            # Log retraining event
            mlflow.log_metric('retrain_triggered_mape', current_mape)
            mlflow.log_metric('retrain_timestamp', datetime.utcnow().timestamp())
            
            return True
        
        return False
    
    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get current ensemble status and health"""
        
        status = {
            'models': list(self.models.keys()),
            'ensemble_weights': self.ensemble_weights,
            'last_training': None,
            'performance_summary': {},
            'health_score': 0.0
        }
        
        if self.performance_history:
            # Calculate health score
            health_scores = []
            performance_summary = {}
            
            for model_name, perf in self.performance_history.items():
                # Model health based on recent performance
                model_health = min(1.0, max(0.0, (100 - perf.mape) / 100))
                health_scores.append(model_health * self.ensemble_weights.get(model_name, 0.0))
                
                performance_summary[model_name] = {
                    'mse': perf.mse,
                    'mae': perf.mae,
                    'mape': perf.mape,
                    'last_updated': perf.last_updated.isoformat(),
                    'health_score': model_health
                }
            
            status['health_score'] = sum(health_scores)
            status['performance_summary'] = performance_summary
            
            # Find most recent training
            if self.performance_history:
                latest_update = max(
                    perf.last_updated for perf in self.performance_history.values()
                )
                status['last_training'] = latest_update.isoformat()
        
        return status


# Factory function
def create_ensemble_manager(config: EnsembleConfig = None) -> EnsembleManager:
    """Create and return configured ensemble manager"""
    return EnsembleManager(config)
