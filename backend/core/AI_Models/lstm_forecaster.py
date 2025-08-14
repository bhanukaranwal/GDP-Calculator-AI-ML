"""
Advanced LSTM Model for GDP Forecasting
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pickle
import json

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna

from core.config import settings
from core.models.gdp_models import ForecastResult

logger = logging.getLogger(__name__)


class LSTMForecaster:
    """
    Advanced LSTM model for GDP forecasting with attention mechanism
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        self.training_history = None
        self.feature_names = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default model configuration"""
        return {
            'sequence_length': 12,
            'lstm_units': [128, 64, 32],
            'dropout_rate': 0.2,
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001,
            'patience': 10,
            'validation_split': 0.2,
            'use_attention': True,
            'use_batch_norm': True,
            'l1_reg': 0.01,
            'l2_reg': 0.01
        }
    
    def _create_model(self, input_shape: Tuple[int, int]) -> Model:
        """Create LSTM model with attention mechanism"""
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            self.config['lstm_units'][0],
            return_sequences=True,
            input_shape=input_shape,
            kernel_regularizer=l1_l2(
                l1=self.config['l1_reg'],
                l2=self.config['l2_reg']
            )
        ))
        
        if self.config['use_batch_norm']:
            model.add(BatchNormalization())
        
        model.add(Dropout(self.config['dropout_rate']))
        
        # Additional LSTM layers
        for i, units in enumerate(self.config['lstm_units'][1:], 1):
            return_seq = i < len(self.config['lstm_units']) - 1
            
            model.add(LSTM(
                units,
                return_sequences=return_seq,
                kernel_regularizer=l1_l2(
                    l1=self.config['l1_reg'],
                    l2=self.config['l2_reg']
                )
            ))
            
            if self.config['use_batch_norm']:
                model.add(BatchNormalization())
            
            model.add(Dropout(self.config['dropout_rate']))
        
        # Dense layers
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(self.config['dropout_rate']))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def _prepare_sequences(
        self,
        data: np.ndarray,
        target: np.ndarray,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features"""
        
        # Moving averages
        for window in [3, 6, 12]:
            df[f'ma_{window}'] = df['gdp'].rolling(window=window).mean()
            df[f'ma_ratio_{window}'] = df['gdp'] / df[f'ma_{window}']
        
        # Exponential moving averages
        for span in [3, 6, 12]:
            df[f'ema_{span}'] = df['gdp'].ewm(span=span).mean()
        
        # Rate of change
        for periods in [1, 2, 4]:
            df[f'roc_{periods}'] = df['gdp'].pct_change(periods=periods)
        
        # Volatility (rolling standard deviation)
        for window in [3, 6, 12]:
            df[f'volatility_{window}'] = df['gdp'].rolling(window=window).std()
        
        # Momentum indicators
        df['momentum_3'] = df['gdp'] - df['gdp'].shift(3)
        df['momentum_6'] = df['gdp'] - df['gdp'].shift(6)
        
        # Seasonal features
        if 'period' in df.columns:
            df['quarter'] = pd.to_datetime(df['period']).dt.quarter
            df['month'] = pd.to_datetime(df['period']).dt.month
            df['year'] = pd.to_datetime(df['period']).dt.year
        
        # Lag features
        for lag in [1, 2, 3, 4]:
            df[f'gdp_lag_{lag}'] = df['gdp'].shift(lag)
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for GDP data"""
        
        # Add technical features
        df = self._add_technical_features(df)
        
        # Add external economic indicators if available
        if 'inflation' in df.columns:
            df['real_gdp_growth'] = df['gdp'].pct_change() - df['inflation']
        
        if 'unemployment' in df.columns:
            df['okun_indicator'] = df['unemployment'].diff() * -2.5  # Okun's law approximation
        
        # Add cyclical components using Hodrick-Prescott filter
        try:
            from statsmodels.tsa.filters.hp_filter import hpfilter
            cycle, trend = hpfilter(df['gdp'].dropna(), lamb=1600)
            df.loc[cycle.index, 'gdp_cycle'] = cycle
            df.loc[trend.index, 'gdp_trend'] = trend
        except ImportError:
            logger.warning("statsmodels not available for HP filter")
        
        return df
    
    async def train(
        self,
        data: pd.DataFrame,
        target_column: str = 'gdp',
        feature_columns: List[str] = None,
        optimize_hyperparameters: bool = False
    ) -> Dict[str, Any]:
        """Train the LSTM model"""
        
        logger.info("Starting LSTM model training...")
        
        try:
            # Feature engineering
            engineered_data = self._engineer_features(data.copy())
            
            # Select features
            if feature_columns is None:
                # Use all numeric columns except target
                feature_columns = [col for col in engineered_data.select_dtypes(include=[np.number]).columns 
                                 if col != target_column]
            
            self.feature_names = feature_columns
            
            # Prepare data
            features = engineered_data[feature_columns].fillna(method='ffill').fillna(method='bfill')
            target = engineered_data[target_column].fillna(method='ffill').fillna(method='bfill')
            
            # Scale features and target
            features_scaled = self.feature_scaler.fit_transform(features)
            target_scaled = self.scaler.fit_transform(target.values.reshape(-1, 1)).flatten()
            
            # Create sequences
            X, y = self._prepare_sequences(
                features_scaled,
                target_scaled,
                self.config['sequence_length']
            )
            
            # Split data
            split_idx = int(len(X) * (1 - self.config['validation_split']))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Optimize hyperparameters if requested
            if optimize_hyperparameters:
                best_params = await self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
                self.config.update(best_params)
            
            # Create and train model
            self.model = self._create_model((X.shape[1], X.shape[2]))
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config['patience'],
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                ),
                ModelCheckpoint(
                    f"{settings.MODEL_STORAGE_PATH}/lstm_best_model.h5",
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            self.training_history = history.history
            self.is_trained = True
            
            # Evaluate model
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
            
            # Inverse transform predictions
            train_pred_inv = self.scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
            val_pred_inv = self.scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()
            y_train_inv = self.scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
            y_val_inv = self.scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train_inv, train_pred_inv)
            val_mse = mean_squared_error(y_val_inv, val_pred_inv)
            train_mae = mean_absolute_error(y_train_inv, train_pred_inv)
            val_mae = mean_absolute_error(y_val_inv, val_pred_inv)
            
            training_results = {
                'model_type': 'LSTM',
                'train_mse': train_mse,
                'val_mse': val_mse,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'train_rmse': np.sqrt(train_mse),
                'val_rmse': np.sqrt(val_mse),
                'epochs_trained': len(history.history['loss']),
                'final_lr': float(self.model.optimizer.learning_rate),
                'config': self.config,
                'feature_names': self.feature_names
            }
            
            logger.info(f"LSTM training completed. Validation RMSE: {np.sqrt(val_mse):.4f}")
            return training_results
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}", exc_info=True)
            raise
    
    async def _optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            # Suggest hyperparameters
            lstm_units_1 = trial.suggest_int('lstm_units_1', 64, 256, step=32)
            lstm_units_2 = trial.suggest_int('lstm_units_2', 32, 128, step=16)
            lstm_units_3 = trial.suggest_int('lstm_units_3', 16, 64, step=8)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            # Create temporary config
            temp_config = self.config.copy()
            temp_config.update({
                'lstm_units': [lstm_units_1, lstm_units_2, lstm_units_3],
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': 20  # Reduced for optimization
            })
            
            # Create temporary model
            temp_model = self._create_model((X_train.shape[1], X_train.shape[2]))
            
            # Train with early stopping
            history = temp_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=temp_config['epochs'],
                batch_size=temp_config['batch_size'],
                callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
                verbose=0
            )
            
            return min(history.history['val_loss'])
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        
        best_params = study.best_params
        return {
            'lstm_units': [
                best_params['lstm_units_1'],
                best_params['lstm_units_2'],
                best_params['lstm_units_3']
            ],
            'dropout_rate': best_params['dropout_rate'],
            'learning_rate': best_params['learning_rate'],
            'batch_size': best_params['batch_size']
        }
    
    async def predict(
        self,
        input_data: pd.DataFrame,
        forecast_horizon: int = 4
    ) -> List[float]:
        """Generate predictions"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Feature engineering
            engineered_data = self._engineer_features(input_data.copy())
            
            # Prepare features
            features = engineered_data[self.feature_names].fillna(method='ffill').fillna(method='bfill')
            features_scaled = self.feature_scaler.transform(features)
            
            # Get last sequence
            if len(features_scaled) < self.config['sequence_length']:
                # Pad with last available values if insufficient data
                padding_needed = self.config['sequence_length'] - len(features_scaled)
                last_values = features_scaled[-1:].repeat(padding_needed, axis=0)
                features_scaled = np.vstack([last_values, features_scaled])
            
            last_sequence = features_scaled[-self.config['sequence_length']:]
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(forecast_horizon):
                # Predict next value
                next_pred = self.model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
                next_pred_scaled = next_pred[0, 0]
                
                # Inverse transform prediction
                next_pred_actual = self.scaler.inverse_transform([[next_pred_scaled]])[0, 0]
                predictions.append(float(next_pred_actual))
                
                # Update sequence for next prediction
                # Create next feature vector (simplified - in practice, would need actual future features)
                next_features = current_sequence[-1].copy()
                next_features[0] = next_pred_scaled  # Update GDP value
                
                # Shift sequence
                current_sequence = np.vstack([current_sequence[1:], next_features.reshape(1, -1)])
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making LSTM predictions: {str(e)}", exc_info=True)
            raise
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Save Keras model
        self.model.save(f"{filepath}_model.h5")
        
        # Save scalers and config
        model_data = {
            'config': self.config,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'scaler': self.scaler,
            'feature_scaler': self.feature_scaler
        }
        
        with open(f"{filepath}_data.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"LSTM model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            # Load Keras model
            self.model = tf.keras.models.load_model(f"{filepath}_model.h5")
            
            # Load scalers and config
            with open(f"{filepath}_data.pkl", 'rb') as f:
                model_data = pickle.load(f)
            
            self.config = model_data['config']
            self.feature_names = model_data['feature_names']
            self.training_history = model_data['training_history']
            self.scaler = model_data['scaler']
            self.feature_scaler = model_data['feature_scaler']
            self.is_trained = True
            
            logger.info(f"LSTM model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading LSTM model: {str(e)}", exc_info=True)
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (simplified for LSTM)"""
        if not self.is_trained:
            return {}
        
        # For LSTM, we can use the first layer weights as a proxy for importance
        try:
            first_layer_weights = self.model.layers[0].get_weights()[0]
            feature_importance = np.mean(np.abs(first_layer_weights), axis=1)
            
            importance_dict = {}
            for i, feature in enumerate(self.feature_names):
                if i < len(feature_importance):
                    importance_dict[feature] = float(feature_importance[i])
            
            # Normalize to sum to 1
            total_importance = sum(importance_dict.values())
            if total_importance > 0:
                importance_dict = {k: v/total_importance for k, v in importance_dict.items()}
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {}


# Factory function
def create_lstm_forecaster(config: Dict[str, Any] = None) -> LSTMForecaster:
    """Create and return LSTM forecaster"""
    return LSTMForecaster(config)
