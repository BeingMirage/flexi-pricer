import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
from django.utils import timezone
import warnings
warnings.filterwarnings('ignore')

class PricingMLOptimizer:
    def __init__(self):
        self.demand_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.price_elasticity_model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.models_trained = False
        
    def prepare_features(self, product, sales_data, traffic_data, historical_prices=None):
        """Prepare features for ML models"""
        features = {}
        
        # Basic product features
        features['price'] = float(product.price)
        features['inventory'] = product.inventory
        features['days_since_created'] = (timezone.now() - product.created_at).days if hasattr(product, 'created_at') else 30
        
        # Sales velocity features (last 7, 14, 30 days)
        for days in [7, 14, 30]:
            cutoff_date = timezone.now() - timedelta(days=days)
            recent_sales = [s for s in sales_data if s['timestamp'] >= cutoff_date]
            features[f'sales_velocity_{days}d'] = sum(s['quantity'] for s in recent_sales) / days
            features[f'sales_count_{days}d'] = len(recent_sales)
        
        # Traffic features
        for days in [7, 14, 30]:
            cutoff_date = timezone.now() - timedelta(days=days)
            recent_traffic = [t for t in traffic_data if t['timestamp'] >= cutoff_date]
            features[f'traffic_avg_{days}d'] = np.mean([t['visits'] for t in recent_traffic]) if recent_traffic else 0
            features[f'traffic_total_{days}d'] = sum(t['visits'] for t in recent_traffic)
        
        # Conversion rate features
        for days in [7, 14, 30]:
            cutoff_date = timezone.now() - timedelta(days=days)
            recent_sales = [s for s in sales_data if s['timestamp'] >= cutoff_date]
            recent_traffic = [t for t in traffic_data if t['timestamp'] >= cutoff_date]
            
            total_sales = sum(s['quantity'] for s in recent_sales)
            total_visits = sum(t['visits'] for t in recent_traffic)
            features[f'conversion_rate_{days}d'] = total_sales / total_visits if total_visits > 0 else 0
        
        # Price elasticity features (if historical data available)
        if historical_prices and len(historical_prices) > 1:
            price_changes = np.diff([p['price'] for p in historical_prices])
            demand_changes = np.diff([p['demand'] for p in historical_prices])
            features['price_elasticity'] = np.mean(demand_changes / price_changes) if np.any(price_changes != 0) else -1.0
        else:
            features['price_elasticity'] = -1.0  # Default inelastic
        
        # Seasonal features
        current_date = timezone.now()
        features['day_of_week'] = current_date.weekday()
        features['month'] = current_date.month
        features['is_weekend'] = 1 if current_date.weekday() >= 5 else 0
        
        # Competition features (simulated)
        features['competitor_price_ratio'] = np.random.uniform(0.8, 1.2)  # Simulated competitor pricing
        features['market_demand_index'] = np.random.uniform(0.7, 1.3)  # Simulated market conditions
        
        return features
    
    def train_demand_model(self, training_data):
        """Train demand prediction model"""
        X = []
        y = []
        
        for data_point in training_data:
            features = data_point['features']
            demand = data_point['demand']
            
            feature_vector = list(features.values())
            X.append(feature_vector)
            y.append(demand)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.demand_model.fit(X_scaled, y)
        
        return True
    
    def predict_demand(self, features):
        """Predict demand for given features"""
        if not self.models_trained:
            return None
        
        feature_vector = list(features.values())
        X_scaled = self.scaler.transform([feature_vector])
        predicted_demand = self.demand_model.predict(X_scaled)[0]
        
        return max(0, predicted_demand)  # Ensure non-negative demand
    
    def calculate_optimal_price(self, product, sales_data, traffic_data, historical_prices=None):
        """Calculate optimal price using ML models"""
        features = self.prepare_features(product, sales_data, traffic_data, historical_prices)
        
        # Predict demand at current price
        current_demand = self.predict_demand(features)
        
        if current_demand is None:
            # Fallback to rule-based pricing if ML model not trained
            return self.rule_based_pricing(product, sales_data, traffic_data)
        
        # Price optimization using gradient descent
        optimal_price = self.optimize_price_gradient_descent(features, current_demand)
        
        # Apply business constraints
        optimal_price = self.apply_business_constraints(optimal_price, product.price)
        
        return {
            'optimal_price': optimal_price,
            'current_demand': current_demand,
            'predicted_revenue': optimal_price * current_demand,
            'confidence_score': self.calculate_confidence_score(features),
            'ml_model_used': 'Random Forest + Gradient Descent'
        }
    
    def optimize_price_gradient_descent(self, features, current_demand, learning_rate=0.01, iterations=100):
        """Optimize price using gradient descent"""
        current_price = features['price']
        best_price = current_price
        best_revenue = current_price * current_demand
        
        for _ in range(iterations):
            # Calculate revenue at current price
            revenue = current_price * current_demand
            
            # Estimate demand change with price change (using price elasticity)
            price_elasticity = features['price_elasticity']
            if price_elasticity == -1.0:  # Default inelastic
                price_elasticity = -0.5
            
            # Calculate gradient (derivative of revenue with respect to price)
            demand_change = price_elasticity * (current_demand / current_price)
            gradient = current_demand + current_price * demand_change
            
            # Update price
            new_price = current_price + learning_rate * gradient
            
            # Update demand based on price change
            price_change_ratio = (new_price - current_price) / current_price
            new_demand = current_demand * (1 + price_elasticity * price_change_ratio)
            
            # Calculate new revenue
            new_revenue = new_price * new_demand
            
            if new_revenue > best_revenue:
                best_revenue = new_revenue
                best_price = new_price
            
            current_price = new_price
            current_demand = new_demand
        
        return best_price
    
    def apply_business_constraints(self, optimal_price, current_price):
        """Apply business constraints to optimal price"""
        # Maximum price increase/decrease limits
        max_increase = current_price * 1.5  # 50% max increase
        max_decrease = current_price * 0.5  # 50% max decrease
        
        # Minimum price constraints
        min_price = 1.0  # Minimum $1 price
        
        # Apply constraints
        constrained_price = max(min_price, min(max_increase, max(optimal_price, max_decrease)))
        
        return constrained_price
    
    def calculate_confidence_score(self, features):
        """Calculate confidence score for the prediction"""
        # Higher confidence for more data points
        data_richness = min(1.0, features['sales_count_30d'] / 100)
        
        # Higher confidence for stable patterns
        sales_stability = 1.0 - abs(features['sales_velocity_7d'] - features['sales_velocity_30d']) / max(features['sales_velocity_30d'], 1)
        
        # Higher confidence for inelastic products
        elasticity_confidence = 1.0 - abs(features['price_elasticity']) / 2.0
        
        confidence = (data_richness + sales_stability + elasticity_confidence) / 3
        return max(0.1, min(1.0, confidence))
    
    def rule_based_pricing(self, product, sales_data, traffic_data):
        """Fallback rule-based pricing (original logic)"""
        # Calculate metrics
        week_ago = timezone.now() - timedelta(days=7)
        recent_sales = [s for s in sales_data if s['timestamp'] >= week_ago]
        recent_traffic = [t for t in traffic_data if t['timestamp'] >= week_ago]
        
        total_sales = sum(s['quantity'] for s in recent_sales)
        total_visits = sum(t['visits'] for t in recent_traffic)
        
        sales_velocity = total_sales / 7
        conversion_rate = total_sales / total_visits if total_visits > 0 else 0
        
        # Apply rule-based logic
        current_price = float(product.price)
        new_price = current_price
        
        if sales_velocity > 5 and product.inventory < 20:
            new_price = current_price * 1.1
        elif conversion_rate < 0.05 and total_visits > 50:
            new_price = current_price * 0.9
        elif sales_velocity < 1 and product.inventory > 50:
            new_price = current_price * 0.7
        
        return {
            'optimal_price': new_price,
            'current_demand': sales_velocity,
            'predicted_revenue': new_price * sales_velocity,
            'confidence_score': 0.5,
            'ml_model_used': 'Rule-based (fallback)'
        }
    
    def generate_training_data(self, products, sales_data, traffic_data):
        """Generate synthetic training data for ML models"""
        training_data = []
        
        for product in products:
            # Generate multiple scenarios with different prices
            base_price = float(product.price)
            
            for price_multiplier in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
                # Create synthetic scenario
                synthetic_product = type('Product', (), {
                    'price': base_price * price_multiplier,
                    'inventory': product.inventory,
                    'created_at': product.created_at if hasattr(product, 'created_at') else timezone.now() - timedelta(days=30)
                })()
                
                # Generate synthetic sales based on price elasticity
                features = self.prepare_features(synthetic_product, sales_data, traffic_data)
                
                # Simulate demand based on price (inverse relationship)
                base_demand = features['sales_velocity_7d']
                price_elasticity = -0.8  # Typical price elasticity
                price_change_ratio = (synthetic_product.price - base_price) / base_price
                synthetic_demand = base_demand * (1 + price_elasticity * price_change_ratio)
                
                training_data.append({
                    'features': features,
                    'demand': max(0, synthetic_demand)
                })
        
        return training_data

# Global ML optimizer instance
ml_optimizer = PricingMLOptimizer() 