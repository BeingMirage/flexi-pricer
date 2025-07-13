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
        self.stock_prediction_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.stock_scaler = StandardScaler()
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
    
    def prepare_stock_features(self, product, sales_data, traffic_data):
        """Prepare features specifically for stock prediction"""
        features = {}
        
        # Current inventory
        features['current_inventory'] = product.inventory
        
        # Historical sales patterns
        for days in [7, 14, 30, 60, 90]:
            cutoff_date = timezone.now() - timedelta(days=days)
            recent_sales = [s for s in sales_data if s['timestamp'] >= cutoff_date]
            features[f'daily_sales_avg_{days}d'] = sum(s['quantity'] for s in recent_sales) / days
            features[f'sales_volatility_{days}d'] = np.std([s['quantity'] for s in recent_sales]) if recent_sales else 0
        
        # Seasonal patterns
        current_date = timezone.now()
        features['month'] = current_date.month
        features['quarter'] = (current_date.month - 1) // 3 + 1
        features['is_holiday_season'] = 1 if current_date.month in [11, 12] else 0  # Nov-Dec
        features['is_summer'] = 1 if current_date.month in [6, 7, 8] else 0
        
        # Growth trends
        week_ago = timezone.now() - timedelta(days=7)
        month_ago = timezone.now() - timedelta(days=30)
        
        recent_sales = [s for s in sales_data if s['timestamp'] >= week_ago]
        older_sales = [s for s in sales_data if week_ago > s['timestamp'] >= month_ago]
        
        recent_avg = sum(s['quantity'] for s in recent_sales) / 7 if recent_sales else 0
        older_avg = sum(s['quantity'] for s in older_sales) / 23 if older_sales else 0
        
        features['sales_growth_rate'] = (recent_avg - older_avg) / max(older_avg, 1)
        
        # Traffic-based demand indicators
        recent_traffic = [t for t in traffic_data if t['timestamp'] >= week_ago]
        features['traffic_trend'] = np.mean([t['visits'] for t in recent_traffic]) if recent_traffic else 0
        
        # Product lifecycle features
        if hasattr(product, 'created_at'):
            days_since_created = (timezone.now() - product.created_at).days
            features['product_age_days'] = days_since_created
            features['is_new_product'] = 1 if days_since_created < 30 else 0
        else:
            features['product_age_days'] = 30
            features['is_new_product'] = 0
        
        return features
    
    def predict_stock_requirements(self, product, sales_data, traffic_data, prediction_days=30):
        """Predict stock requirements for the next N days"""
        features = self.prepare_stock_features(product, sales_data, traffic_data)
        
        # Calculate current daily sales rate
        week_ago = timezone.now() - timedelta(days=7)
        recent_sales = [s for s in sales_data if s['timestamp'] >= week_ago]
        current_daily_sales = sum(s['quantity'] for s in recent_sales) / 7 if recent_sales else 0
        
        # Apply seasonal adjustments
        seasonal_multiplier = self.calculate_seasonal_multiplier(features)
        adjusted_daily_sales = current_daily_sales * seasonal_multiplier
        
        # Calculate safety stock (buffer for uncertainty)
        safety_stock = self.calculate_safety_stock(features, adjusted_daily_sales)
        
        # Predict total demand for the period
        predicted_demand = adjusted_daily_sales * prediction_days
        
        # Calculate required stock
        required_stock = predicted_demand + safety_stock
        
        # Calculate current stock status
        current_stock = product.inventory
        stock_deficit = max(0, required_stock - current_stock)
        stock_surplus = max(0, current_stock - required_stock)
        
        # Calculate stockout risk
        days_until_stockout = current_stock / adjusted_daily_sales if adjusted_daily_sales > 0 else float('inf')
        stockout_risk = self.calculate_stockout_risk(days_until_stockout, prediction_days)
        
        return {
            'prediction_days': prediction_days,
            'current_daily_sales': current_daily_sales,
            'adjusted_daily_sales': adjusted_daily_sales,
            'seasonal_multiplier': seasonal_multiplier,
            'predicted_demand': predicted_demand,
            'safety_stock': safety_stock,
            'required_stock': required_stock,
            'current_stock': current_stock,
            'stock_deficit': stock_deficit,
            'stock_surplus': stock_surplus,
            'days_until_stockout': days_until_stockout,
            'stockout_risk': stockout_risk,
            'recommendation': self.generate_stock_recommendation(stock_deficit, stock_surplus, stockout_risk)
        }
    
    def calculate_seasonal_multiplier(self, features):
        """Calculate seasonal adjustment multiplier"""
        base_multiplier = 1.0
        
        # Holiday season boost (Nov-Dec)
        if features['is_holiday_season']:
            base_multiplier *= 1.3
        
        # Summer boost for certain products
        if features['is_summer']:
            base_multiplier *= 1.1
        
        # Weekend effect
        base_multiplier *= 1.05  # Slight weekend boost
        
        # Growth trend adjustment
        growth_rate = features['sales_growth_rate']
        if growth_rate > 0.1:  # 10% growth
            base_multiplier *= (1 + growth_rate)
        elif growth_rate < -0.1:  # 10% decline
            base_multiplier *= (1 + growth_rate * 0.5)  # Less aggressive decline
        
        return base_multiplier
    
    def calculate_safety_stock(self, features, daily_sales):
        """Calculate safety stock based on demand variability"""
        # Higher safety stock for more volatile products
        volatility_7d = features['sales_volatility_7d']
        volatility_30d = features['sales_volatility_30d']
        
        # Use the higher volatility for safety stock calculation
        max_volatility = max(volatility_7d, volatility_30d)
        
        # Safety stock = 2 * standard deviation * sqrt(lead time)
        # Assuming 7-day lead time for reordering
        lead_time_days = 7
        safety_stock = 2 * max_volatility * np.sqrt(lead_time_days)
        
        # Minimum safety stock of 1 day of sales
        min_safety_stock = daily_sales
        safety_stock = max(safety_stock, min_safety_stock)
        
        return safety_stock
    
    def calculate_stockout_risk(self, days_until_stockout, prediction_days):
        """Calculate probability of stockout within prediction period"""
        if days_until_stockout >= prediction_days:
            return 0.0  # No risk
        elif days_until_stockout <= 0:
            return 1.0  # Already out of stock
        
        # Simple linear risk calculation
        risk = 1 - (days_until_stockout / prediction_days)
        return min(1.0, max(0.0, risk))
    
    def generate_stock_recommendation(self, stock_deficit, stock_surplus, stockout_risk):
        """Generate actionable stock recommendations"""
        if stockout_risk > 0.8:
            return {
                'action': 'urgent_restock',
                'message': 'URGENT: High risk of stockout. Restock immediately.',
                'priority': 'critical'
            }
        elif stockout_risk > 0.5:
            return {
                'action': 'restock',
                'message': 'Medium risk of stockout. Plan restocking soon.',
                'priority': 'high'
            }
        elif stock_deficit > 0:
            return {
                'action': 'restock',
                'message': f'Restock {int(stock_deficit)} units to meet predicted demand.',
                'priority': 'medium'
            }
        elif stock_surplus > stock_deficit * 2:
            return {
                'action': 'reduce_ordering',
                'message': 'High inventory levels. Consider reducing future orders.',
                'priority': 'low'
            }
        else:
            return {
                'action': 'maintain',
                'message': 'Inventory levels appear optimal.',
                'priority': 'low'
            }
    
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

    def train_simple_model(self):
        """Train a simple ML model with basic synthetic data for immediate use"""
        # Create simple synthetic training data
        X = []
        y = []
        
        # Generate synthetic scenarios
        for price in [50, 100, 200, 500, 1000]:
            for inventory in [10, 25, 50, 100, 200]:
                for sales_velocity in [1, 3, 5, 8, 12]:
                    for conversion_rate in [0.02, 0.05, 0.08, 0.12, 0.15]:
                        # Create feature vector
                        features = {
                            'price': price,
                            'inventory': inventory,
                            'sales_velocity_7d': sales_velocity,
                            'conversion_rate_7d': conversion_rate,
                            'traffic_avg_7d': sales_velocity / max(conversion_rate, 0.01),
                            'day_of_week': 3,  # Wednesday
                            'month': 7,  # July
                            'is_weekend': 0,
                            'price_elasticity': -0.8,
                            'competitor_price_ratio': 1.0,
                            'market_demand_index': 1.0
                        }
                        
                        # Simulate demand based on price elasticity
                        base_demand = sales_velocity
                        price_elasticity = -0.8
                        optimal_price = 200  # Assume optimal price is $200
                        price_ratio = price / optimal_price
                        simulated_demand = base_demand * (price_ratio ** price_elasticity)
                        
                        X.append(list(features.values()))
                        y.append(max(0, simulated_demand))
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.demand_model.fit(X_scaled, y)
        self.models_trained = True
        
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