from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
from datetime import timedelta
import random
from .models import Product, Sale, Traffic
from .serializers import ProductSerializer, SaleSerializer, TrafficSerializer
from .ml_models import ml_optimizer

# Create your views here.

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

    @action(detail=False, methods=['post'])
    def generate_mock_data(self, request):
        """Generate mock products, sales, and traffic data with diverse scenarios"""
        # Create mock products with different scenarios
        products_data = [
            # High demand, low inventory scenario
            {'name': 'Laptop Pro', 'category': 'Electronics', 'price': 999.99, 'inventory': 15, 'scenario': 'high_demand_low_inventory'},
            # High traffic, low conversion scenario
            {'name': 'Smartphone X', 'category': 'Electronics', 'price': 699.99, 'inventory': 80, 'scenario': 'high_traffic_low_conversion'},
            # Slow mover scenario
            {'name': 'Yoga Mat', 'category': 'Sports', 'price': 29.99, 'inventory': 150, 'scenario': 'slow_mover'},
            # Very high demand scenario
            {'name': 'Gaming Console', 'category': 'Electronics', 'price': 499.99, 'inventory': 8, 'scenario': 'very_high_demand'},
            # High conversion rate scenario
            {'name': 'Wireless Headphones', 'category': 'Electronics', 'price': 199.99, 'inventory': 45, 'scenario': 'high_conversion'},
            # Very low inventory scenario
            {'name': 'Coffee Maker', 'category': 'Home', 'price': 89.99, 'inventory': 5, 'scenario': 'very_low_inventory'},
            # Balanced scenario
            {'name': 'Running Shoes', 'category': 'Sports', 'price': 129.99, 'inventory': 60, 'scenario': 'balanced'},
            # Another slow mover
            {'name': 'Bluetooth Speaker', 'category': 'Electronics', 'price': 79.99, 'inventory': 120, 'scenario': 'slow_mover'},
        ]
        
        products = []
        for data in products_data:
            product, created = Product.objects.get_or_create(
                name=data['name'],
                defaults={
                    'name': data['name'],
                    'category': data['category'],
                    'price': data['price'],
                    'inventory': data['inventory']
                }
            )
            products.append(product)
        
        # Generate mock sales and traffic for the last 30 days with scenario-based patterns
        for product in products:
            product_data = next(p for p in products_data if p['name'] == product.name)
            scenario = product_data['scenario']
            
            for i in range(30):
                date = timezone.now() - timedelta(days=i)
                
                # Generate sales based on scenario
                if scenario == 'high_demand_low_inventory':
                    base_sales = random.randint(8, 15)  # High sales
                elif scenario == 'high_traffic_low_conversion':
                    base_sales = random.randint(1, 3)   # Low sales despite high traffic
                elif scenario == 'slow_mover':
                    base_sales = random.randint(0, 1)   # Very low sales
                elif scenario == 'very_high_demand':
                    base_sales = random.randint(12, 20) # Very high sales
                elif scenario == 'high_conversion':
                    base_sales = random.randint(6, 10)  # Good sales
                elif scenario == 'very_low_inventory':
                    base_sales = random.randint(5, 8)   # Moderate sales but low inventory
                elif scenario == 'balanced':
                    base_sales = random.randint(3, 6)   # Balanced sales
                else:
                    base_sales = random.randint(2, 5)   # Default
                
                # Add weekend effect
                if date.weekday() >= 5:  # Weekend
                    base_sales = int(base_sales * 1.3)
                
                for _ in range(base_sales):
                    Sale.objects.create(
                        product=product,
                        quantity=random.randint(1, 3),
                        timestamp=date
                    )
                
                # Generate traffic based on scenario
                if scenario == 'high_traffic_low_conversion':
                    base_traffic = random.randint(100, 200)  # High traffic
                elif scenario == 'high_demand_low_inventory':
                    base_traffic = random.randint(80, 150)   # Good traffic
                elif scenario == 'slow_mover':
                    base_traffic = random.randint(20, 50)    # Low traffic
                elif scenario == 'very_high_demand':
                    base_traffic = random.randint(150, 250)  # Very high traffic
                elif scenario == 'high_conversion':
                    base_traffic = random.randint(60, 100)   # Moderate traffic
                elif scenario == 'very_low_inventory':
                    base_traffic = random.randint(70, 120)   # Good traffic
                elif scenario == 'balanced':
                    base_traffic = random.randint(50, 90)    # Balanced traffic
                else:
                    base_traffic = random.randint(40, 80)    # Default
                
                # Add weekend effect
                if date.weekday() >= 5:  # Weekend
                    base_traffic = int(base_traffic * 1.2)
                
                Traffic.objects.create(
                    product=product,
                    visits=base_traffic,
                    timestamp=date
                )
        
        return Response({
            'message': 'Mock data generated successfully with diverse scenarios',
            'scenarios_created': [p['scenario'] for p in products_data]
        }, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['post'])
    def train_ml_models(self, request):
        """Train ML models with existing data or simple synthetic data"""
        try:
            # Get all data
            products = Product.objects.all()
            sales_data = list(Sale.objects.values('product', 'quantity', 'timestamp'))
            traffic_data = list(Traffic.objects.values('product', 'visits', 'timestamp'))
            
            # Try to generate training data from existing data first
            training_data = ml_optimizer.generate_training_data(products, sales_data, traffic_data)
            
            if training_data and len(training_data) > 10:
                # Train with real data if we have enough
                ml_optimizer.train_demand_model(training_data)
                ml_optimizer.models_trained = True
                
                return Response({
                    'message': 'ML models trained successfully with real data',
                    'training_samples': len(training_data),
                    'models_trained': ['Random Forest Demand Predictor', 'Price Elasticity Model']
                }, status=status.HTTP_200_OK)
            else:
                # Fallback to simple synthetic training
                ml_optimizer.train_simple_model()
                
                return Response({
                    'message': 'ML models trained with synthetic data for immediate use',
                    'training_samples': 'Synthetic data generated',
                    'models_trained': ['Random Forest Demand Predictor (Synthetic)', 'Price Elasticity Model'],
                    'note': 'Generate more mock data for better ML predictions'
                }, status=status.HTTP_200_OK)
                
        except Exception as e:
            return Response({
                'message': f'Error training ML models: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=True, methods=['post'])
    def optimize_price(self, request, pk=None):
        """Optimize price using both rule-based and ML approaches"""
        product = self.get_object()
        
        # Get sales and traffic data
        sales_data = list(Sale.objects.filter(product=product).values('quantity', 'timestamp'))
        traffic_data = list(Traffic.objects.filter(product=product).values('visits', 'timestamp'))
        
        # Try ML-based optimization first
        if ml_optimizer.models_trained:
            ml_result = ml_optimizer.calculate_optimal_price(product, sales_data, traffic_data)
            
            # Update product price
            product.price = ml_result['optimal_price']
            product.save()
            
            return Response({
                'product_id': product.id,
                'product_name': product.name,
                'old_price': float(request.data.get('current_price', product.price)),
                'new_price': ml_result['optimal_price'],
                'recommendation': f"ML-optimized price: {ml_result['ml_model_used']}",
                'confidence_score': ml_result['confidence_score'],
                'predicted_revenue': ml_result['predicted_revenue'],
                'current_demand': ml_result['current_demand'],
                'optimization_method': 'ML-based'
            })
        else:
            # Fallback to rule-based optimization
            return self.rule_based_optimize_price(product, sales_data, traffic_data)

    @action(detail=True, methods=['get'])
    def predict_stock(self, request, pk=None):
        """Predict stock requirements for the next N days"""
        product = self.get_object()
        
        # Get prediction period from query params (default 30 days)
        prediction_days = int(request.query_params.get('days', 30))
        
        # Get sales and traffic data
        sales_data = list(Sale.objects.filter(product=product).values('quantity', 'timestamp'))
        traffic_data = list(Traffic.objects.filter(product=product).values('visits', 'timestamp'))
        
        # Predict stock requirements
        stock_prediction = ml_optimizer.predict_stock_requirements(
            product, sales_data, traffic_data, prediction_days
        )
        
        return Response({
            'product_id': product.id,
            'product_name': product.name,
            'prediction': stock_prediction
        })

    @action(detail=False, methods=['get'])
    def stock_dashboard(self, request):
        """Get stock predictions for all products"""
        products = Product.objects.all()
        stock_predictions = []
        
        for product in products:
            sales_data = list(Sale.objects.filter(product=product).values('quantity', 'timestamp'))
            traffic_data = list(Traffic.objects.filter(product=product).values('visits', 'timestamp'))
            
            # Get 30-day prediction
            prediction = ml_optimizer.predict_stock_requirements(product, sales_data, traffic_data, 30)
            
            stock_predictions.append({
                'product_id': product.id,
                'product_name': product.name,
                'current_inventory': product.inventory,
                'stockout_risk': prediction['stockout_risk'],
                'days_until_stockout': prediction['days_until_stockout'],
                'required_stock': prediction['required_stock'],
                'stock_deficit': prediction['stock_deficit'],
                'recommendation': prediction['recommendation']
            })
        
        # Sort by stockout risk (highest first)
        stock_predictions.sort(key=lambda x: x['stockout_risk'], reverse=True)
        
        return Response({
            'stock_predictions': stock_predictions,
            'summary': {
                'total_products': len(products),
                'high_risk_products': len([p for p in stock_predictions if p['stockout_risk'] > 0.7]),
                'medium_risk_products': len([p for p in stock_predictions if 0.3 < p['stockout_risk'] <= 0.7]),
                'low_risk_products': len([p for p in stock_predictions if p['stockout_risk'] <= 0.3])
            }
        })

    def rule_based_optimize_price(self, product, sales_data, traffic_data):
        """Original rule-based price optimization with improved thresholds"""
        # Calculate sales velocity (sales per day in last 7 days)
        week_ago = timezone.now() - timedelta(days=7)
        recent_sales = [s for s in sales_data if s['timestamp'] >= week_ago]
        recent_traffic = [t for t in traffic_data if t['timestamp'] >= week_ago]
        
        total_sales = sum(sale['quantity'] for sale in recent_sales)
        total_visits = sum(traffic['visits'] for traffic in recent_traffic)
        
        sales_velocity = total_sales / 7
        conversion_rate = total_sales / total_visits if total_visits > 0 else 0
        
        # Improved pricing logic with more sensitive thresholds
        current_price = float(product.price)
        new_price = current_price
        recommendation = "Keep price same"
        
        # More sensitive thresholds for better recommendations
        if sales_velocity > 3 and product.inventory < 30:  # Lowered threshold from 5 to 3, inventory from 20 to 30
            new_price = current_price * 1.15  # Increased from 10% to 15%
            recommendation = "Increase price due to high demand and low inventory"
        elif conversion_rate < 0.08 and total_visits > 30:  # Increased threshold from 0.05 to 0.08, visits from 50 to 30
            new_price = current_price * 0.85  # Increased discount from 10% to 15%
            recommendation = "Lower price due to high traffic but low conversion"
        elif sales_velocity < 2 and product.inventory > 40:  # Increased threshold from 1 to 2, inventory from 50 to 40
            new_price = current_price * 0.65  # Increased discount from 30% to 35%
            recommendation = "Trigger flash sale for slow moving product"
        elif sales_velocity > 5:  # High demand regardless of inventory
            new_price = current_price * 1.1
            recommendation = "Increase price due to very high demand"
        elif conversion_rate > 0.15:  # High conversion rate
            new_price = current_price * 1.05
            recommendation = "Slight price increase due to high conversion rate"
        elif product.inventory < 10:  # Very low inventory
            new_price = current_price * 1.2
            recommendation = "Significant price increase due to very low inventory"
        
        # Update product price
        product.price = new_price
        product.save()
        
        return Response({
            'product_id': product.id,
            'product_name': product.name,
            'old_price': current_price,
            'new_price': new_price,
            'recommendation': recommendation,
            'sales_velocity': sales_velocity,
            'conversion_rate': conversion_rate,
            'total_visits': total_visits,
            'inventory': product.inventory,
            'optimization_method': 'Rule-based (Improved)'
        })

    @action(detail=True, methods=['get'])
    def price_analysis(self, request, pk=None):
        """Get detailed price analysis for a product"""
        product = self.get_object()
        
        # Get historical data
        sales_data = list(Sale.objects.filter(product=product).values('quantity', 'timestamp'))
        traffic_data = list(Traffic.objects.filter(product=product).values('visits', 'timestamp'))
        
        # Calculate various metrics
        week_ago = timezone.now() - timedelta(days=7)
        month_ago = timezone.now() - timedelta(days=30)
        
        recent_sales = [s for s in sales_data if s['timestamp'] >= week_ago]
        recent_traffic = [t for t in traffic_data if t['timestamp'] >= week_ago]
        
        monthly_sales = [s for s in sales_data if s['timestamp'] >= month_ago]
        monthly_traffic = [t for t in traffic_data if t['timestamp'] >= month_ago]
        
        # Calculate metrics
        total_sales_week = sum(sale['quantity'] for sale in recent_sales)
        total_visits_week = sum(traffic['visits'] for traffic in recent_traffic)
        total_sales_month = sum(sale['quantity'] for sale in monthly_sales)
        total_visits_month = sum(traffic['visits'] for traffic in monthly_traffic)
        
        sales_velocity_week = total_sales_week / 7
        sales_velocity_month = total_sales_month / 30
        conversion_rate_week = total_sales_week / total_visits_week if total_visits_week > 0 else 0
        conversion_rate_month = total_sales_month / total_visits_month if total_visits_month > 0 else 0
        
        # ML predictions if available
        ml_insights = {}
        if ml_optimizer.models_trained:
            features = ml_optimizer.prepare_features(product, sales_data, traffic_data)
            predicted_demand = ml_optimizer.predict_demand(features)
            confidence_score = ml_optimizer.calculate_confidence_score(features)
            
            ml_insights = {
                'predicted_demand': predicted_demand,
                'confidence_score': confidence_score,
                'price_elasticity': features.get('price_elasticity', -1.0),
                'optimal_price_range': {
                    'min': float(product.price) * 0.8,
                    'max': float(product.price) * 1.2
                }
            }
        
        return Response({
            'product_id': product.id,
            'product_name': product.name,
            'current_price': float(product.price),
            'inventory': product.inventory,
            'metrics': {
                'sales_velocity_week': sales_velocity_week,
                'sales_velocity_month': sales_velocity_month,
                'conversion_rate_week': conversion_rate_week,
                'conversion_rate_month': conversion_rate_month,
                'total_visits_week': total_visits_week,
                'total_visits_month': total_visits_month,
                'total_sales_week': total_sales_week,
                'total_sales_month': total_sales_month
            },
            'ml_insights': ml_insights,
            'recommendations': self.generate_recommendations(
                sales_velocity_week, conversion_rate_week, product.inventory, total_visits_week
            )
        })

    def generate_recommendations(self, sales_velocity, conversion_rate, inventory, total_visits):
        """Generate actionable recommendations"""
        recommendations = []
        
        if sales_velocity > 5 and inventory < 20:
            recommendations.append({
                'type': 'price_increase',
                'message': 'High demand with low inventory - consider price increase',
                'priority': 'high'
            })
        
        if conversion_rate < 0.05 and total_visits > 50:
            recommendations.append({
                'type': 'price_decrease',
                'message': 'Low conversion rate despite high traffic - consider price decrease',
                'priority': 'medium'
            })
        
        if sales_velocity < 1 and inventory > 50:
            recommendations.append({
                'type': 'flash_sale',
                'message': 'Slow moving product with high inventory - consider flash sale',
                'priority': 'medium'
            })
        
        if not recommendations:
            recommendations.append({
                'type': 'maintain',
                'message': 'Current pricing strategy appears optimal',
                'priority': 'low'
            })
        
        return recommendations

class SaleViewSet(viewsets.ModelViewSet):
    queryset = Sale.objects.all()
    serializer_class = SaleSerializer

class TrafficViewSet(viewsets.ModelViewSet):
    queryset = Traffic.objects.all()
    serializer_class = TrafficSerializer
