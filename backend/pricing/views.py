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
        """Generate mock products, sales, and traffic data"""
        # Create mock products
        products_data = [
            {'name': 'Laptop Pro', 'category': 'Electronics', 'price': 999.99, 'inventory': 50},
            {'name': 'Smartphone X', 'category': 'Electronics', 'price': 699.99, 'inventory': 100},
            {'name': 'Wireless Headphones', 'category': 'Electronics', 'price': 199.99, 'inventory': 75},
            {'name': 'Gaming Console', 'category': 'Electronics', 'price': 499.99, 'inventory': 25},
            {'name': 'Coffee Maker', 'category': 'Home', 'price': 89.99, 'inventory': 30},
            {'name': 'Running Shoes', 'category': 'Sports', 'price': 129.99, 'inventory': 60},
            {'name': 'Yoga Mat', 'category': 'Sports', 'price': 29.99, 'inventory': 120},
            {'name': 'Bluetooth Speaker', 'category': 'Electronics', 'price': 79.99, 'inventory': 45},
        ]
        
        products = []
        for data in products_data:
            product, created = Product.objects.get_or_create(
                name=data['name'],
                defaults=data
            )
            products.append(product)
        
        # Generate mock sales and traffic for the last 30 days (more data for ML)
        for product in products:
            for i in range(30):
                date = timezone.now() - timedelta(days=i)
                
                # Generate random sales with more realistic patterns
                base_sales = random.randint(0, 8)
                # Add weekend effect
                if date.weekday() >= 5:  # Weekend
                    base_sales = int(base_sales * 1.5)
                
                for _ in range(base_sales):
                    Sale.objects.create(
                        product=product,
                        quantity=random.randint(1, 3),
                        timestamp=date
                    )
                
                # Generate random traffic with realistic patterns
                base_traffic = random.randint(20, 150)
                # Add weekend effect
                if date.weekday() >= 5:  # Weekend
                    base_traffic = int(base_traffic * 1.3)
                
                Traffic.objects.create(
                    product=product,
                    visits=base_traffic,
                    timestamp=date
                )
        
        return Response({'message': 'Mock data generated successfully'}, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['post'])
    def train_ml_models(self, request):
        """Train ML models with existing data"""
        try:
            # Get all data
            products = Product.objects.all()
            sales_data = list(Sale.objects.values('product', 'quantity', 'timestamp'))
            traffic_data = list(Traffic.objects.values('product', 'visits', 'timestamp'))
            
            # Generate training data
            training_data = ml_optimizer.generate_training_data(products, sales_data, traffic_data)
            
            # Train the model
            if training_data:
                ml_optimizer.train_demand_model(training_data)
                ml_optimizer.models_trained = True
                
                return Response({
                    'message': 'ML models trained successfully',
                    'training_samples': len(training_data),
                    'models_trained': ['Random Forest Demand Predictor', 'Price Elasticity Model']
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    'message': 'Insufficient data for training. Generate more mock data first.'
                }, status=status.HTTP_400_BAD_REQUEST)
                
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

    def rule_based_optimize_price(self, product, sales_data, traffic_data):
        """Original rule-based price optimization"""
        # Calculate sales velocity (sales per day in last 7 days)
        week_ago = timezone.now() - timedelta(days=7)
        recent_sales = [s for s in sales_data if s['timestamp'] >= week_ago]
        recent_traffic = [t for t in traffic_data if t['timestamp'] >= week_ago]
        
        total_sales = sum(sale['quantity'] for sale in recent_sales)
        total_visits = sum(traffic['visits'] for traffic in recent_traffic)
        
        sales_velocity = total_sales / 7
        conversion_rate = total_sales / total_visits if total_visits > 0 else 0
        
        # Pricing logic based on flowchart
        current_price = float(product.price)
        new_price = current_price
        recommendation = "Keep price same"
        
        if sales_velocity > 5 and product.inventory < 20:  # High demand + low inventory
            new_price = current_price * 1.1  # Increase by 10%
            recommendation = "Increase price due to high demand and low inventory"
        elif conversion_rate < 0.05 and total_visits > 50:  # High traffic + low conversion
            new_price = current_price * 0.9  # Decrease by 10%
            recommendation = "Lower price due to high traffic but low conversion"
        elif sales_velocity < 1 and product.inventory > 50:  # Slow mover
            new_price = current_price * 0.7  # Flash sale - 30% off
            recommendation = "Trigger flash sale for slow moving product"
        
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
            'optimization_method': 'Rule-based'
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
