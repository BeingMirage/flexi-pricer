from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
from datetime import timedelta
import random
from .models import Product, Sale, Traffic
from .serializers import ProductSerializer, SaleSerializer, TrafficSerializer

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
        
        # Generate mock sales and traffic for the last 7 days
        for product in products:
            for i in range(7):
                date = timezone.now() - timedelta(days=i)
                
                # Generate random sales
                sales_count = random.randint(0, 10)
                for _ in range(sales_count):
                    Sale.objects.create(
                        product=product,
                        quantity=random.randint(1, 3),
                        timestamp=date
                    )
                
                # Generate random traffic
                Traffic.objects.create(
                    product=product,
                    visits=random.randint(10, 100),
                    timestamp=date
                )
        
        return Response({'message': 'Mock data generated successfully'}, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['post'])
    def optimize_price(self, request, pk=None):
        """Optimize price based on sales velocity and traffic conversion"""
        product = self.get_object()
        
        # Calculate sales velocity (sales per day in last 7 days)
        week_ago = timezone.now() - timedelta(days=7)
        recent_sales = Sale.objects.filter(
            product=product,
            timestamp__gte=week_ago
        )
        total_sales = sum(sale.quantity for sale in recent_sales)
        sales_velocity = total_sales / 7
        
        # Calculate traffic conversion
        recent_traffic = Traffic.objects.filter(
            product=product,
            timestamp__gte=week_ago
        )
        total_visits = sum(traffic.visits for traffic in recent_traffic)
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
            'inventory': product.inventory
        })

class SaleViewSet(viewsets.ModelViewSet):
    queryset = Sale.objects.all()
    serializer_class = SaleSerializer

class TrafficViewSet(viewsets.ModelViewSet):
    queryset = Traffic.objects.all()
    serializer_class = TrafficSerializer
