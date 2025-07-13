from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ProductViewSet, SaleViewSet, TrafficViewSet

router = DefaultRouter()
router.register(r'products', ProductViewSet)
router.register(r'sales', SaleViewSet)
router.register(r'traffic', TrafficViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
] 