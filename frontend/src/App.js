import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import './App.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const API_BASE_URL = 'http://localhost:8000/api';

function App() {
  const [products, setProducts] = useState([]);
  const [sales, setSales] = useState([]);
  const [traffic, setTraffic] = useState([]);
  const [loading, setLoading] = useState(false);
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [stockPredictions, setStockPredictions] = useState([]);
  const [activeTab, setActiveTab] = useState('dashboard');

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [productsRes, salesRes, trafficRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/products/`),
        axios.get(`${API_BASE_URL}/sales/`),
        axios.get(`${API_BASE_URL}/traffic/`)
      ]);
      
      setProducts(productsRes.data);
      setSales(salesRes.data);
      setTraffic(trafficRes.data);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const generateMockData = async () => {
    try {
      setLoading(true);
      await axios.post(`${API_BASE_URL}/products/generate_mock_data/`);
      await fetchData();
      alert('Mock data generated successfully!');
    } catch (error) {
      console.error('Error generating mock data:', error);
      alert('Error generating mock data');
    } finally {
      setLoading(false);
    }
  };

  const trainMLModels = async () => {
    try {
      setLoading(true);
      await axios.post(`${API_BASE_URL}/products/train_ml_models/`);
      alert('ML models trained successfully!');
    } catch (error) {
      console.error('Error training ML models:', error);
      alert('Error training ML models');
    } finally {
      setLoading(false);
    }
  };

  const optimizePrice = async (productId) => {
    try {
      setLoading(true);
      const response = await axios.post(`${API_BASE_URL}/products/${productId}/optimize_price/`);
      setOptimizationResult(response.data);
      await fetchData(); // Refresh data after optimization
      alert(`Price optimized for ${response.data.product_name}!`);
    } catch (error) {
      console.error('Error optimizing price:', error);
      alert('Error optimizing price');
    } finally {
      setLoading(false);
    }
  };

  const fetchStockPredictions = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/products/stock_dashboard/`);
      setStockPredictions(response.data.stock_predictions);
    } catch (error) {
      console.error('Error fetching stock predictions:', error);
    } finally {
      setLoading(false);
    }
  };

  const getSalesVelocity = (productId) => {
    const productSales = sales.filter(sale => sale.product === productId);
    const totalQuantity = productSales.reduce((sum, sale) => sum + sale.quantity, 0);
    return (totalQuantity / 7).toFixed(2);
  };

  const getConversionRate = (productId) => {
    const productSales = sales.filter(sale => sale.product === productId);
    const productTraffic = traffic.filter(t => t.product === productId);
    
    const totalSales = productSales.reduce((sum, sale) => sum + sale.quantity, 0);
    const totalVisits = productTraffic.reduce((sum, t) => sum + t.visits, 0);
    
    return totalVisits > 0 ? ((totalSales / totalVisits) * 100).toFixed(2) : 0;
  };

  const salesChartData = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    datasets: products.map(product => ({
      label: product.name,
      data: Array.from({ length: 7 }, (_, i) => {
        const daySales = sales.filter(sale => 
          sale.product === product.id && 
          new Date(sale.timestamp).getDay() === (i + 1) % 7
        );
        return daySales.reduce((sum, sale) => sum + sale.quantity, 0);
      }),
      borderColor: `hsl(${Math.random() * 360}, 70%, 50%)`,
      backgroundColor: `hsla(${Math.random() * 360}, 70%, 50%, 0.1)`,
      tension: 0.1
    }))
  };

  const trafficChartData = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    datasets: products.map(product => ({
      label: product.name,
      data: Array.from({ length: 7 }, (_, i) => {
        const dayTraffic = traffic.filter(t => 
          t.product === product.id && 
          new Date(t.timestamp).getDay() === (i + 1) % 7
        );
        return dayTraffic.reduce((sum, t) => sum + t.visits, 0);
      }),
      backgroundColor: `hsla(${Math.random() * 360}, 70%, 50%, 0.8)`,
    }))
  };

  const stockoutRiskChartData = {
    labels: stockPredictions.map(p => p.product_name),
    datasets: [{
      label: 'Stockout Risk (%)',
      data: stockPredictions.map(p => (p.stockout_risk * 100).toFixed(1)),
      backgroundColor: stockPredictions.map(p => {
        if (p.stockout_risk > 0.7) return 'rgba(255, 99, 132, 0.8)'; // Red for high risk
        if (p.stockout_risk > 0.3) return 'rgba(255, 205, 86, 0.8)'; // Yellow for medium risk
        return 'rgba(75, 192, 192, 0.8)'; // Green for low risk
      }),
      borderColor: stockPredictions.map(p => {
        if (p.stockout_risk > 0.7) return 'rgba(255, 99, 132, 1)';
        if (p.stockout_risk > 0.3) return 'rgba(255, 205, 86, 1)';
        return 'rgba(75, 192, 192, 1)';
      }),
      borderWidth: 1
    }]
  };

  const getRiskColor = (risk) => {
    if (risk > 0.7) return '#ff6b6b'; // Red
    if (risk > 0.3) return '#ffd93d'; // Yellow
    return '#6bcf7f'; // Green
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'critical': return '#ff6b6b';
      case 'high': return '#ffa726';
      case 'medium': return '#ffd93d';
      case 'low': return '#6bcf7f';
      default: return '#6c757d';
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>AI-Powered Pricing & Inventory Optimization System</h1>
        <div className="header-controls">
          <button 
            onClick={generateMockData} 
            disabled={loading}
            className="btn btn-primary"
          >
            {loading ? 'Generating...' : 'Generate Mock Data'}
          </button>
          <button 
            onClick={trainMLModels} 
            disabled={loading}
            className="btn btn-secondary"
          >
            {loading ? 'Training...' : 'Train ML Models'}
          </button>
          <button 
            onClick={fetchData} 
            disabled={loading}
            className="btn btn-secondary"
          >
            Refresh Data
          </button>
        </div>
      </header>

      <nav className="tab-navigation">
        <button 
          className={`tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          Dashboard
        </button>
        <button 
          className={`tab-btn ${activeTab === 'stock' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('stock');
            fetchStockPredictions();
          }}
        >
          Stock Predictions
        </button>
      </nav>

      <main className="App-main">
        {activeTab === 'dashboard' && (
          <>
            {optimizationResult && (
              <div className="optimization-result">
                <h3>Latest Price Optimization</h3>
                <div className="result-card">
                  <p><strong>Product:</strong> {optimizationResult.product_name}</p>
                  <p><strong>Old Price:</strong> ${optimizationResult.old_price}</p>
                  <p><strong>New Price:</strong> ${optimizationResult.new_price}</p>
                  <p><strong>Recommendation:</strong> {optimizationResult.recommendation}</p>
                  <p><strong>Sales Velocity:</strong> {optimizationResult.sales_velocity}</p>
                  <p><strong>Conversion Rate:</strong> {(optimizationResult.conversion_rate * 100).toFixed(2)}%</p>
                  {optimizationResult.confidence_score && (
                    <p><strong>ML Confidence:</strong> {(optimizationResult.confidence_score * 100).toFixed(1)}%</p>
                  )}
                </div>
              </div>
            )}

            <div className="charts-section">
              <div className="chart-container">
                <h3>Sales Velocity (Last 7 Days)</h3>
                <Line data={salesChartData} />
              </div>
              <div className="chart-container">
                <h3>Traffic (Last 7 Days)</h3>
                <Bar data={trafficChartData} />
              </div>
            </div>

            <div className="products-section">
              <h3>Products Dashboard</h3>
              <div className="products-grid">
                {products.map(product => (
                  <div key={product.id} className="product-card">
                    <h4>{product.name}</h4>
                    <p><strong>Category:</strong> {product.category}</p>
                    <p><strong>Price:</strong> ${product.price}</p>
                    <p><strong>Inventory:</strong> {product.inventory}</p>
                    <p><strong>Sales Velocity:</strong> {getSalesVelocity(product.id)}/day</p>
                    <p><strong>Conversion Rate:</strong> {getConversionRate(product.id)}%</p>
                    <button 
                      onClick={() => optimizePrice(product.id)}
                      disabled={loading}
                      className="btn btn-optimize"
                    >
                      Optimize Price
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {activeTab === 'stock' && (
          <div className="stock-predictions-section">
            <h3>Stock Predictions Dashboard</h3>
            
            <div className="stock-summary">
              <div className="summary-card">
                <h4>High Risk Products</h4>
                <span className="summary-number high-risk">
                  {stockPredictions.filter(p => p.stockout_risk > 0.7).length}
                </span>
              </div>
              <div className="summary-card">
                <h4>Medium Risk Products</h4>
                <span className="summary-number medium-risk">
                  {stockPredictions.filter(p => 0.3 < p.stockout_risk && p.stockout_risk <= 0.7).length}
                </span>
              </div>
              <div className="summary-card">
                <h4>Low Risk Products</h4>
                <span className="summary-number low-risk">
                  {stockPredictions.filter(p => p.stockout_risk <= 0.3).length}
                </span>
              </div>
            </div>

            <div className="stock-chart">
              <h4>Stockout Risk by Product</h4>
              <Bar data={stockoutRiskChartData} />
            </div>

            <div className="stock-predictions-grid">
              {stockPredictions.map(prediction => (
                <div key={prediction.product_id} className="stock-prediction-card">
                  <h4>{prediction.product_name}</h4>
                  <div className="stock-metrics">
                    <p><strong>Current Inventory:</strong> {prediction.current_inventory}</p>
                    <p><strong>Required Stock (30 days):</strong> {Math.round(prediction.required_stock)}</p>
                    <p><strong>Stock Deficit:</strong> {Math.round(prediction.stock_deficit)}</p>
                    <p><strong>Days Until Stockout:</strong> {prediction.days_until_stockout.toFixed(1)}</p>
                  </div>
                  <div className="risk-indicator">
                    <span 
                      className="risk-badge"
                      style={{ backgroundColor: getRiskColor(prediction.stockout_risk) }}
                    >
                      {(prediction.stockout_risk * 100).toFixed(1)}% Risk
                    </span>
                  </div>
                  <div className="recommendation">
                    <p><strong>Recommendation:</strong></p>
                    <p 
                      className="recommendation-text"
                      style={{ color: getPriorityColor(prediction.recommendation.priority) }}
                    >
                      {prediction.recommendation.message}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
