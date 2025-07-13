import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line, Bar } from 'react-chartjs-2';
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
  Legend
);

const API_BASE_URL = 'http://localhost:8000/api';

function App() {
  const [products, setProducts] = useState([]);
  const [sales, setSales] = useState([]);
  const [traffic, setTraffic] = useState([]);
  const [loading, setLoading] = useState(false);
  const [optimizationResult, setOptimizationResult] = useState(null);

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

  return (
    <div className="App">
      <header className="App-header">
        <h1>Real-Time Pricing Optimization System</h1>
        <div className="header-controls">
          <button 
            onClick={generateMockData} 
            disabled={loading}
            className="btn btn-primary"
          >
            {loading ? 'Generating...' : 'Generate Mock Data'}
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

      <main className="App-main">
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
      </main>
    </div>
  );
}

export default App;
