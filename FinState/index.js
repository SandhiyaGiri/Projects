require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const path = require('path');

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

const ALPHA_VANTAGE_KEY = '7OCZBVRR8RCVRPVQ';

// Indices and stocks per country (ETF proxies for indices)
const countryData = {
  US: {
    indices: [
      { symbol: 'SPY', name: 'S&P 500 (SPY ETF)' },
      { symbol: 'QQQ', name: 'NASDAQ 100 (QQQ ETF)' },
      { symbol: 'DIA', name: 'Dow Jones (DIA ETF)' }
    ],
    stocks: [
      { symbol: 'AAPL', name: 'Apple' },
      { symbol: 'MSFT', name: 'Microsoft' },
      { symbol: 'GOOGL', name: 'Alphabet' },
      { symbol: 'AMZN', name: 'Amazon' },
      { symbol: 'TSLA', name: 'Tesla' }
    ]
  },
  IN: {
    indices: [
      { symbol: 'NIFTYBEES.BSE', name: 'NIFTY 50 (ETF)' }
    ],
    stocks: [
      { symbol: 'RELIANCE.BSE', name: 'Reliance' },
      { symbol: 'TCS.BSE', name: 'TCS' },
      { symbol: 'INFY.BSE', name: 'Infosys' },
      { symbol: 'HDFCBANK.BSE', name: 'HDFC Bank' },
      { symbol: 'ICICIBANK.BSE', name: 'ICICI Bank' }
    ]
  },
  UK: {
    indices: [
      { symbol: 'ISF.L', name: 'FTSE 100 (ETF)' }
    ],
    stocks: [
      { symbol: 'HSBA.L', name: 'HSBC' },
      { symbol: 'BP.L', name: 'BP' },
      { symbol: 'VOD.L', name: 'Vodafone' },
      { symbol: 'GSK.L', name: 'GSK' },
      { symbol: 'RIO.L', name: 'Rio Tinto' }
    ]
  },
  JP: {
    indices: [
      { symbol: '1321.T', name: 'Nikkei 225 (ETF)' }
    ],
    stocks: [
      { symbol: '7203.T', name: 'Toyota' },
      { symbol: '6758.T', name: 'Sony' },
      { symbol: '9984.T', name: 'SoftBank' },
      { symbol: '8306.T', name: 'Mitsubishi UFJ' },
      { symbol: '7267.T', name: 'Honda' }
    ]
  }
};

// Endpoint to get indices for a country
app.get('/api/indices', (req, res) => {
  const { country } = req.query;
  if (!countryData[country]) return res.json([]);
  res.json(countryData[country].indices);
});

// Endpoint to get stocks for a country
app.get('/api/stocks', (req, res) => {
  const { country } = req.query;
  if (!countryData[country]) return res.json([]);
  res.json(countryData[country].stocks);
});

// Helper: Fetch daily time series from Alpha Vantage
async function fetchAlphaVantage(symbol) {
  const url = `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=${symbol}&outputsize=compact&apikey=${ALPHA_VANTAGE_KEY}`;
  try {
    const response = await axios.get(url, { headers: { 'User-Agent': 'request' } });
    return response.data;
  } catch (error) {
    console.error('Alpha Vantage error:', error?.response?.data || error.message);
    throw error;
  }
}

// Comparison endpoint
app.get('/api/compare', async (req, res) => {
  const { symbol1, symbol2, startDate, endDate } = req.query;
  if (!symbol1 || !symbol2) {
    return res.status(400).json({ error: 'Both symbol1 and symbol2 are required.' });
  }
  try {
    const [data1, data2] = await Promise.all([
      fetchAlphaVantage(symbol1),
      fetchAlphaVantage(symbol2)
    ]);
    // Filter by date range if provided
    function filterByDateRange(ts) {
      if (!ts) return ts;
      return Object.fromEntries(Object.entries(ts).filter(([date]) => {
        return (!startDate || date >= startDate) && (!endDate || date <= endDate);
      }));
    }
    if (data1['Time Series (Daily)']) {
      data1['Time Series (Daily)'] = filterByDateRange(data1['Time Series (Daily)']);
    }
    if (data2['Time Series (Daily)']) {
      data2['Time Series (Daily)'] = filterByDateRange(data2['Time Series (Daily)']);
    }
    res.json({
      [symbol1]: data1,
      [symbol2]: data2
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch comparison data' });
  }
});

// Serve the main page
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
}); 
