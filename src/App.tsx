import React, { useState } from 'react';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import StockSearch from './components/StockSearch';
import Portfolio from './components/Portfolio';
import MarketOverview from './components/MarketOverview';
import { ThemeProvider } from './context/ThemeContext';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  const renderActiveComponent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard />;
      case 'search':
        return <StockSearch />;
      case 'portfolio':
        return <Portfolio />;
      case 'market':
        return <MarketOverview />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <ThemeProvider>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-300">
        <Header activeTab={activeTab} setActiveTab={setActiveTab} />
        <main className="container mx-auto px-4 py-6">
          {renderActiveComponent()}
        </main>
      </div>
    </ThemeProvider>
  );
}

export default App;