import React from 'react';
import StockCard from './StockCard';
import PredictionChart from './PredictionChart';
import MarketSummary from './MarketSummary';
import RecommendationsList from './RecommendationsList';
import { popularStocks } from '../data/stockData';

const Dashboard: React.FC = () => {
  const topStocks = popularStocks.slice(0, 6);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          Market Dashboard
        </h2>
        <div className="text-sm text-gray-500 dark:text-gray-400">
          Last updated: {new Date().toLocaleTimeString()}
        </div>
      </div>

      <MarketSummary />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Market Prediction Analysis
            </h3>
            <PredictionChart />
          </div>
        </div>

        <div className="space-y-6">
          <RecommendationsList />
        </div>
      </div>

      <div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Top Stocks to Watch
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {topStocks.map((stock) => (
            <StockCard key={stock.symbol} stock={stock} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;