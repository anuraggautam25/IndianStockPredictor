import React from 'react';
import { TrendingUp, TrendingDown, AlertCircle } from 'lucide-react';
import { Stock } from '../types/stock';

interface StockCardProps {
  stock: Stock;
}

const StockCard: React.FC<StockCardProps> = ({ stock }) => {
  const isPositive = stock.change >= 0;
  const recommendation = stock.recommendation;

  const getRecommendationColor = (rec: string) => {
    switch (rec) {
      case 'Strong Buy':
        return 'text-green-600 bg-green-100 dark:bg-green-900 dark:text-green-300';
      case 'Buy':
        return 'text-emerald-600 bg-emerald-100 dark:bg-emerald-900 dark:text-emerald-300';
      case 'Hold':
        return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900 dark:text-yellow-300';
      case 'Sell':
        return 'text-red-600 bg-red-100 dark:bg-red-900 dark:text-red-300';
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-700 dark:text-gray-300';
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-200 dark:border-gray-700 p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-bold text-gray-900 dark:text-white">
            {stock.symbol}
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            {stock.name}
          </p>
        </div>
        {isPositive ? (
          <TrendingUp className="h-6 w-6 text-green-500" />
        ) : (
          <TrendingDown className="h-6 w-6 text-red-500" />
        )}
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-2xl font-bold text-gray-900 dark:text-white">
            ₹{stock.price.toFixed(2)}
          </span>
          <div className={`flex items-center space-x-1 ${
            isPositive ? 'text-green-600' : 'text-red-600'
          }`}>
            <span className="font-semibold">
              {isPositive ? '+' : ''}{stock.change.toFixed(2)}
            </span>
            <span className="text-sm">
              ({stock.changePercent.toFixed(2)}%)
            </span>
          </div>
        </div>

        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600 dark:text-gray-400">Volume:</span>
          <span className="font-medium text-gray-900 dark:text-white">
            {stock.volume.toLocaleString()}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600 dark:text-gray-400">
            Prediction:
          </span>
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRecommendationColor(recommendation)}`}>
            {recommendation}
          </span>
        </div>

        <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-2 text-xs text-gray-500 dark:text-gray-400">
            <AlertCircle className="h-3 w-3" />
            <span>Target: ₹{stock.targetPrice.toFixed(2)}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StockCard;