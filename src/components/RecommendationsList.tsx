import React from 'react';
import { Star, TrendingUp, Shield, AlertTriangle } from 'lucide-react';

const RecommendationsList: React.FC = () => {
  const recommendations = [
    {
      symbol: 'RELIANCE',
      action: 'Strong Buy',
      targetPrice: 3250,
      currentPrice: 2856,
      potential: '+13.8%',
      risk: 'Low',
      reason: 'Strong Q3 results, expanding digital business',
      icon: Star,
      actionColor: 'text-green-600 bg-green-100 dark:bg-green-900 dark:text-green-300'
    },
    {
      symbol: 'TCS',
      action: 'Buy',
      targetPrice: 4200,
      currentPrice: 3986,
      potential: '+5.4%',
      risk: 'Low',
      reason: 'Stable IT demand, strong order book',
      icon: TrendingUp,
      actionColor: 'text-emerald-600 bg-emerald-100 dark:bg-emerald-900 dark:text-emerald-300'
    },
    {
      symbol: 'HDFC Bank',
      action: 'Hold',
      targetPrice: 1750,
      currentPrice: 1698,
      potential: '+3.1%',
      risk: 'Medium',
      reason: 'Credit growth concerns, regulatory issues',
      icon: Shield,
      actionColor: 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900 dark:text-yellow-300'
    },
    {
      symbol: 'ONGC',
      action: 'Sell',
      targetPrice: 220,
      currentPrice: 245,
      potential: '-10.2%',
      risk: 'High',
      reason: 'Oil price volatility, declining reserves',
      icon: AlertTriangle,
      actionColor: 'text-red-600 bg-red-100 dark:bg-red-900 dark:text-red-300'
    }
  ];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        AI Recommendations
      </h3>
      
      <div className="space-y-4">
        {recommendations.map((rec) => {
          const Icon = rec.icon;
          return (
            <div
              key={rec.symbol}
              className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <Icon className="h-4 w-4 text-gray-500 dark:text-gray-400" />
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {rec.symbol}
                  </span>
                </div>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${rec.actionColor}`}>
                  {rec.action}
                </span>
              </div>
              
              <div className="grid grid-cols-2 gap-2 text-sm mb-2">
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Current:</span>
                  <span className="ml-1 font-medium text-gray-900 dark:text-white">
                    ₹{rec.currentPrice}
                  </span>
                </div>
                <div>
                  <span className="text-gray-500 dark:text-gray-400">Target:</span>
                  <span className="ml-1 font-medium text-gray-900 dark:text-white">
                    ₹{rec.targetPrice}
                  </span>
                </div>
              </div>
              
              <div className="flex items-center justify-between text-sm">
                <span className={`font-medium ${
                  rec.potential.startsWith('+') ? 'text-green-600' : 'text-red-600'
                }`}>
                  {rec.potential} potential
                </span>
                <span className={`px-2 py-1 rounded text-xs ${
                  rec.risk === 'Low' ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300' :
                  rec.risk === 'Medium' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300' :
                  'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
                }`}>
                  {rec.risk} Risk
                </span>
              </div>
              
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                {rec.reason}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default RecommendationsList;