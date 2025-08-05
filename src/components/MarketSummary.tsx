import React from 'react';
import { TrendingUp, TrendingDown, Activity, DollarSign } from 'lucide-react';

const MarketSummary: React.FC = () => {
  const marketData = [
    {
      name: 'Nifty 50',
      value: '28,956.73',
      change: '+245.87',
      changePercent: '+0.85%',
      isPositive: true,
      icon: TrendingUp,
    },
    {
      name: 'Sensex',
      value: '95,842.16',
      change: '+823.12',
      changePercent: '+0.86%',
      isPositive: true,
      icon: TrendingUp,
    },
    {
      name: 'Bank Nifty',
      value: '60,234.45',
      change: '-123.67',
      changePercent: '-0.20%',
      isPositive: false,
      icon: TrendingDown,
    },
    {
      name: 'Market Volume',
      value: '₹4.2T',
      change: '+8.5%',
      changePercent: 'vs yesterday',
      isPositive: true,
      icon: Activity,
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {marketData.map((item) => {
        const Icon = item.icon;
        return (
          <div
            key={item.name}
            className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-gray-700"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">
                {item.name}
              </h3>
              <Icon className={`h-5 w-5 ${
                item.isPositive ? 'text-green-500' : 'text-red-500'
              }`} />
            </div>
            
            <div className="space-y-2">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {item.value}
              </div>
              <div className={`flex items-center space-x-1 text-sm font-medium ${
                item.isPositive ? 'text-green-600' : 'text-red-600'
              }`}>
                <span>{item.change}</span>
                <span className="text-xs">({item.changePercent})</span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default MarketSummary;