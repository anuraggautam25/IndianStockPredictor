import React from 'react';
import { Globe, TrendingUp, TrendingDown, Activity, DollarSign } from 'lucide-react';

const MarketOverview: React.FC = () => {
  const sectorPerformance = [
    { name: 'Technology', change: 2.45, color: 'text-green-600' },
    { name: 'Banking', change: -0.85, color: 'text-red-600' },
    { name: 'Pharmaceuticals', change: 1.78, color: 'text-green-600' },
    { name: 'Energy', change: -1.23, color: 'text-red-600' },
    { name: 'FMCG', change: 0.95, color: 'text-green-600' },
    { name: 'Metals', change: -2.15, color: 'text-red-600' },
    { name: 'Auto', change: 1.34, color: 'text-green-600' },
    { name: 'Realty', change: 3.67, color: 'text-green-600' },
  ];

  const marketStats = [
    {
      title: 'Market Cap',
      value: '₹425.6T',
      change: '+2.1%',
      icon: DollarSign,
      positive: true
    },
    {
      title: 'FII Net',
      value: '₹3,456 Cr',
      change: 'Inflow',
      icon: TrendingUp,
      positive: true
    },
    {
      title: 'DII Net',
      value: '₹2,789 Cr',
      change: 'Inflow',
      icon: TrendingUp,
      positive: true
    },
    {
      title: 'Put/Call Ratio',
      value: '0.87',
      change: 'Bullish',
      icon: Activity,
      positive: true
    }
  ];

  const globalMarkets = [
    { name: 'Dow Jones', value: '43,456.78', change: '+0.65%', positive: true },
    { name: 'NASDAQ', value: '17,234.56', change: '+1.23%', positive: true },
    { name: 'Nikkei', value: '28,956.43', change: '-0.45%', positive: false },
    { name: 'Hang Seng', value: '19,876.32', change: '+0.89%', positive: true },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          Market Overview
        </h2>
        <div className="flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400">
          <Globe className="h-4 w-4" />
          <span>Live Market Data</span>
        </div>
      </div>

      {/* Market Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {marketStats.map((stat) => {
          const Icon = stat.icon;
          return (
            <div
              key={stat.title}
              className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-gray-700"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  {stat.title}
                </h3>
                <Icon className={`h-5 w-5 ${
                  stat.positive ? 'text-green-500' : 'text-red-500'
                }`} />
              </div>
              <div className="space-y-2">
                <div className="text-2xl font-bold text-gray-900 dark:text-white">
                  {stat.value}
                </div>
                <div className={`text-sm font-medium ${
                  stat.positive ? 'text-green-600' : 'text-red-600'
                }`}>
                  {stat.change}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sector Performance */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Sector Performance
          </h3>
          <div className="space-y-3">
            {sectorPerformance.map((sector) => (
              <div key={sector.name} className="flex items-center justify-between">
                <span className="text-gray-900 dark:text-white font-medium">
                  {sector.name}
                </span>
                <div className="flex items-center space-x-2">
                  <span className={`font-semibold ${sector.color}`}>
                    {sector.change > 0 ? '+' : ''}{sector.change.toFixed(2)}%
                  </span>
                  {sector.change > 0 ? (
                    <TrendingUp className="h-4 w-4 text-green-500" />
                  ) : (
                    <TrendingDown className="h-4 w-4 text-red-500" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Global Markets */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Global Markets
          </h3>
          <div className="space-y-4">
            {globalMarkets.map((market) => (
              <div key={market.name} className="flex items-center justify-between">
                <div>
                  <div className="font-medium text-gray-900 dark:text-white">
                    {market.name}
                  </div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">
                    {market.value}
                  </div>
                </div>
                <div className={`flex items-center space-x-1 ${
                  market.positive ? 'text-green-600' : 'text-red-600'
                }`}>
                  <span className="font-semibold">
                    {market.change}
                  </span>
                  {market.positive ? (
                    <TrendingUp className="h-4 w-4" />
                  ) : (
                    <TrendingDown className="h-4 w-4" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Market Heat Map */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Market Heat Map
        </h3>
        <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2">
          {[
            { name: 'RELIANCE', change: 2.45 },
            { name: 'TCS', change: 1.78 },
            { name: 'HDFCBANK', change: -0.95 },
            { name: 'INFY', change: 2.34 },
            { name: 'ICICIBANK', change: -1.23 },
            { name: 'HINDUNILVR', change: 0.87 },
            { name: 'SBIN', change: -2.15 },
            { name: 'BHARTIARTL', change: 1.45 },
            { name: 'ITC', change: 0.65 },
            { name: 'KOTAKBANK', change: -0.78 },
            { name: 'LT', change: 1.98 },
            { name: 'ASIANPAINT', change: -1.34 },
            { name: 'MARUTI', change: 2.67 },
            { name: 'HCLTECH', change: 1.23 },
            { name: 'AXISBANK', change: -1.67 },
            { name: 'TITAN', change: 3.45 },
          ].map((stock) => (
            <div
              key={stock.name}
              className={`p-3 rounded-lg text-center text-xs font-medium ${
                stock.change > 0
                  ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                  : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
              }`}
            >
              <div className="font-semibold">{stock.name}</div>
              <div className="text-xs mt-1">
                {stock.change > 0 ? '+' : ''}{stock.change.toFixed(2)}%
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MarketOverview;