import React, { useState } from 'react';
import { Search, Filter, TrendingUp, TrendingDown } from 'lucide-react';
import StockCard from './StockCard';
import { popularStocks } from '../data/stockData';

const StockSearch: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterBy, setFilterBy] = useState('all');

  const filteredStocks = popularStocks.filter((stock) => {
    const matchesSearch = stock.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      stock.symbol.toLowerCase().includes(searchTerm.toLowerCase());

    if (filterBy === 'all') return matchesSearch;
    if (filterBy === 'gainers') return matchesSearch && stock.change > 0;
    if (filterBy === 'losers') return matchesSearch && stock.change < 0;
    if (filterBy === 'buy') return matchesSearch && (stock.recommendation === 'Buy' || stock.recommendation === 'Strong Buy');

    return matchesSearch;
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          Stock Search & Analysis
        </h2>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
        <div className="flex flex-col md:flex-row gap-4 mb-6">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search stocks by name or symbol..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>

          <div className="relative">
            <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <select
              value={filterBy}
              onChange={(e) => setFilterBy(e.target.value)}
              className="pl-10 pr-8 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white appearance-none"
            >
              <option value="all">All Stocks</option>
              <option value="gainers">Top Gainers</option>
              <option value="losers">Top Losers</option>
              <option value="buy">Buy Recommendations</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredStocks.length > 0 ? (
            filteredStocks.map((stock) => (
              <StockCard key={stock.symbol} stock={stock} />
            ))
          ) : (
            <div className="col-span-full text-center py-8 text-gray-500 dark:text-gray-400">
              No stocks found matching your criteria.
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <div className="flex items-center space-x-3 mb-4">
            <TrendingUp className="h-6 w-6 text-green-500" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Top Gainers
            </h3>
          </div>
          <div className="space-y-3">
            {popularStocks
              .filter(stock => stock.change > 0)
              .sort((a, b) => b.changePercent - a.changePercent)
              .slice(0, 3)
              .map((stock) => (
                <div key={stock.symbol} className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-gray-900 dark:text-white">
                      {stock.symbol}
                    </div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      ₹{stock.price.toFixed(2)}
                    </div>
                  </div>
                  <div className="text-green-600 font-semibold">
                    +{stock.changePercent.toFixed(2)}%
                  </div>
                </div>
              ))}
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <div className="flex items-center space-x-3 mb-4">
            <TrendingDown className="h-6 w-6 text-red-500" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Top Losers
            </h3>
          </div>
          <div className="space-y-3">
            {popularStocks
              .filter(stock => stock.change < 0)
              .sort((a, b) => a.changePercent - b.changePercent)
              .slice(0, 3)
              .map((stock) => (
                <div key={stock.symbol} className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-gray-900 dark:text-white">
                      {stock.symbol}
                    </div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      ₹{stock.price.toFixed(2)}
                    </div>
                  </div>
                  <div className="text-red-600 font-semibold">
                    {stock.changePercent.toFixed(2)}%
                  </div>
                </div>
              ))}
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
          <div className="flex items-center space-x-3 mb-4">
            <div className="h-6 w-6 bg-gradient-to-r from-emerald-500 to-blue-600 rounded"></div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              AI Picks
            </h3>
          </div>
          <div className="space-y-3">
            {popularStocks
              .filter(stock => stock.recommendation === 'Strong Buy')
              .slice(0, 3)
              .map((stock) => (
                <div key={stock.symbol} className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-gray-900 dark:text-white">
                      {stock.symbol}
                    </div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      ₹{stock.price.toFixed(2)}
                    </div>
                  </div>
                  <div className="px-2 py-1 bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-300 rounded text-xs font-medium">
                    Strong Buy
                  </div>
                </div>
              ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StockSearch;