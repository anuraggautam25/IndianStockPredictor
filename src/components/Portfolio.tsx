import React, { useState } from 'react';
import { PlusCircle, TrendingUp, TrendingDown, Pi as Pie, BarChart3 } from 'lucide-react';
import AddStockModal from './AddStockModal';

const Portfolio: React.FC = () => {
  const [holdings, setHoldings] = useState([
    {
      symbol: 'RELIANCE',
      name: 'Reliance Industries',
      quantity: 50,
      avgPrice: 2650,
      currentPrice: 2856,
      investedValue: 132500,
      currentValue: 142800,
      pnl: 10300,
      pnlPercent: 7.77
    },
    {
      symbol: 'TCS',
      name: 'Tata Consultancy Services',
      quantity: 25,
      avgPrice: 3850,
      currentPrice: 3986,
      investedValue: 96250,
      currentValue: 99650,
      pnl: 3400,
      pnlPercent: 3.53
    },
    {
      symbol: 'INFY',
      name: 'Infosys Limited',
      quantity: 75,
      avgPrice: 1456,
      currentPrice: 1532,
      investedValue: 109200,
      currentValue: 114900,
      pnl: 5700,
      pnlPercent: 5.22
    },
    {
      symbol: 'HDFCBANK',
      name: 'HDFC Bank',
      quantity: 40,
      avgPrice: 1750,
      currentPrice: 1698,
      investedValue: 70000,
      currentValue: 67920,
      pnl: -2080,
      pnlPercent: -2.97
    }
  ]);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);

  const handleAddStock = (newStock: {
    symbol: string;
    name: string;
    quantity: number;
    avgPrice: number;
    currentPrice: number;
  }) => {
    const investedValue = newStock.quantity * newStock.avgPrice;
    const currentValue = newStock.quantity * newStock.currentPrice;
    const pnl = currentValue - investedValue;
    const pnlPercent = (pnl / investedValue) * 100;

    const newHolding = {
      symbol: newStock.symbol,
      name: newStock.name,
      quantity: newStock.quantity,
      avgPrice: newStock.avgPrice,
      currentPrice: newStock.currentPrice,
      investedValue,
      currentValue,
      pnl,
      pnlPercent
    };

    setHoldings(prev => [...prev, newHolding]);
  };
  const totalInvested = holdings.reduce((sum, holding) => sum + holding.investedValue, 0);
  const totalCurrent = holdings.reduce((sum, holding) => sum + holding.currentValue, 0);
  const totalPnL = totalCurrent - totalInvested;
  const totalPnLPercent = (totalPnL / totalInvested) * 100;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          My Portfolio
        </h2>
        
        <button 
          onClick={() => setIsAddModalOpen(true)}
          className="flex items-center space-x-2 bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded-lg transition-colors"
        >
          <PlusCircle className="h-4 w-4" />
          <span>Add Stock</span>
        </button>
      </div>

      {/* Portfolio Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-3 mb-2">
            <BarChart3 className="h-5 w-5 text-blue-500" />
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">
              Total Invested
            </h3>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            ₹{totalInvested.toLocaleString()}
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-3 mb-2">
            <Pie className="h-5 w-5 text-emerald-500" />
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">
              Current Value
            </h3>
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            ₹{totalCurrent.toLocaleString()}
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-3 mb-2">
            {totalPnL >= 0 ? (
              <TrendingUp className="h-5 w-5 text-green-500" />
            ) : (
              <TrendingDown className="h-5 w-5 text-red-500" />
            )}
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">
              Total P&L
            </h3>
          </div>
          <div className={`text-2xl font-bold ${
            totalPnL >= 0 ? 'text-green-600' : 'text-red-600'
          }`}>
            {totalPnL >= 0 ? '+' : ''}₹{totalPnL.toLocaleString()}
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-3 mb-2">
            {totalPnLPercent >= 0 ? (
              <TrendingUp className="h-5 w-5 text-green-500" />
            ) : (
              <TrendingDown className="h-5 w-5 text-red-500" />
            )}
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400">
              Return %
            </h3>
          </div>
          <div className={`text-2xl font-bold ${
            totalPnLPercent >= 0 ? 'text-green-600' : 'text-red-600'
          }`}>
            {totalPnLPercent >= 0 ? '+' : ''}{totalPnLPercent.toFixed(2)}%
          </div>
        </div>
      </div>

      {/* Holdings Table */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Holdings
          </h3>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Stock
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Qty
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Avg Price
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Current Price
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Invested
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Current Value
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  P&L
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {holdings.map((holding) => (
                <tr key={holding.symbol} className="hover:bg-gray-50 dark:hover:bg-gray-750">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div>
                      <div className="text-sm font-medium text-gray-900 dark:text-white">
                        {holding.symbol}
                      </div>
                      <div className="text-sm text-gray-500 dark:text-gray-400">
                        {holding.name}
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    {holding.quantity}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    ₹{holding.avgPrice.toFixed(2)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    ₹{holding.currentPrice.toFixed(2)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    ₹{holding.investedValue.toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    ₹{holding.currentValue.toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className={`text-sm font-medium ${
                      holding.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {holding.pnl >= 0 ? '+' : ''}₹{holding.pnl.toLocaleString()}
                    </div>
                    <div className={`text-xs ${
                      holding.pnl >= 0 ? 'text-green-500' : 'text-red-500'
                    }`}>
                      ({holding.pnl >= 0 ? '+' : ''}{holding.pnlPercent.toFixed(2)}%)
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <AddStockModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onAddStock={handleAddStock}
      />
    </div>
  );
};

export default Portfolio;