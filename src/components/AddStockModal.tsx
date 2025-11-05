import React, { useState } from 'react';
import { X, Search, Plus } from 'lucide-react';
import { popularStocks } from '../data/stockData';

interface AddStockModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAddStock: (stock: {
    symbol: string;
    name: string;
    quantity: number;
    avgPrice: number;
    currentPrice: number;
  }) => void;
}

const AddStockModal: React.FC<AddStockModalProps> = ({ isOpen, onClose, onAddStock }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedStock, setSelectedStock] = useState<any>(null);
  const [quantity, setQuantity] = useState('');
  const [avgPrice, setAvgPrice] = useState('');
  const [errors, setErrors] = useState<{ [key: string]: string }>({});

  const filteredStocks = popularStocks.filter(stock =>
    stock.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
    stock.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const validateForm = () => {
    const newErrors: { [key: string]: string } = {};

    if (!selectedStock) {
      newErrors.stock = 'Please select a stock';
    }
    if (!quantity || parseInt(quantity) <= 0) {
      newErrors.quantity = 'Please enter a valid quantity';
    }
    if (!avgPrice || parseFloat(avgPrice) <= 0) {
      newErrors.avgPrice = 'Please enter a valid average price';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) return;

    onAddStock({
      symbol: selectedStock.symbol,
      name: selectedStock.name,
      quantity: parseInt(quantity),
      avgPrice: parseFloat(avgPrice),
      currentPrice: selectedStock.price,
    });

    // Reset form
    setSelectedStock(null);
    setQuantity('');
    setAvgPrice('');
    setSearchTerm('');
    setErrors({});
    onClose();
  };

  const handleStockSelect = (stock: any) => {
    setSelectedStock(stock);
    setAvgPrice(stock.price.toString());
    setSearchTerm('');
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl w-full max-w-md max-h-[90vh] overflow-hidden">
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">
            Add Stock to Portfolio
          </h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            <X className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* Stock Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Select Stock
            </label>
            {selectedStock ? (
              <div className="flex items-center justify-between p-3 bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 rounded-lg">
                <div>
                  <div className="font-semibold text-gray-900 dark:text-white">
                    {selectedStock.symbol}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    {selectedStock.name}
                  </div>
                  <div className="text-sm text-emerald-600 dark:text-emerald-400">
                    Current: ₹{selectedStock.price.toFixed(2)}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setSelectedStock(null)}
                  className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search stocks..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  />
                </div>
                {searchTerm && (
                  <div className="max-h-40 overflow-y-auto border border-gray-200 dark:border-gray-600 rounded-lg">
                    {filteredStocks.slice(0, 5).map((stock) => (
                      <button
                        key={stock.symbol}
                        type="button"
                        onClick={() => handleStockSelect(stock)}
                        className="w-full p-3 text-left hover:bg-gray-50 dark:hover:bg-gray-700 border-b border-gray-100 dark:border-gray-600 last:border-b-0"
                      >
                        <div className="font-medium text-gray-900 dark:text-white">
                          {stock.symbol}
                        </div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">
                          {stock.name} - ₹{stock.price.toFixed(2)}
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
            {errors.stock && (
              <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.stock}</p>
            )}
          </div>

          {/* Quantity */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Quantity
            </label>
            <input
              type="number"
              value={quantity}
              onChange={(e) => setQuantity(e.target.value)}
              placeholder="Enter quantity"
              min="1"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
            {errors.quantity && (
              <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.quantity}</p>
            )}
          </div>

          {/* Average Price */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Average Price (₹)
            </label>
            <input
              type="number"
              value={avgPrice}
              onChange={(e) => setAvgPrice(e.target.value)}
              placeholder="Enter average price"
              step="0.01"
              min="0.01"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
            {errors.avgPrice && (
              <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.avgPrice}</p>
            )}
          </div>

          {/* Investment Summary */}
          {selectedStock && quantity && avgPrice && (
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                Investment Summary
              </h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Total Investment:</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    ₹{(parseInt(quantity || '0') * parseFloat(avgPrice || '0')).toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Current Value:</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    ₹{(parseInt(quantity || '0') * selectedStock.price).toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Unrealized P&L:</span>
                  <span className={`font-medium ${
                    (parseInt(quantity || '0') * selectedStock.price) - (parseInt(quantity || '0') * parseFloat(avgPrice || '0')) >= 0
                      ? 'text-green-600' : 'text-red-600'
                  }`}>
                    ₹{((parseInt(quantity || '0') * selectedStock.price) - (parseInt(quantity || '0') * parseFloat(avgPrice || '0'))).toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex space-x-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 flex items-center justify-center space-x-2 bg-emerald-600 hover:bg-emerald-700 text-white px-4 py-2 rounded-lg transition-colors"
            >
              <Plus className="h-4 w-4" />
              <span>Add Stock</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AddStockModal;