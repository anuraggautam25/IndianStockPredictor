import React from 'react';

const PredictionChart: React.FC = () => {
  const data = [
    { month: 'Jan', value: 25400 },
    { month: 'Feb', value: 26200 },
    { month: 'Mar', value: 25800 },
    { month: 'Apr', value: 27100 },
    { month: 'May', value: 26800 },
    { month: 'Jun', value: 28200 },
    { month: 'Jul', value: 29100 },
    { month: 'Aug', value: 28600 },
  ];

  const maxValue = Math.max(...data.map(d => d.value));
  const minValue = Math.min(...data.map(d => d.value));

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-emerald-500 rounded-full"></div>
            <span className="text-sm text-gray-600 dark:text-gray-400">Nifty 50</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            <span className="text-sm text-gray-600 dark:text-gray-400">Predicted</span>
          </div>
        </div>
        <div className="text-sm text-gray-500 dark:text-gray-400">
          Last 8 months
        </div>
      </div>

      <div className="relative h-64">
        <svg viewBox="0 0 400 200" className="w-full h-full">
          {/* Grid lines */}
          {[0, 1, 2, 3, 4].map((i) => (
            <line
              key={i}
              x1="0"
              y1={i * 40 + 20}
              x2="400"
              y2={i * 40 + 20}
              stroke="currentColor"
              className="text-gray-200 dark:text-gray-700"
              strokeWidth="0.5"
            />
          ))}

          {/* Chart line */}
          <polyline
            points={data
              .map((d, i) => {
                const x = (i * 400) / (data.length - 1);
                const y = 180 - ((d.value - minValue) / (maxValue - minValue)) * 160;
                return `${x},${y}`;
              })
              .join(' ')}
            fill="none"
            stroke="currentColor"
            className="text-emerald-500"
            strokeWidth="2"
          />

          {/* Data points */}
          {data.map((d, i) => {
            const x = (i * 400) / (data.length - 1);
            const y = 180 - ((d.value - minValue) / (maxValue - minValue)) * 160;
            return (
              <circle
                key={i}
                cx={x}
                cy={y}
                r="4"
                fill="currentColor"
                className="text-emerald-500"
              />
            );
          })}
        </svg>

        {/* X-axis labels */}
        <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-gray-500 dark:text-gray-400">
          {data.map((d, i) => (
            <span key={i}>{d.month}</span>
          ))}
        </div>

        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 bottom-0 flex flex-col justify-between text-xs text-gray-500 dark:text-gray-400">
          {[maxValue, (maxValue + minValue) / 2, minValue].map((value, i) => (
            <span key={i}>{Math.round(value).toLocaleString()}</span>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <div className="text-center">
          <div className="text-lg font-semibold text-green-600">+12.5%</div>
          <div className="text-sm text-gray-500 dark:text-gray-400">YTD Growth</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-semibold text-blue-600">29,500</div>
          <div className="text-sm text-gray-500 dark:text-gray-400">Predicted Target</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-semibold text-emerald-600">85%</div>
          <div className="text-sm text-gray-500 dark:text-gray-400">Confidence</div>
        </div>
      </div>
    </div>
  );
};

export default PredictionChart;