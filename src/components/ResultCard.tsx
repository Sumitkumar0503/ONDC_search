'use client';

import { useState } from 'react';
import { ChevronDown, ChevronUp, Target, Zap, Search, TrendingUp } from 'lucide-react';
import { motion } from 'framer-motion';

interface ResultCardProps {
  result: {
    name: string;
    price_value: number;
    price_currency: string;
    veg_non_veg_final: string;
    provider_name: string;
    category_id: string;
    _scores: {
      final: number;
      cosine: number;
      lexical: number;
      constraint: number;
      distance: number;
      business: number;
      reranker: number;
    };
    _snippet: string;
  };
}

export default function ResultCard({ result }: ResultCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const confidenceScore = Math.round(result._scores.final * 100);
  const isDietVeg = result.veg_non_veg_final === 'veg';
  const isDietNonVeg = result.veg_non_veg_final === 'non_veg';

  // Determine confidence level styling
  const getConfidenceStyle = () => {
    if (confidenceScore >= 90) return 'border-green-300 bg-green-50';
    if (confidenceScore >= 70) return 'border-orange-300 bg-orange-50';
    return 'border-gray-300 bg-gray-50';
  };

  const getConfidenceBadgeStyle = () => {
    if (confidenceScore >= 90) return 'bg-green-500 text-white';
    if (confidenceScore >= 70) return 'bg-orange-500 text-white';
    return 'bg-gray-500 text-white';
  };

  const scoreItems = [
    { 
      label: 'Semantic', 
      value: result._scores.cosine, 
      icon: Zap, 
      description: 'How well the meaning matches your query'
    },
    { 
      label: 'Keyword', 
      value: result._scores.lexical, 
      icon: Search, 
      description: 'Direct keyword overlap with your search'
    },
    { 
      label: 'Constraints', 
      value: result._scores.constraint, 
      icon: Target, 
      description: 'Matches your filters (diet, price, etc.)'
    },
    { 
      label: 'Relevance', 
      value: result._scores.reranker, 
      icon: TrendingUp, 
      description: 'AI-powered relevance assessment'
    },
  ];

  return (
    <motion.div
      whileHover={{ y: -2, boxShadow: "0 10px 25px rgba(0, 0, 0, 0.15)" }}
      className={`bg-white rounded-xl card-shadow hover:card-shadow-hover transition-all duration-200 p-6 border ${getConfidenceStyle()}`}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h3 className="font-semibold text-base text-gray-900 mb-1 leading-tight">
            {result.name}
          </h3>
          <p className="text-sm text-gray-500 mb-2">
            {result.provider_name} • {result.category_id}
          </p>
        </div>
        
        {/* Confidence Badge */}
        <div className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${getConfidenceBadgeStyle()}`}>
          <Target className="h-3 w-3" />
          {confidenceScore}%
        </div>
      </div>

      {/* Price and Diet */}
      <div className="flex items-center justify-between mb-4">
        <div className="font-semibold text-lg text-gray-900">
          {result.price_currency}{result.price_value?.toFixed(2) || 'N/A'}
        </div>
        
        {/* Diet Indicator */}
        {(isDietVeg || isDietNonVeg) && (
          <div className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
            isDietVeg 
              ? 'bg-green-100 text-green-700' 
              : 'bg-red-100 text-red-700'
          }`}>
            <div className={`w-2 h-2 rounded-full ${
              isDietVeg ? 'bg-green-500' : 'bg-red-500'
            }`} />
            {isDietVeg ? 'Veg' : 'Non-Veg'}
          </div>
        )}
      </div>

      {/* Match Explanation Toggle */}
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between w-full text-sm text-gray-600 hover:text-gray-800 transition-colors duration-200 mb-2"
      >
        <span className="font-medium">Why this matched?</span>
        {isExpanded ? (
          <ChevronUp className="h-4 w-4" />
        ) : (
          <ChevronDown className="h-4 w-4" />
        )}
      </motion.button>

      {/* Expanded Score Details */}
      <motion.div
        initial={false}
        animate={{
          height: isExpanded ? 'auto' : 0,
          opacity: isExpanded ? 1 : 0,
        }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
        className="overflow-hidden"
      >
        <div className="space-y-3 pt-2">
          {scoreItems.map((item, index) => {
            const IconComponent = item.icon;
            const percentage = Math.round(item.value * 100);
            
            return (
              <div key={index} className="flex items-center gap-3">
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <IconComponent className="h-4 w-4 text-gray-400 flex-shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-gray-700">
                        {item.label}
                      </span>
                      <span className="text-xs text-gray-500">
                        {percentage}%
                      </span>
                    </div>
                    {/* Progress Bar */}
                    <div className="w-full bg-gray-200 rounded-full h-1.5">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${percentage}%` }}
                        transition={{ delay: index * 0.1, duration: 0.5 }}
                        className={`h-1.5 rounded-full ${
                          percentage >= 70 ? 'bg-green-500' : 
                          percentage >= 40 ? 'bg-yellow-500' : 'bg-gray-400'
                        }`}
                      />
                    </div>
                    <p className="text-xs text-gray-400 mt-1">
                      {item.description}
                    </p>
                  </div>
                </div>
              </div>
            );
          })}

          {/* Snippet Preview */}
          {result._snippet && (
            <div className="mt-4 pt-3 border-t border-gray-100">
              <p className="text-xs text-gray-500 mb-1 font-medium">Content Preview:</p>
              <p className="text-xs text-gray-600 bg-gray-50 p-2 rounded leading-relaxed">
                {result._snippet}
              </p>
            </div>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
}