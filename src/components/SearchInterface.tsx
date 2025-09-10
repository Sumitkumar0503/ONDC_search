'use client';

import { useState } from 'react';
import { Search, Pizza, Gift, Heart } from 'lucide-react';
import { motion } from 'framer-motion';
import ResultCard from './ResultCard';
import { searchProducts } from '@/lib/api';

interface SearchResult {
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
}

interface SearchResponse {
  q: string;
  results: SearchResult[];
  no_results?: boolean;
  reason?: string;
  suggestions?: string[];
  timings: {
    total_ms: number;
  };
}

export default function SearchInterface() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null);
  const [isDemoMode, setIsDemoMode] = useState(false);

  const sampleQueries = [
    { text: 'quick dinner ideas', icon: Pizza },
    { text: 'birthday treats', icon: Gift },
    { text: 'healthy options', icon: Heart },
  ];

  const handleSearch = async (searchQuery: string = query) => {
    if (!searchQuery.trim()) return;
    
    setLoading(true);
    try {
      const response = await searchProducts(searchQuery);
      setSearchResponse(response);
      setResults(response.results || []);
      
      // Check if we're in demo mode (when API is not available)
      const isDemo = response.results?.some(result => 
        result.provider_name?.includes('Demo') || 
        result.name?.includes('Sample')
      );
      setIsDemoMode(isDemo || false);
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
      setSearchResponse(null);
      setIsDemoMode(true);
    } finally {
      setLoading(false);
    }
  };

  const handleSampleQuery = (sampleQuery: string) => {
    setQuery(sampleQuery);
    handleSearch(sampleQuery);
  };

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="hero-gradient rounded-3xl p-12 text-center"
      >
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Experience AI-Powered Search
        </h1>
        <p className="text-lg text-gray-600 mb-8">
          Find products by meaning, not just keywords
        </p>

        {/* Search Bar */}
        <div className="max-w-2xl mx-auto mb-6">
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
              <Search className="h-5 w-5 text-gray-400" />
            </div>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Try: 'healthy snacks for movie night' or 'gifts under ₹500'"
              className="block w-full pl-12 pr-4 py-4 text-lg border-0 rounded-2xl card-shadow focus:ring-2 focus:ring-green-500 focus:shadow-lg transition-all duration-200 placeholder-gray-400"
            />
          </div>
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => handleSearch()}
            disabled={loading}
            className="mt-4 bg-green-600 hover:bg-green-700 text-white font-medium py-3 px-8 rounded-xl transition-colors duration-200 disabled:opacity-50"
          >
            {loading ? 'Searching...' : 'Search'}
          </motion.button>
        </div>

        {/* Sample Query Chips */}
        <div className="flex flex-wrap justify-center gap-3">
          {sampleQueries.map((sample, index) => {
            const IconComponent = sample.icon;
            return (
              <motion.button
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.1 * index, duration: 0.3 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => handleSampleQuery(sample.text)}
                className="flex items-center gap-2 bg-white hover:bg-gray-50 text-gray-700 px-4 py-2 rounded-full card-shadow hover:card-shadow-hover transition-all duration-200"
              >
                <IconComponent className="h-4 w-4" />
                <span className="text-sm font-medium">{sample.text}</span>
              </motion.button>
            );
          })}
        </div>
      </motion.div>

      {/* Results Section */}
      {searchResponse && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4 }}
        >
          {results.length > 0 ? (
            <>
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h2 className="text-2xl font-semibold text-gray-900">
                    Search Results
                  </h2>
                  {isDemoMode && (
                    <div className="flex items-center gap-2 mt-1">
                      <div className="bg-orange-100 text-orange-700 px-2 py-1 rounded text-xs font-medium">
                        DEMO MODE
                      </div>
                      <span className="text-xs text-gray-500">
                        API not available - showing sample data
                      </span>
                    </div>
                  )}
                </div>
                <div className="text-sm text-gray-500">
                  Found {results.length} results in {Math.round(searchResponse.timings.total_ms)}ms
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {results.map((result, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 * index, duration: 0.4 }}
                  >
                    <ResultCard result={result} />
                  </motion.div>
                ))}
              </div>
            </>
          ) : (
            <div className="text-center py-12">
              <div className="text-gray-500 mb-4">
                <Search className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <h3 className="text-lg font-medium mb-2">No Results Found</h3>
                <p>{searchResponse.reason || 'No matches found. Try another query.'}</p>
              </div>
              {searchResponse.suggestions && searchResponse.suggestions.length > 0 && (
                <div className="mt-6">
                  <p className="text-sm text-gray-600 mb-3">Try these suggestions:</p>
                  <div className="flex flex-wrap justify-center gap-2">
                    {searchResponse.suggestions.slice(0, 5).map((suggestion, index) => (
                      <button
                        key={index}
                        onClick={() => handleSampleQuery(suggestion)}
                        className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-1 rounded-full text-sm transition-colors duration-200"
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}