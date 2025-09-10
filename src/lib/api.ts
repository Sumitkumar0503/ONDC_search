// Environment-specific API configuration
const getAPIBaseURL = () => {
  // In production, we'll either use a deployed backend or a mock/demo mode
  if (typeof window !== 'undefined' && window.location.hostname !== 'localhost') {
    // Production environment - you'll need to set this in Vercel
    return process.env.NEXT_PUBLIC_API_BASE_URL || 'https://your-api-domain.com';
  }
  // Development environment
  return process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8080';
};

const API_BASE_URL = getAPIBaseURL();

export interface SearchResult {
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
  [key: string]: string | number | boolean | object | null | undefined;
}

export interface SearchResponse {
  q: string;
  normalized_q?: string;
  used_q?: string;
  intent: {
    must_have_keywords: string[];
    diet: string | null;
    numeric_filters: Array<{
      field: string;
      op: string;
      value: number;
    }>;
    pincode: string | null;
  };
  top_k: number;
  top_n: number;
  results: SearchResult[];
  no_results?: boolean;
  reason?: string;
  domains_present?: string[];
  suggestions?: string[];
  did_autocorrect?: boolean;
  corrected_q?: string;
  timings: {
    intent_ms: number;
    vector_retrieval_ms: number;
    prefilter_ms: number;
    autocorrect_ms: number;
    rank_ms: number;
    total_ms: number;
  };
}

// Mock data for demo purposes when API is not available
const getMockSearchResponse = (query: string): SearchResponse => ({
  q: query,
  normalized_q: query.toLowerCase(),
  used_q: query.toLowerCase(),
  intent: {
    must_have_keywords: [],
    diet: null,
    numeric_filters: [],
    pincode: null,
  },
  top_k: 10,
  top_n: 10,
  results: [
    {
      name: "Sample Pizza Margherita",
      price_value: 299,
      price_currency: "₹",
      veg_non_veg_final: "veg",
      provider_name: "Demo Restaurant",
      category_id: "food-pizza",
      _scores: {
        final: 0.89,
        cosine: 0.85,
        lexical: 0.78,
        constraint: 1.0,
        distance: 0.0,
        business: 0.88,
        reranker: 0.91,
      },
      _snippet: `A classic vegetarian pizza with fresh tomatoes, mozzarella cheese, and basil. Perfect for ${query}`,
    },
    {
      name: "Chocolate Birthday Cake",
      price_value: 450,
      price_currency: "₹",
      veg_non_veg_final: "veg",
      provider_name: "Sweet Treats Bakery",
      category_id: "dessert-cake",
      _scores: {
        final: 0.82,
        cosine: 0.80,
        lexical: 0.70,
        constraint: 0.9,
        distance: 0.0,
        business: 0.85,
        reranker: 0.87,
      },
      _snippet: `Rich chocolate cake perfect for celebrations. Great match for "${query}" with premium ingredients.`,
    },
  ],
  timings: {
    intent_ms: 45,
    vector_retrieval_ms: 120,
    prefilter_ms: 15,
    autocorrect_ms: 0,
    rank_ms: 80,
    total_ms: 260,
  },
});

export async function searchProducts(
  query: string,
  options: {
    pincode?: string;
    top_k?: number;
    top_n?: number;
    pretty?: boolean;
  } = {}
): Promise<SearchResponse> {
  const params = new URLSearchParams({
    q: query,
    top_k: (options.top_k || 10).toString(),
    top_n: (options.top_n || 10).toString(),
    ...(options.pincode && { pincode: options.pincode }),
    ...(options.pretty && { pretty: '1' }),
  });

  try {
    const response = await fetch(`${API_BASE_URL}/search?${params}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
      // Add timeout to prevent hanging
      signal: AbortSignal.timeout(10000), // 10 second timeout
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  } catch (error) {
    console.warn('API not available, using demo data:', error);
    // Return mock data for demo purposes
    return getMockSearchResponse(query);
  }
}

export async function getSuggestions(query: string): Promise<{ q: string; suggestions: string[] }> {
  const params = new URLSearchParams({ q: query });
  
  const response = await fetch(`${API_BASE_URL}/suggest?${params}`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

export async function ingestData(folder: string = './data/inbox'): Promise<{ indexed_items: number }> {
  const response = await fetch(`${API_BASE_URL}/ingest`, {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ folder }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}