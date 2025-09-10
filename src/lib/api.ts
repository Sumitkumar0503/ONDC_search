const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8080';

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

  const response = await fetch(`${API_BASE_URL}/search?${params}`, {
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