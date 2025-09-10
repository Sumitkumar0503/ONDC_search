'use client';

import SearchInterface from '@/components/SearchInterface';

export default function Home() {
  return (
    <div className="min-h-screen bg-white">
      {/* Top Banner */}
      <div className="bg-green-50 border-b border-green-100 py-3">
        <div className="max-w-6xl mx-auto px-4 text-center">
          <p className="text-sm text-green-800 font-medium">
            Vector Search Demo - See how AI understands your intent, not just keywords
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 py-8">
        <SearchInterface />
      </div>
    </div>
  );
}
